##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

# Adapted From the Following:
# Github: https://github.com/nityasagarjena/PaiNN-model/blob/main/PaiNN/model.py
# Paper: https://arxiv.org/pdf/2102.03150

import numpy as np
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as geom_nn
from torch.utils.checkpoint import checkpoint
import torch_scatter

from .Base import Base


class CHIRALStack(Base):
    """
    Generates angles, distances, to/from indices, radial basis
    functions and spherical basis functions for learning.
    """

    def __init__(
        self,
        # edge_dim: int,   # To-Do: Add edge_features
        num_radial: int,
        radius: float,
        *args,
        **kwargs
    ):
        # self.edge_dim = edge_dim
        self.num_radial = num_radial
        self.radius = radius

        super().__init__(*args, **kwargs)

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(self.get_conv(self.input_dim, self.hidden_dim))
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())
        self.chiral_updates = torch.nn.ModuleList()
        self.chiral_updates.append(self.get_update(self.hidden_dim))
        
    def get_update(self, node_size):
        chiral_update = ChiralUpdate(node_size=node_size)
        return geom_nn.Sequential("x, xc", [(chiral_update, "x, xc -> x")])

    def get_conv(self, input_dim, output_dim, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        self_inter = ChiralMessage(node_size=118)

        if not last_layer:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist, triplet_index",
                [
                    (self_inter, "x, xc, v, edge_index, triplet_index, pos -> x, xc, v"),
                    (lambda x, xc, pos: [x, xc, pos], "x, xc, pos -> x, xc, pos"),
                ],
            )
        else:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist",
                [
                    (self_inter, "x, xc, v, edge_index, triplet_index, pos -> x, xc, v"),
                    (lambda x, xc, pos: [x, xc, pos], "x, xc, pos -> x, xc, pos"),
                ],
            )

    def forward(self, data):
        data, conv_args = self._conv_args(
            data
        )  # Added v to data here (necessary for PAINN Stack)
        x = data.x
        xc = data.xc
        v = data.v
        pos = data.pos

        ### encoder part ####
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                c, xc, v, pos = conv(x=x, xc=xc, v=v, pos=pos, **conv_args)  # Added v here
            else:
                c, xc, v, pos = checkpoint(  # Added v here
                    conv, use_reentrant=False, x=x, v=v, pos=pos, **conv_args
                )
            x = self.activation_function(feat_layer(c))
        # Add chirality
        for chiral_update in self.chiral_updates:
            x = chiral_update(x, xc)

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if data.batch is None:
            x_graph = x.mean(dim=0, keepdim=True)
        else:
            x_graph = geom_nn.global_mean_pool(x, data.batch.to(x.device))
        outputs = []
        outputs_var = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = self.graph_shared(x_graph)
                output_head = headloc(x_graph_head)
                outputs.append(output_head[:, :head_dim])
                outputs_var.append(output_head[:, head_dim:] ** 2)
            else:
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        c, v, pos = conv(x=x, v=v, pos=pos, **conv_args)
                        c = batch_norm(c)
                        x = self.activation_function(c)
                    x_node = x
                else:
                    x_node = headloc(x=x, batch=data.batch)
                outputs.append(x_node[:, :head_dim])
                outputs_var.append(x_node[:, head_dim:] ** 2)
        if self.var_output:
            return outputs, outputs_var
        return outputs
    
    def loss_ct(useless1, pred, value, useless2):
        # Reshape 'value' to [batch_size, 3] where each row is a one-hot encoding of the class
        value = value.view(-1, 3).long()

        # Convert the one-hot encoding into class indices using argmax
        value = torch.argmax(value, dim=1)

        # Apply cross-entropy loss
        # 'pred[0]' should have shape [batch_size, num_classes] and 'value' should have shape [batch_size]
        class_weights = torch.tensor([1.0, 50.0, 50.0])  # Adjust the weights based on the class distribution
        loss = F.cross_entropy(pred[0], value, weight=class_weights)
       
        # Calculate overall predictions
        predictions = torch.argmax(pred[0], dim=1)
        # Calculate accuracy for each class
        class_accuracies = []
        for i in range(3):  # Assuming 3 classes
            mask = (value == i)  # Identify samples of class 'i'
            if mask.sum() > 0:  # Avoid division by zero
                accuracy = (predictions[mask] == value[mask]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=pred[0].device)  # If no samples for this class, accuracy is 0
            class_accuracies.append(accuracy)
            print(f"Accuracy for class {i}: {accuracy.item()}")

        return loss, [loss]
    
    def loss(useless1, pred, value, useless2):
        # Extract the first tensor from the list
        pred_tensor = pred[0].view(-1)  # Flatten to match the shape of `value`
        
        # Compute the Mean Squared Error Loss
        # loss = F.mse_loss(pred_tensor, value)
        loss = F.l1_loss(pred_tensor, value)
    
        return loss, [loss]

    def _multihead(self):
        ############multiple heads/taks################
        # shared dense layers for heads with graph level output
        dim_sharedlayers = 0
        if "graph" in self.config_heads:
            denselayers = []
            dim_sharedlayers = self.config_heads["graph"]["dim_sharedlayers"]
            denselayers.append(nn.Linear(self.hidden_dim, dim_sharedlayers))
            denselayers.append(self.activation_function)
            for ishare in range(self.config_heads["graph"]["num_sharedlayers"] - 1):
                denselayers.append(nn.Linear(dim_sharedlayers, dim_sharedlayers))
                denselayers.append(self.activation_function)
            self.graph_shared = nn.Sequential(*denselayers)

        if "node" in self.config_heads:
            self.num_conv_layers_node = self.config_heads["node"]["num_headlayers"]
            self.hidden_dim_node = self.config_heads["node"]["dim_headlayers"]
            self._init_node_conv()

        inode_feature = 0
        for ihead in range(self.num_heads):
            # mlp for each head output
            if self.head_type[ihead] == "graph":
                num_head_hidden = self.config_heads["graph"]["num_headlayers"]
                dim_head_hidden = self.config_heads["graph"]["dim_headlayers"]
                denselayers = []
                denselayers.append(nn.Linear(dim_sharedlayers, dim_head_hidden[0]))
                denselayers.append(self.activation_function)
                for ilayer in range(num_head_hidden - 1):
                    denselayers.append(
                        nn.Linear(dim_head_hidden[ilayer], dim_head_hidden[ilayer + 1])
                    )
                    denselayers.append(self.activation_function)
                denselayers.append(
                    nn.Linear(
                        dim_head_hidden[-1],
                        self.head_dims[ihead] * (1 + self.var_output),
                    )
                )
                head_NN = nn.Sequential(*denselayers)
            elif self.head_type[ihead] == "node":
                self.node_NN_type = self.config_heads["node"]["type"]
                head_NN = nn.ModuleList()
                if self.node_NN_type == "mlp" or self.node_NN_type == "mlp_per_node":
                    self.num_mlp = 1 if self.node_NN_type == "mlp" else self.num_nodes
                    assert (
                        self.num_nodes is not None
                    ), "num_nodes must be positive integer for MLP"
                    # """if different graphs in the dataset have different size, one MLP is shared across all nodes """
                    head_NN = MLPNode(
                        self.hidden_dim,
                        self.head_dims[ihead] * (1 + self.var_output),
                        self.num_mlp,
                        self.hidden_dim_node,
                        self.config_heads["node"]["type"],
                        self.activation_function,
                    )
                elif self.node_NN_type == "conv":
                    for conv, batch_norm in zip(
                        self.convs_node_hidden, self.batch_norms_node_hidden
                    ):
                        head_NN.append(conv)
                        head_NN.append(batch_norm)
                    head_NN.append(self.convs_node_output[inode_feature])
                    head_NN.append(self.batch_norms_node_output[inode_feature])
                    inode_feature += 1
                else:
                    raise ValueError(
                        "Unknown head NN structure for node features"
                        + self.node_NN_type
                        + "; currently only support 'mlp', 'mlp_per_node' or 'conv' (can be set with config['NeuralNetwork']['Architecture']['output_heads']['node']['type'], e.g., ./examples/ci_multihead.json)"
                    )
            else:
                raise ValueError(
                    "Unknown head type"
                    + self.head_type[ihead]
                    + "; currently only support 'graph' or 'node'"
                )
            self.heads_NN.append(head_NN)

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "CHIRAl requires node positions (data.pos) to be set."

        # Calculate relative vectors and distances
        i, j = data.edge_index[0], data.edge_index[1]
        diff = data.pos[i] - data.pos[j]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        norm_diff = diff / dist.unsqueeze(-1)

        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(data.x.size(0), 3, data.x.size(1), device=data.x.device)
        data.v = v
        xc = torch.zeros_like(data.x)
        data.xc = xc

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),  # Shape: [2, num_edges]
            "diff": norm_diff,  # Shape: [num_edges, 3]
            "dist": dist,        # Shape: [num_edges]
        }
        
        # Ordered Triplet Indices
        triplet_index = []
        base_nodes = data.edge_index[0].unique()
        for base_node_id in base_nodes:
            # Get neighbors
            neighbors = data.edge_index[1][data.edge_index[0] == base_node_id]
            
            # Skip nodes with fewer than 3 neighbors
            if neighbors.size(0) < 3:
                continue
            
            # Generate all possible triplets of neighbors
            for triplet in itertools.combinations(neighbors.tolist(), 3):
                triplet = list(triplet)  # Convert from tuple to list for manipulation

                # Extract atomic numbers from one-hot encoding
                atomic_numbers_one_hot = data.x[triplet][:, :118]  # Assuming first 118 features are one-hot
                atomic_numbers = torch.argmax(atomic_numbers_one_hot, dim=1)  # Shape: [3]

                # Sort triplet based on atomic numbers (highest to lowest)
                highest_to_lowest_indices = torch.argsort(atomic_numbers, descending=True)
                ordered_triplet = [triplet[idx] for idx in highest_to_lowest_indices]

                # Create a full triplet including the base node
                triplet_full = [base_node_id] + ordered_triplet  # Shape: [4]

                # Append to the triplet list
                triplet_index.append(triplet_full)

            if triplet_index:
                # Convert list of triplets to a tensor
                triplet_index = torch.tensor(triplet_index, dtype=torch.long, device=data.edge_index.device)  # Shape: [num_triplets, 4]
            else:
                # Handle case with no triplets
                triplet_index = torch.empty((0, 4), dtype=torch.long, device=data.edge_index.device)

            data.triplet_index = triplet_index  # Shape: [num_triplets, 4]

        return data, conv_args
    

class ChiralMessage(torch.nn.Module):
    def __init__(self, node_size: int):
        super(ChiralMessage, self).__init__()
        self.node_size = node_size
        
        self.scalar_gate = nn.Sequential(torch.nn.Linear(2*node_size, node_size), torch.nn.SiLU(), torch.nn.Linear(node_size, 3*node_size))
        self.chiral_gate = nn.Sequential(torch.nn.Linear(2*node_size, node_size), torch.nn.SiLU(), torch.nn.Linear(node_size, node_size))
        self.update_V = torch.nn.Linear(node_size, node_size)
        
    def forward(self, node_scalar, node_chiral, node_vector, edge_index, triplet_index, pos):
        """Scalar / Vector Section"""
        # Compute scalar gates
        message_gate = self.scalar_gate(torch.cat((node_scalar[edge_index[:, 0]], node_scalar[edge_index[:, 1]]), dim=1))  # Shape: [num_edges, 2 * node_size]  -->  [num_edges, 3 * node_size]
        # To-Do: Filter based on distance? Worry about later
        gate_vector, gate_edge, messages_ss = torch.split(
            message_gate,
            self.node_size,
            dim=1,
        ) # Each shape: [num_edges, node_size]
        
        # Make messages
        messages_vv = gate_vector * node_vector[edge_index[:, 1]]
        messages_ev = gate_edge * (pos[edge_index[:, 1]] - pos[edge_index[:, 0]])
        # Aggregate
        message_ss = torch_scatter.scatter(messages_ss, edge_index[:, 0], dim=0, dim_size=node_scalar.size(0), reduce="add")
        message_vv = torch_scatter.scatter(messages_vv, edge_index[:, 0], dim=0, dim_size=node_vector.size(0), reduce="add")
        message_ev = torch_scatter.scatter(messages_ev, edge_index[:, 0], dim=0, dim_size=node_vector.size(0), reduce="add")

        # Self Cross-Message
        Vv = self.update_V(node_vector)
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        message_vs = node_scalar * Vv_norm
        """"""
        
        """Chiral Section"""
        if triplet_index.numel() > 0:
            # Create gates and messages from triplets
            base_nodes = triplet_index[:, 0]  # Shape: [num_triplets]
            triplet1 = triplet_index[:, 1]    # Shape: [num_triplets]
            triplet2 = triplet_index[:, 2]    # Shape: [num_triplets]
            triplet3 = triplet_index[:, 3]    # Shape: [num_triplets]
            
            # Compute chiral gates for each triplet node
            chiral_gate_t1 = self.chiral_gate(torch.cat((node_chiral[base_nodes], node_chiral[triplet1]), dim=1))  # Shape: [num_triplets, node_size]
            chiral_gate_t2 = self.chiral_gate(torch.cat((node_chiral[base_nodes], node_chiral[triplet2]), dim=1))  # Shape: [num_triplets, node_size]
            chiral_gate_t3 = self.chiral_gate(torch.cat((node_chiral[base_nodes], node_chiral[triplet3]), dim=1))  # Shape: [num_triplets, node_size]
            # Aggregate chiral gates
            chiral_gate = chiral_gate_t1 + chiral_gate_t2 + chiral_gate_t3  # Shape: [num_triplets, node_size]
            
            # Calculate scalar triple products for each triplet
            rel_pos1 = pos[base_nodes] - pos[triplet1]  # Shape: [num_triplets, 3]
            rel_pos2 = pos[base_nodes] - pos[triplet2]  # Shape: [num_triplets, 3]
            rel_pos3 = pos[base_nodes] - pos[triplet3]  # Shape: [num_triplets, 3]
            scalar_triple_products = torch.einsum(
                'bi,bi->b', rel_pos1, torch.cross(rel_pos2, rel_pos3, dim=1)
            )  # Shape: [num_triplets]
            # Compute chiral messages
            chiral_messages = chiral_gate / (scalar_triple_products + 1e-2).unsqueeze(-1)  # Shape: [num_triplets, node_size]

            # Aggregate chiral messages for each base node
            message_chiral = torch_scatter.scatter(
                chiral_messages,
                base_nodes,  # Automatically sets zero if base_node not in triplet_index
                dim=0,
                dim_size=node_scalar.size(0),
                reduce="add"
            )  # Shape: [num_nodes, node_size]
        else:
            # If there are no triplets, initialize chiral messages to zero
            message_chiral = torch.zeros_like(node_chiral).to(node_chiral.device)
        """"""
        
        return (node_scalar + message_ss + message_vs), (node_chiral + message_chiral), (node_vector + message_vv + message_ev)


class ChiralUpdate(torch.nn.Module):
    """Use triple product of v vectors to update x"""
    def __init__(self, node_size: int):
        super(ChiralUpdate, self).__init__()

    def forward(self, node_scalar, node_chiral):
        return node_scalar + node_chiral


def check_increasing(indices):
    # Find minimum point
    pivot = torch.argmin(indices)
    # Check along after pivot point
    for i in range(len(indices)-1):
        if indices[(pivot + i) % 3] > indices[(pivot + i + 1) % 3]:
            return False
    return True


class MLPNode(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_mlp,
        hidden_dim_node,
        node_type,
        activation_function,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_type = node_type
        self.num_mlp = num_mlp
        self.activation_function = activation_function

        self.mlp = nn.ModuleList()
        for _ in range(self.num_mlp):
            denselayers = []
            denselayers.append(nn.Linear(self.input_dim, hidden_dim_node[0]))
            denselayers.append(self.activation_function)
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    nn.Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(self.activation_function)
            denselayers.append(nn.Linear(hidden_dim_node[-1], output_dim))
            denselayers.append(nn.Softmax(dim=1))  # New Line for classification task of chirality
            self.mlp.append(nn.Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            outs = torch.zeros(
                (x.shape[0], self.output_dim),
                dtype=x.dtype,
                device=x.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"