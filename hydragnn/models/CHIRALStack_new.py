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

import sys


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
        self.graph_convs.append(self.get_conv(self.hidden_dim, self.hidden_dim))
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())
        # Etc
        self.encoder = nn.Linear(118, self.hidden_dim)
        self.chiral_updates = torch.nn.ModuleList()
        self.chiral_updates.append(self.get_update(self.hidden_dim))
        
    def get_update(self, node_size):
        chiral_update = ChiralUpdate(node_size=node_size)
        return geom_nn.Sequential("x, xc", [(chiral_update, "x, xc -> x")])

    def get_conv(self, input_dim, output_dim, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        self_inter = ChiralMessage(node_size=input_dim)

        if not last_layer:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist, triplet_index",
                [
                    (self_inter, "x, xc, v, edge_index, diff, dist, triplet_index, pos -> x, xc, v"),
                    (lambda x, xc, v, pos: [x, xc, v, pos], "x, xc, v, pos -> x, xc, v, pos"),
                ],
            )
        else:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist, triplet_index",
                [
                    (self_inter, "x, xc, v, edge_index, diff, dist, triplet_index, pos -> x, xc, v"),
                    (lambda x, xc, v, pos: [x, xc, v, pos], "x, xc, v, pos -> x, xc, v, pos"),
                ],
            )

    def forward(self, data):
        data, conv_args = self._conv_args(data)
        x = data.x
        xc = data.xc
        v = data.v
        pos = data.pos

        ### encoder part ####
        x = self.encoder(x)
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                c, xc, v, pos = conv(x=x, xc=xc, v=v, pos=pos, **conv_args)
            else:
                c, xc, v, pos = checkpoint(
                    conv, use_reentrant=False, x=x, xc=xc, v=v, pos=pos, **conv_args
                )
            x = self.activation_function(feat_layer(c))
        # Add chirality
        for chiral_update in self.chiral_updates:
            x = chiral_update(x, xc)
            # print("Chirality Updated")
            # print(x[0])
            # sys.exit(1)

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
    
    def loss(useless1, pred, value, useless2):
        # Reshape 'value' to [num_nodes, 3] where each row is a one-hot encoding of the class
        value = value.view(-1, 3)

        # Convert the one-hot encoding into class indices using argmax
        value = torch.argmax(value, dim=1).long()

        # Apply cross-entropy loss
        # 'pred[0]' should have shape [num_nodes, num_classes] and 'value' should have shape [num_nodes]
        # class_weights = torch.tensor([0.1, 50.0, 50.0]).to('cuda:0')  # Adjust the weights based on the class distribution
        class_weights = torch.tensor([0.1, 50.0, 50.0])
        loss = F.cross_entropy(pred[0], value, weight=class_weights)
       
        # Calculate overall predictions
        predictions = torch.argmax(pred[0], dim=1)
        # Calculate accuracy for each class
        class_accuracies = []
        if torch.rand(1) < 0.09:  # Print only 1% of the time
            for i in range(3):  # Assuming 3 classes
                mask = (value == i)  # Identify samples of class 'i'
                if mask.sum() > 0:  # Avoid division by zero
                    accuracy = (predictions[mask] == value[mask]).float().mean()
                else:
                    accuracy = torch.tensor(0.0, device=pred[0].device)  # If no samples for this class, accuracy is 0
                class_accuracies.append(accuracy)
                print(f"Accuracy for class {i}: {accuracy.item()}")

        return loss, [loss]
    
    def loss_binding(useless1, pred, value, useless2):
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
        ), "CHIRAL requires node positions (data.pos) to be set."

        # Calculate relative vectors and distances
        i, j = data.edge_index[0], data.edge_index[1]
        diff = data.pos[i] - data.pos[j]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        norm_diff = diff / dist.unsqueeze(-1)

        # Instantiate tensor to hold equivariant traits
        # v = torch.zeros(data.x.size(0), 3, data.x.size(1), device=data.x.device)
        v = torch.zeros(data.x.size(0), 3, self.hidden_dim, device=data.x.device)
        data.v = v
        # xc = torch.zeros_like(data.x)
        xc = torch.zeros(data.x.size(0), self.hidden_dim, device=data.x.device)
        data.xc = xc

            
        # Ordered Triplet Indices
        triplet_index = []
        base_nodes = data.edge_index[0].unique()
        for base_node_id in base_nodes:
            # Get neighbors
            neighbors = data.edge_index[1][data.edge_index[0] == base_node_id]
            
            # Skip nodes with fewer than 3 neighbors
            # if neighbors.size(0) < 3:
            if neighbors.size(0) != 4:
                continue
            if torch.argmax(data.x[base_node_id][:118]) != 6:
                continue
            
            # # NOTE METHOD 1
            # # Get ordering from priority of neighbors
            # edge_priority = data.edge_priority[data.edge_index[0] == base_node_id]
            # highest_to_lowest_indices = torch.argsort(edge_priority, descending=True)
            # ordered_neighbors = neighbors[highest_to_lowest_indices]
            # # Take out the highest index three neighbors
            # top_three_neighbors = ordered_neighbors[:3]
            # ordered_triplet = top_three_neighbors.tolist()
            # triplet_full = [base_node_id] + ordered_triplet
            # triplet_index.append(triplet_full)
            # # Check stp
            # rel1 = data.pos[base_node_id] - data.pos[ordered_triplet[0]]
            # rel2 = data.pos[base_node_id] - data.pos[ordered_triplet[1]]
            # rel3 = data.pos[base_node_id] - data.pos[ordered_triplet[2]]
            # stp = torch.dot(rel1, torch.linalg.cross(rel2, rel3))
            # chiral_tag = data.y[3*base_node_id:3*(base_node_id+1)].squeeze().tolist()
            # if chiral_tag == [0,1,0]:
            #     if stp < 0:
            #         print("Consistent STP")
            #     else:
            #         print("Inconsistent STP")
            # elif chiral_tag == [0,0,1]:
            #     if stp > 0:
            #         print("Consistent STP")
            #     else:
            #         print("Inconsistent STP")
                    
            
            # NOTE METHOD 2
            # Get ordering from priority of neighbors
            edge_priority = data.edge_priority[data.edge_index[0] == base_node_id]
            highest_to_lowest_indices = torch.argsort(edge_priority, descending=True)
            ordered_neighbors = neighbors[highest_to_lowest_indices]
            # Separate the neighbors
            top_three_neighbors = ordered_neighbors[:3]
            low_neighbor = ordered_neighbors[3]
            ordered_triplet = top_three_neighbors.tolist()
            triplet_full = [base_node_id] + ordered_triplet
            triplet_index.append(triplet_full)
            # Check stp
            rel1 = data.pos[base_node_id] - data.pos[low_neighbor]
            rel2 = data.pos[ordered_triplet[0]] - data.pos[ordered_triplet[1]]
            rel3 = data.pos[ordered_triplet[1]] - data.pos[ordered_triplet[2]]
            # Make rel2 and rel3 orthogonal to rel1 by taking out that part
            # rel2 = rel2 - torch.dot(rel2, rel1) * rel1 / torch.dot(rel1, rel1)
            # rel3 = rel3 - torch.dot(rel3, rel1) * rel1 / torch.dot(rel1, rel1)
            stp = torch.dot(rel1, torch.linalg.cross(rel2, rel3))
            chiral_tag = data.y[3*base_node_id:3*(base_node_id+1)].squeeze().tolist()
            if chiral_tag == [0,1,0]:
                if stp < 0:
                    print("Consistent STP")
                else:
                    print("Inconsistent STP")
            elif chiral_tag == [0,0,1]:
                if stp > 0:
                    print("Consistent STP")
                else:
                    print("Inconsistent STP")
            
            
            # # Generate all possible triplets of neighbors
            # for triplet in itertools.combinations(neighbors.tolist(), 3):
            #     triplet = list(triplet)  # Convert from tuple to list for manipulation

            #     # Extract atomic numbers from one-hot encoding
            #     atomic_numbers_one_hot = data.x[triplet][:, :118]  # Assuming first 118 features are one-hot
            #     atomic_numbers = torch.argmax(atomic_numbers_one_hot, dim=1)  # Shape: [3]

            #     # Sort triplet based on atomic numbers (highest to lowest)
            #     highest_to_lowest_indices = torch.argsort(atomic_numbers, descending=True)
            #     ordered_triplet = [triplet[idx] for idx in highest_to_lowest_indices]

            #     # Create a full triplet including the base node
            #     triplet_full = [base_node_id] + ordered_triplet  # Shape: [4]

            #     # Append to the triplet list
            #     triplet_index.append(triplet_full)

        if triplet_index:
            # Convert list of triplets to a tensor
            triplet_index = torch.tensor(triplet_index, dtype=torch.long, device=data.edge_index.device)  # Shape: [num_triplets, 4]
        else:
            # Handle case with no triplets
            triplet_index = torch.empty((0, 4), dtype=torch.long, device=data.edge_index.device)
        
        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),  # Shape: [2, num_edges]
            "diff": diff,  # Shape: [num_edges, 3]
            "dist": dist,        # Shape: [num_edges]
            "triplet_index": triplet_index,  # Shape: [num_triplets, 4]
        }

        return data, conv_args  # Ensure this is inside the function and outside the loop


class ChiralMessage(torch.nn.Module):
    def __init__(self, node_size: int):
        super(ChiralMessage, self).__init__()
        self.node_size = node_size
        cutoff = 5.0
        num_radial = 5
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        activation1 = nn.SiLU()
        activation2 = nn.SiLU()
        
        self.scalar_gate = nn.Sequential(torch.nn.Linear(2*node_size, node_size), activation1, torch.nn.Linear(node_size, 4*node_size))
        self.chiral_edge_gate = nn.Sequential(torch.nn.Linear(node_size, node_size), activation1, torch.nn.Linear(node_size, node_size))
        self.chiral_vector_gate = nn.Sequential(torch.nn.Linear(2*node_size, node_size), activation1, torch.nn.Linear(node_size, node_size))
        self.update_V = nn.Linear(node_size, node_size)
        self.scalar_vector_gate = nn.Sequential(torch.nn.Linear(node_size, node_size), activation1, torch.nn.Linear(node_size, node_size))
        self.embed_chiral = nn.Sequential(nn.Linear(2*node_size, node_size), activation1, nn.Linear(node_size, node_size))
        self.embed_chiral_edge = nn.Sequential(nn.Linear(node_size, node_size), activation1)
        # , nn.Linear(node_size, node_size)
        self.embed_chiral_vector = nn.Sequential(nn.Linear(node_size, node_size), activation1, nn.Linear(node_size, node_size))
        self.embed_chiral_message = nn.Sequential(nn.Linear(2*node_size, node_size), activation1, nn.Linear(node_size, node_size))
        
        self.scalar_filter_layer = nn.Linear(num_radial, 4*node_size)
        self.chiral_edge_filter_layer = nn.Linear(num_radial, 1)
        self.chiral_vector_filter_layer = nn.Linear(num_radial, 1)
        self.chiral_vector_filter = nn.Sequential(nn.Linear(node_size, node_size), activation2, nn.Linear(node_size, 64))
        
        self.chiral_env_embedding = nn.Sequential(nn.Linear(3*node_size, node_size), activation1, nn.Linear(node_size, node_size))
        self.chiral_edge_env_filter = nn.Sequential(nn.Linear(2*node_size, node_size), activation1, nn.Linear(node_size, 1))
        self.chiral_vector_env_filter = nn.Sequential(nn.Linear(2*node_size, node_size), activation1, nn.Linear(node_size, 1))
        
        self.scalar_update = nn.Sequential(nn.Linear(node_size, node_size), activation2, nn.Linear(node_size, node_size))
        self.chiral_update = nn.Sequential(nn.Linear(node_size, node_size), activation2, nn.Linear(node_size, node_size))
        
    def forward(self, node_scalar, node_chiral, node_vector, edge_index, edge_diff, edge_dist, triplet_index, pos):
        """Scalar / Vector Section"""
        # Compute scalar gates
        message_gate = self.scalar_gate(torch.cat((node_scalar[edge_index[:, 0]], node_scalar[edge_index[:, 1]]), dim=1))  # Shape: [num_edges, 2 * node_size]  -->  [num_edges, 4 * node_size]
        scalar_filter_weight = self.scalar_filter_layer(sinc_expansion(edge_dist, self.num_radial, self.cutoff)) * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        message_gate = message_gate * scalar_filter_weight
        gate_vv, gate_ev, messages_ss, env_chiral = torch.split(
            message_gate,
            self.node_size,
            dim=1,
        ) # Each Shape: [num_edges, node_size]
        
        # Apply softmax to vector gates
        gate_vv = torch.softmax(gate_vv, dim=1)
        gate_ev = torch.softmax(gate_ev, dim=1)
        
        # Make neighbor messages
        messages_vv = gate_vv.unsqueeze(1) * node_vector[edge_index[:, 1]]  # Shape: [num_edges, 3, node_size]
        # messages_ev = gate_edge.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)  # Shape: [num_edges, 3, node_size]
        messages_ev = gate_ev.unsqueeze(1) * edge_diff.unsqueeze(-1)  # Shape: [num_edges, 3, node_size]
        
        # Aggregate
        message_ss = torch_scatter.scatter(messages_ss, edge_index[:, 0], dim=0, dim_size=node_scalar.size(0), reduce="sum")
        message_vv = torch_scatter.scatter(messages_vv, edge_index[:, 0], dim=0, dim_size=node_vector.size(0), reduce="sum")
        message_ev = torch_scatter.scatter(messages_ev, edge_index[:, 0], dim=0, dim_size=node_vector.size(0), reduce="sum")
        env_chiral1 = torch_scatter.scatter(env_chiral, edge_index[:, 0], dim=0, dim_size=node_chiral.size(0), reduce="sum")
        env_chiral2 = torch_scatter.scatter(env_chiral, edge_index[:, 0], dim=0, dim_size=node_chiral.size(0), reduce="max")
        env_chiral3 = torch_scatter.scatter(env_chiral, edge_index[:, 0], dim=0, dim_size=node_chiral.size(0), reduce="min")
        env_chiral = torch.cat((env_chiral1, env_chiral2, env_chiral3), dim=1)
        env_chiral = self.chiral_env_embedding(env_chiral)

        # Self Cross-Message
        Vv = self.update_V(node_vector)
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        gate_vs = self.scalar_vector_gate(node_scalar)
        message_vs = gate_vs * Vv_norm
        # message_vs = torch.zeros_like(node_scalar).to(node_scalar.device)
        """"""
        
        """Chiral Section"""
        if triplet_index.numel() > 0:
            # Create gates and messages from triplets
            base_nodes = triplet_index[:, 0]  # Shape: [num_triplets]
            triplet1 = triplet_index[:, 1]    # Shape: [num_triplets]
            triplet2 = triplet_index[:, 2]    # Shape: [num_triplets]
            triplet3 = triplet_index[:, 3]    # Shape: [num_triplets]
            
            # Compute chiral edge and vector scalar messages
            ## Edge
            # chiral_edge_t1 = self.chiral_edge_gate(node_scalar[triplet1])  # Shape: [num_triplets, node_size]
            # chiral_edge_t2 = self.chiral_edge_gate(node_scalar[triplet2])  # Shape: [num_triplets, node_size]
            # chiral_edge_t3 = self.chiral_edge_gate(node_scalar[triplet3])  # Shape: [num_triplets, node_size]
            # stacked = torch.stack((chiral_edge_t1, chiral_edge_t2, chiral_edge_t3), dim=0)  # Shape: [3, num_triplets, node_size]
            # chiral_edge = self.embed_chiral_edge(torch.sum(stacked, dim=0))         # Shape: [num_triplets, node_size]
            chiral_edge = self.embed_chiral_edge(node_scalar[base_nodes] + node_scalar[triplet1] + node_scalar[triplet2] + node_scalar[triplet3])
            # ## Vector
            # # chiral_vector_t1 = self.chiral_vector_gate(torch.cat((node_scalar[base_nodes], node_scalar[triplet1]), dim=1))  # Shape: [num_triplets, node_size]
            # # chiral_vector_t2 = self.chiral_vector_gate(torch.cat((node_scalar[base_nodes], node_scalar[triplet2]), dim=1))  # Shape: [num_triplets, node_size]
            # # chiral_vector_t3 = self.chiral_vector_gate(torch.cat((node_scalar[base_nodes], node_scalar[triplet3]), dim=1))  # Shape: [num_triplets, node_size]
            # # chiral_vector = torch.mean(torch.stack((chiral_vector_t1, chiral_vector_t2, chiral_vector_t3)), dim=0)  # Shape: [num_triplets, node_size]
            chiral_vector = self.embed_chiral_vector(node_scalar[base_nodes] + node_scalar[triplet1] + node_scalar[triplet2] + node_scalar[triplet3])
            # # chiral_vector_filter = torch.softmax((self.chiral_vector_filter(node_scalar[triplet1] + node_scalar[triplet2] + node_scalar[triplet3])), dim=1)
            chiral_vector_filter = self.chiral_vector_filter(node_scalar[triplet1] + node_scalar[triplet2] + node_scalar[triplet3])
            # # chiral_vector_filter = torch.ones(chiral_vector.shape[0], 64, device=chiral_vector.device)
            
            chiral_environment = env_chiral[base_nodes]
            chiral_edge_env = self.chiral_edge_env_filter(torch.cat((chiral_environment, chiral_edge), dim=1))
            chiral_vector_env = self.chiral_vector_env_filter(torch.cat((chiral_environment, chiral_vector), dim=1))
            
            # Calculate scalar triple products for each triplet in Edge and Vector
            ## Edge
            ### Scalar Triple Product
            edge_rel_pos1 = pos[base_nodes] - pos[triplet1]  # Shape: [num_triplets, 3]
            edge_rel_pos2 = pos[base_nodes] - pos[triplet2]  # Shape: [num_triplets, 3]
            edge_rel_pos3 = pos[base_nodes] - pos[triplet3]  # Shape: [num_triplets, 3]
            edge_triple_products = torch.einsum(
                'bi,bi->b', edge_rel_pos1, torch.cross(edge_rel_pos2, edge_rel_pos3, dim=1)
            )  # Shape: [num_triplets]
            ### Message
            edge_triple_product_mag = torch.abs(edge_triple_products) + 1e-3
            chiral_edge_filter_weight = self.chiral_edge_filter_layer(sinc_expansion(edge_triple_product_mag, self.num_radial, self.cutoff**3)) * cosine_cutoff(edge_triple_product_mag, self.cutoff**3).unsqueeze(-1)
            messages_edge_chiral = chiral_edge * chiral_edge_filter_weight
            messages_edge_chiral = chiral_edge * chiral_edge_env
            # messages_edge_chiral = chiral_edge
            messages_edge_chiral = messages_edge_chiral * torch.sign(edge_triple_products.unsqueeze(-1))  # Shape: [num_triplets, node_size]
            message_chiral_edge = torch_scatter.scatter(
                messages_edge_chiral,
                base_nodes,  # Automatically sets zero if base_node not in triplet_index
                dim=0,
                dim_size=node_scalar.size(0),
                reduce="sum"
            )  # Shape: [num_nodes, node_size]
      
            
            
            # Vector
            ## Initialize the aggregated chiral vector messages
            message_chiral_vector_total = torch.zeros_like(node_chiral).to(node_chiral.device)
            messages_list = []
            ## Iterate over the v indices (0, 1, 2, etc...)
            for dim1 in range(4):
                for dim2 in range(4):
                    for dim3 in range(4):
                        ### Scalar Triple Product
                        vec_rel_pos1 = pos[base_nodes] - pos[triplet1] + node_vector[triplet1, :, dim1]  # Shape: [num_triplets, 3]
                        vec_rel_pos2 = pos[base_nodes] - pos[triplet2] + node_vector[triplet2, :, dim2]  # Shape: [num_triplets, 3]
                        vec_rel_pos3 = pos[base_nodes] - pos[triplet3] + node_vector[triplet3, :, dim3]  # Shape: [num_triplets, 3]
                        
                        vector_triple_products = torch.einsum(
                            'bi,bi->b',
                            vec_rel_pos1,
                            torch.cross(vec_rel_pos2, vec_rel_pos3, dim=1)
                        )  # Shape: [num_triplets]
                        
                        ### Message
                        vector_triple_product_mag = 1 / (torch.abs(vector_triple_products) + 1e-2)  # Shape: [num_triplets]
                        chiral_vector_filter_weight = (
                            self.chiral_vector_filter_layer(
                                sinc_expansion(vector_triple_product_mag, self.num_radial, 100)
                            ) * cosine_cutoff(vector_triple_product_mag, 100).unsqueeze(-1)
                        )  # Shape: [num_triplets, node_size]
                        chiral_vector_filter_weight = (torch.abs(vector_triple_products) + 1e-2).unsqueeze(-1)  # Shape: [num_triplets, 1]
                        
                        # messages_vector_chiral = chiral_vector * (vector_triple_products).unsqueeze(-1)  # Shape: [num_triplets, node_size]
                        messages_vector_chiral = chiral_vector * chiral_vector_filter_weight  # Shape: [num_triplets, node_size]
                        messages_vector_chiral = chiral_vector * torch.sign(vector_triple_products).unsqueeze(-1)  # Shape: [num_triplets, 3]
                        messages_list.append(messages_vector_chiral)
            
            messages_vector_chiral_stacked = torch.stack(messages_list, dim=-1)  # Shape: [num_triplets, node_size, 64]
            messages_vector_chiral_filtered = messages_vector_chiral_stacked * chiral_vector_filter.unsqueeze(1)  # Shape: [num_triplets, node_size, 64]
            # messages_vector_chiral_filtered = messages_vector_chiral_stacked
            messages_vector_chiral_total = messages_vector_chiral_filtered.mean(dim=-1).squeeze(-1)  # Shape: [num_triplets, node_size]
            messages_vector_chiral_total = messages_vector_chiral_total * chiral_vector_env
            # Aggregate the messages for the current dimension
            message_vector_chiral_total = torch_scatter.scatter(
                messages_vector_chiral_total,
                base_nodes,  # Shape: [num_triplets]
                dim=0,
                dim_size=node_scalar.size(0),
                reduce="sum"
            )  # Shape: [num_nodes, node_size]
            
            # Final Message
            message_chiral = message_chiral_edge + message_vector_chiral_total
            # message_chiral = message_chiral_edge
            # message_chiral = self.embed_chiral(torch.cat((node_scalar, message_chiral), dim=1))
        else:
            # If there are no triplets, initialize chiral messages to zero
            message_chiral = torch.zeros_like(node_chiral).to(node_chiral.device)
        # message_chiral = torch.zeros_like(node_chiral).to(node_chiral.device)
        """"""
        
        return (self.scalar_update(node_scalar + message_vs + message_ss)), (node_chiral + message_chiral), (node_vector + message_vv + message_ev)
        # return node_scalar, self.chiral_update(node_chiral + message_chiral), node_vector


class ChiralUpdate(torch.nn.Module):
    """Use triple product of v vectors to update x"""
    def __init__(self, node_size: int):
        super(ChiralUpdate, self).__init__()
        self.node_size = node_size
        activation = nn.SiLU()
        
        self.update_node = nn.Sequential(nn.Linear(node_size, node_size), activation)
        # , nn.Linear(node_size, node_size),

    def forward(self, node_scalar, node_chiral):
        return self.update_node(node_scalar + node_chiral)
        # return node_chiral


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
    
    
def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    Calculate sinc radial basis function:

    sin(n * pi * d / d_cut) / d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(
        edge_dist.unsqueeze(-1) * n * torch.pi / cutoff
    ) / edge_dist.unsqueeze(-1)


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5 * (cos(pi * d / d_cut) + 1) for d < d_cut and 0 otherwise
    """
    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )