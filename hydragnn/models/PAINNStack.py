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


import torch
from torch import nn
from torch_geometric import nn as geom_nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .Base import Base


class PAINNStack(Base):
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
        self.graph_convs.append(self.get_conv(118, self.hidden_dim))
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False):
        hidden_dim = output_dim if input_dim == 1 else input_dim
        assert (
            hidden_dim > 1
        ), "PainnNet requires more than one hidden dimension between input_dim and output_dim."
        self_inter = PainnMessage(
            node_size=input_dim, edge_size=self.num_radial, cutoff=self.radius
        )
        cross_inter = PainnUpdate(node_size=input_dim, last_layer=last_layer)
        """
        The following linear layers are to get the correct sizing of embeddings. This is 
        necessary to use the hidden_dim, output_dim of HYDRAGNN's stacked conv layers correctly 
        because node_scalar and node-vector are updated through a sum.
        """
        node_embed_out = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
        )  # Tanh activation is necessary to prevent exploding gradients when learning from random signals in test_graphs.py
        vec_embed_out = nn.Linear(input_dim, output_dim) if not last_layer else None

        if not last_layer:
            return geom_nn.Sequential(
                "x, v, pos, edge_index, diff, dist",
                [
                    (self_inter, "x, v, edge_index, diff, dist -> x, v"),
                    (cross_inter, "x, v -> x, v"),
                    (node_embed_out, "x -> x"),
                    (vec_embed_out, "v -> v"),
                    (lambda x, v, pos: [x, v, pos], "x, v, pos -> x, v, pos"),
                ],
            )
        else:
            return geom_nn.Sequential(
                "x, v, pos, edge_index, diff, dist",
                [
                    (self_inter, "x, v, edge_index, diff, dist -> x, v"),
                    (
                        cross_inter,
                        "x, v -> x",
                    ),  # v is not updated in the last layer to avoid hanging gradients
                    (
                        node_embed_out,
                        "x -> x",
                    ),  # No need to embed down v because it's not used anymore
                    (lambda x, v, pos: [x, v, pos], "x, v, pos -> x, v, pos"),
                ],
            )

    def forward(self, data):
        data, conv_args = self._conv_args(
            data
        )  # Added v to data here (necessary for PAINN Stack)
        x = data.x
        v = data.v
        pos = data.pos

        ### encoder part ####
        for conv, feat_layer in zip(self.graph_convs, self.feature_layers):
            if not self.conv_checkpointing:
                c, v, pos = conv(x=x, v=v, pos=pos, **conv_args)  # Added v here
            else:
                c, v, pos = checkpoint(  # Added v here
                    conv, use_reentrant=False, x=x, v=v, pos=pos, **conv_args
                )
            x = self.activation_function(feat_layer(c))

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

    def _conv_args(self, data):
        assert (
            data.pos is not None
        ), "PAINNNet requires node positions (data.pos) to be set."

        # Calculate relative vectors and distances
        i, j = data.edge_index[0], data.edge_index[1]
        diff = data.pos[i] - data.pos[j]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        norm_diff = diff / dist.unsqueeze(-1)

        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(data.x.size(0), 3, data.x.size(1), device=data.x.device)
        data.v = v

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),
            "diff": norm_diff,
            "dist": dist,
        }

        return data, conv_args
    
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
    
    def loss(useless1, pred, value, useless2):
        # Reshape 'value' to [batch_size, 3] where each row is a one-hot encoding of the class
        value = value.view(-1, 3).long()

        # Convert the one-hot encoding into class indices using argmax
        value = torch.argmax(value, dim=1)

        # Apply cross-entropy loss
        # 'pred[0]' should have shape [batch_size, num_classes] and 'value' should have shape [batch_size]
        class_weights = torch.tensor([1.0, 20.0, 20.0])  # Adjust the weights based on the class distribution
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


class PainnMessage(nn.Module):
    """Message function"""

    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()

        self.node_size = node_size
        self.edge_size = edge_size
        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

        self.filter_layer = nn.Linear(edge_size, node_size * 3)

    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        # remember to use v_j, s_j but not v_i, s_i
        filter_weight = self.filter_layer(
            sinc_expansion(edge_dist, self.edge_size, self.cutoff)
        )
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(
            -1
        )
        scalar_out = self.scalar_message_mlp(node_scalar)
        filter_out = filter_weight * scalar_out[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=1,
        )

        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector = node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (
            edge_diff / edge_dist.unsqueeze(-1)
        ).unsqueeze(-1)
        message_vector = message_vector + edge_vector

        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)

        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector

        return new_node_scalar, new_node_vector


class PainnUpdate(nn.Module):
    """Update function"""

    def __init__(self, node_size: int, last_layer=False):
        super().__init__()

        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        self.last_layer = last_layer

        if not self.last_layer:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 3),
            )
        else:
            self.update_mlp = nn.Sequential(
                nn.Linear(node_size * 2, node_size),
                nn.SiLU(),
                nn.Linear(node_size, node_size * 2),
            )

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        if not self.last_layer:
            a_vv, a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            delta_v = a_vv.unsqueeze(1) * Uv
            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s, node_vector + delta_v
        else:
            a_sv, a_ss = torch.split(
                mlp_output,
                node_vector.shape[-1],
                dim=1,
            )

            inner_prod = torch.sum(Uv * Vv, dim=1)
            delta_s = a_sv * inner_prod + a_ss

            return node_scalar + delta_s


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