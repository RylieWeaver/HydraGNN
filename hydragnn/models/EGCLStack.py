##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from .Base import Base

from ..utils import unsorted_segment_mean



class EGCLStack(Base):
    def __init__(
        self,
        edge_attr_dim: int,
        *args,
        max_neighbours: Optional[int] = None,
        **kwargs,
    ):

        self.edge_dim = (
            0 if edge_attr_dim is None else edge_attr_dim
        )  # Must be named edge_dim to trigger use by Base
        super().__init__(*args, **kwargs)
        pass

    def _init_conv(self):
        last_layer = 1 == self.num_conv_layers
        self.graph_convs.append(
            self.get_conv(self.input_dim, self.hidden_dim, last_layer)
        )
        self.feature_layers.append(nn.Identity())
        for i in range(self.num_conv_layers - 1):
            last_layer = i == self.num_conv_layers - 2
            conv = self.get_conv(self.hidden_dim, self.hidden_dim, last_layer)
            self.graph_convs.append(conv)
            self.feature_layers.append(nn.Identity())

    def get_conv(self, input_dim, output_dim, last_layer=False):
        input_dim=118
        egcl = E_GCL(
            input_channels=input_dim,
            output_channels=output_dim,
            hidden_channels=self.hidden_dim,
            edge_attr_dim=self.edge_dim,
            equivariant=self.equivariance and not last_layer,
        )

        if self.equivariance and not last_layer:
            return Sequential(
                "x, pos, edge_index, edge_attr",
                [
                    (egcl, "x, pos, edge_index, edge_attr -> x, pos"),
                ],
            )
        else:
            return Sequential(
                "x, pos, edge_index, edge_attr",
                [
                    (egcl, "x, pos, edge_index, edge_attr -> x"),
                    (lambda x, pos: [x, pos], "x, pos -> x, pos"),
                ],
            )

    def _conv_args(self, data):
        if self.edge_dim > 0:
            conv_args = {
                "edge_index": data.edge_index,
                "edge_attr": data.edge_attr,
            }
        else:
            conv_args = {
                "edge_index": data.edge_index,
                "edge_attr": None,
            }

        return conv_args
    
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
        class_weights = torch.tensor([1.0, 8.0, 8.0])  # Adjust the weights based on the class distribution
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

    def __str__(self):
        return "EGCLStack"


"""
EGNN
=====

E(n) equivariant graph neural network as
a message passing neural network. The
model uses positional data only to ensure
that the message passing component is
equivariant.

In particular this message passing layer
relies on the angle formed by the triplet
of incomming and outgoing messages.

The three key components of this network are
outlined below. In particular, the convolutional
network that is used for the message passing
the triplet function that generates to/from
information for angular values, and finally
the radial basis embedding that is used to
include radial basis information.

"""


class E_GCL(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels,
        edge_attr_dim=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        recurrent=False,
        coords_weight=1.0,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=True,
        equivariant=False,
    ) -> None:
        super(E_GCL, self).__init__()
        input_edge = input_channels * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        self.equivariant = equivariant
        self.edge_attr_dim = edge_attr_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edge_attr_dim, hidden_channels),
            act_fn,
            nn.Linear(hidden_channels, hidden_channels),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                hidden_channels + input_channels + nodes_attr_dim, hidden_channels
            ),
            act_fn,
            nn.Linear(hidden_channels, output_channels),
        )

        self.clamp = clamp

        if self.equivariant:

            layer = nn.Linear(hidden_channels, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_channels, hidden_channels))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
                self.coords_range = nn.Parameter(torch.ones(1)) * 3
            self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

        self.act_fn = act_fn
        pass

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        # trans = coord_diff * self.coord_mlp(edge_feat)
        # trans = torch.clamp(
        #     trans, min=-100, max=100
        # )  # This is never activated but just in case it case it explosed it may save the train
        # agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        # coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / (norm)

        return radial, coord_diff

    def forward(self, x, coord, edge_index, edge_attr, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # Message Passing
        edge_feat = self.edge_model(x[row], x[col], radial, edge_attr)
        if self.equivariant:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        x, agg = self.node_model(x, edge_index, edge_feat, node_attr)
        if self.equivariant:
            return x, coord
        else:
            return x


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result



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