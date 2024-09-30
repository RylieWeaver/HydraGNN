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
        # assert (
        #     hidden_dim > 1
        # ), "PainnNet requires more than one hidden dimension between input_dim and output_dim."
        self_inter = ChiralMessage(node_size=118)
        # cross_inter = ChiralUpdate(node_size=input_dim)
        """
        The following linear layers are to get the correct sizing of embeddings. This is 
        necessary to use the hidden_dim, output_dim of HYDRAGNN's stacked conv layers correctly 
        because node_scalar and node-vector are updated through a sum.
        """
        # node_embed_out = nn.Sequential(
        #     nn.Linear(input_dim, output_dim),
        #     nn.Tanh(),
        #     nn.Linear(output_dim, output_dim),
        # )  # Tanh activation is necessary to prevent exploding gradients when learning from random signals in test_graphs.py
        # vec_embed_out = nn.Linear(input_dim, output_dim) if not last_layer and input_dim != output_dim else None

        if not last_layer:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist",
                [
                    (self_inter, "x, xc, edge_index, pos -> xc"),
                    # (cross_inter, "x, v -> x"),
                    # (node_embed_out, "x -> x"),
                    # (vec_embed_out, "v -> v") if vec_embed_out else (lambda v: v, "v -> v"),
                    (lambda x, xc, pos: [x, xc, pos], "x, xc, pos -> x, xc, pos"),
                ],
            )
        else:
            return geom_nn.Sequential(
                "x, xc, v, pos, edge_index, diff, dist",
                [
                    (self_inter, "x, xc, edge_index, pos -> xc"),
                    # (
                    #     cross_inter,
                    #     "x, v -> x",
                    # ),  # v is not updated in the last layer to avoid hanging gradients
                    # (
                    #     node_embed_out,
                    #     "x -> x",
                    # ),  # No need to embed down v because it's not used anymore
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
        for conv, feat_layer, chiral_update in zip(self.graph_convs, self.feature_layers, self.chiral_updates):
            if not self.conv_checkpointing:
                c, xc, pos = conv(x=x, xc=xc, v=v, pos=pos, **conv_args)  # Added v here
            else:
                c, xc, pos = checkpoint(  # Added v here
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
    
    def loss(useless1, pred, value, useless2):
        # Reshape 'value' to [batch_size, 3] where each row is a one-hot encoding of the class
        value = value.view(-1, 3).long()

        # Convert the one-hot encoding into class indices using argmax
        value = torch.argmax(value, dim=1)

        # Apply cross-entropy loss
        # 'pred[0]' should have shape [batch_size, num_classes] and 'value' should have shape [batch_size]
        class_weights = torch.tensor([1.0, 100.0, 100.0])  # Adjust the weights based on the class distribution
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
        ), "PAINNNet requires node positions (data.pos) to be set."

        # Calculate relative vectors and distances
        i, j = data.edge_index[0], data.edge_index[1]
        diff = data.pos[i] - data.pos[j]
        dist = diff.pow(2).sum(dim=-1).sqrt()
        norm_diff = diff / dist.unsqueeze(-1)

        # Instantiate tensor to hold equivariant traits
        v = torch.zeros(data.x.size(0), 3, data.x.size(1), device=data.x.device)
        # v = torch.zeros(data.x.size(0), 3, 3, device=data.x.device)
        data.v = v
        xc = torch.zeros_like(data.x)
        data.xc = xc

        conv_args = {
            "edge_index": data.edge_index.t().to(torch.long),
            "diff": norm_diff,
            "dist": dist,
        }

        return data, conv_args
    

class ChiralMessage(torch.nn.Module):
    def __init__(self, node_size: int):
        super(ChiralMessage, self).__init__()
        self.linear1 = torch.nn.Linear(node_size, node_size)
        self.linear2 = torch.nn.Linear(node_size, node_size)
        self.linear3 = torch.nn.Linear(node_size, node_size)
        self.linear4 = torch.nn.Linear(node_size, node_size)
        self.linear5 = torch.nn.Linear(node_size, node_size)
        self.linear6 = torch.nn.Linear(node_size, node_size)
    
    def forward(self, node_scalar, node_chiral, edge_index, pos):
        chiral_message = []
        
        # Iterate over each base node in the graph
        unique_base_nodes = torch.unique(edge_index[:, 0])  # Get unique base nodes from edge_index[0]
        
        for base_node_idx in unique_base_nodes:
            # Base node info
            base_scalar = node_scalar[base_node_idx]
            base_pos = pos[base_node_idx]
            neighbors = edge_index[:, 1][edge_index[:, 0] == base_node_idx]
            neighbors_scalar = node_scalar[neighbors]
            neighbors_pos = pos[neighbors]
            node_message = process_triplets(base_scalar, neighbors_scalar, base_pos, neighbors_pos, self.linear1, self.linear2, self.linear3, self.linear4, self.linear5, self.linear6)
            chiral_message.append(node_message)
        
        return node_chiral[unique_base_nodes] + torch.stack(chiral_message)


class ChiralUpdate(torch.nn.Module):
    """Use triple product of v vectors to update x"""
    def __init__(self, node_size: int):
        super(ChiralUpdate, self).__init__()

    def forward(self, node_scalar, node_chiral):
        return node_scalar + node_chiral
        

def down_and_triple_product(node_vector):
    node_vector = nn.Linear(node_vector.shape[2], 3)(node_vector)
    
    n_nodes = node_vector.shape[0]
    results = torch.zeros((n_nodes, 1))
    
    for i in range(n_nodes):
        vectors = node_vector[i]
        nonzero_mask = torch.sum(vectors.abs(), dim=0) > 0
        nonzero_vectors = vectors[:, nonzero_mask]
        
        if nonzero_vectors.shape[1] == 3:
            triple_product = compute_scalar_triple_product(nonzero_vectors.T)
            results[i] = torch.sign(triple_product)
        elif nonzero_vectors.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Node {i} has {nonzero_vectors.shape[1]} nonzero vectors, expected 1 or 3.")
        
    return results

def filter_and_triple_product(node_vector):
    # Initialize an empty tensor for the results
    n_nodes = node_vector.shape[0]
    results = torch.zeros((n_nodes, 1))

    # Loop through each node and process its vectors
    for i in range(n_nodes):
        # Get the vectors for the current node
        vectors = node_vector[i]  # Shape: [3, 118]
        
        # Find nonzero vectors
        nonzero_mask = torch.sum(vectors.abs(), dim=0) > 0
        nonzero_vectors = vectors[:, nonzero_mask]  # Shape: [3, n_nonzero]

        # Handle cases based on the number of nonzero vectors
        if nonzero_vectors.shape[1] == 3:
            # Compute scalar triple product
            triple_product = compute_scalar_triple_product(nonzero_vectors.T)
            results[i] = torch.sign(triple_product)
        elif nonzero_vectors.shape[1] == 1:
            # Return zero (already initialized to zero in `results`)
            pass
        else:
            # Throw an error for invalid number of nonzero vectors
            # raise ValueError(f"Node {i} has {nonzero_vectors.shape[1]} nonzero vectors, expected 1 or 3.")
            pass
        
    return results

def compute_scalar_triple_product(vectors):
    # Compute scalar triple product of three vectors
    return torch.dot(vectors[0], torch.cross(vectors[1], vectors[2]))

def process_triplets(base_scalar, neighbors_scalar, base_pos, neighbors_pos, linear1, linear2, linear3, linear4, linear5, linear6):
    # Generate all combinations of 3 neighbors
    triplets = list(itertools.combinations(range(len(neighbors_scalar)), 3))
    
    # Initialize the chiral update for the base node
    base_updates = []
    
    # Initialize updates for neighbor nodes
    # neighbor_updates = torch.zeros_like(neighbors_scalar)  # Shape: (num_neighbors, hidden_dim)
    
    for triplet in triplets:
        # Extract positions and scalars of the 3 neighbors
        neighbor_pos = [neighbors_pos[i] for i in triplet]  # List of 3 tensors
        rel_pos = [pos - base_pos for pos in neighbor_pos]  # List of 3 tensors
        
        # Get ordering from relative positions
        indices = torch.argmax(neighbors_scalar, dim=1)  # Shape: (3,)
        is_increasing = torch.all(indices[1:] > indices[:-1])
        ordering = get_ordering(base_pos, neighbor_pos)  # List of indices [i0, i1, i2]
        
        # Order the neighbor positions and scalars accordingly
        rel_pos_ordered = [rel_pos[i] for i in ordering]  # List of 3 tensors
        neighbor_indices_ordered = [triplet[i] for i in ordering]  # Indices in neighbors_scalar
        neighbors_scalar_ordered = neighbors_scalar[neighbor_indices_ordered]  # Shape: (3, hidden_dim)
        
        # Look at indices for increasing or decreasing
        # base_index = torch.argmax(base_scalar)
        # indices = torch.argmax(neighbors_scalar_ordered, dim=1)  # Shape: (3,)
        # is_increasing = check_increasing(indices)
        # if is_increasing:
        #     input_value = torch.tensor([1.0], device=neighbors_scalar.device)
        # else:
        #     input_value = torch.tensor([-1.0], device=neighbors_scalar.device)
        
        # Apply linear layers to neighbor scalars
        # neighbors_scalar_emb = linear2(torch.nn.functional.silu(linear1(neighbors_scalar_ordered)))  # Shape: (3, hidden_dim)
        # neighbors_scalar_emb = neighbors_scalar_ordered
        neighbors_scalar_emb = linear1(neighbors_scalar_ordered)
        
        # Create clockwise and counter-clockwise pairs
        prev_indices = (torch.arange(3) - 1) % 3  # [2, 0, 1]
        next_indices = (torch.arange(3) + 1) % 3  # [1, 2, 0]
        
        # Construct messages by concatenating embeddings of adjacent neighbor pairs
        # clockwise_pairs = torch.cat([neighbors_scalar_emb, neighbors_scalar_emb[prev_indices]], dim=1)  # Shape: (3, 2 * hidden_dim)
        # counter_clockwise_pairs = torch.cat([neighbors_scalar_emb, neighbors_scalar_emb[next_indices]], dim=1)  # Shape: (3, 2 * hidden_dim)
        
        # Compute messages
        # clockwise_message = linear4(torch.nn.functional.silu(linear3(clockwise_pairs)))  # Shape: (3, hidden_dim)
        # counter_clockwise_message = linear4(torch.nn.functional.silu(linear3(counter_clockwise_pairs)))  # Shape: (3, hidden_dim)
        clockwise_message = torch.cat([neighbors_scalar_emb[prev_indices]], dim=1)  # Shape: (3, 2 * hidden_dim)
        counter_clockwise_message = torch.cat([neighbors_scalar_emb[next_indices]], dim=1)  # Shape: (3, 2 * hidden_dim)
        message = clockwise_message - counter_clockwise_message  # Shape: (3, hidden_dim)
        
        # Compute chiral update
        # chiral_update = clockwise_message - counter_clockwise_message  # Shape: (3, hidden_dim)
        chiral_update = linear4(torch.nn.functional.silu(linear3(message)))  # Shape: (3, hidden_dim)
        
        # Update neighbor scalars
        neighbors_chiral = linear6(torch.nn.functional.silu(linear5(neighbors_scalar_emb + chiral_update)))  # Shape: (3, hidden_dim)
        # neighbors_chiral = linear6(torch.nn.functional.silu(linear5(neighbors_chiral)))  # Shape: (3, hidden_dim)
        
        # Aggregate updates for the base node (optional)
        base_update = torch.sum(neighbors_chiral, dim=0)  # Shape: (hidden_dim)
        base_updates.append(base_update)
        
        # Aggregate updates for neighbor nodes
        # for idx, neighbor_idx in enumerate(neighbor_indices_ordered):
        #     neighbor_updates[neighbor_idx] += neighbors_chiral[idx]
        # base_update = torch.ones_like(base_scalar) * input_value
        # base_updates.append(base_update)
        
    # Sum all base updates
    if base_updates:
        total_base_update = torch.sum(torch.stack(base_updates), dim=0)  # Shape: (hidden_dim)
    else:
        total_base_update = torch.zeros_like(base_scalar)
        
    # Return updates
    return total_base_update


def get_ordering(base_pos, neighbor_pos):
    """
    Determines an ordering of neighbor nodes that is invariant under rotation and translation,
    and equivariant to chirality (left/right orientation).

    Parameters:
    base_pos (torch.Tensor): A 3D tensor representing the base node position.
    neighbor_pos (list of torch.Tensor): A list of three 3D tensors representing the neighbor positions.

    Returns:
    list: A list of indices representing the consistent ordering of the neighbor nodes.
    """
    # Stack neighbor positions into a tensor
    neighbor_pos_tensor = torch.stack(neighbor_pos)  # Shape: (3, 3)

    # Compute the centroid of the neighbor positions
    centroid = torch.mean(neighbor_pos_tensor, dim=0)  # Shape: (3,)

    # Center the neighbor positions
    centered_neighbors = neighbor_pos_tensor - centroid  # Shape: (3, 3)

    # Compute vectors from the centroid to each neighbor
    v0 = centered_neighbors[0]
    v1 = centered_neighbors[1]
    v2 = centered_neighbors[2]

    # Compute the normal vector of the plane formed by the neighbor positions
    normal = torch.cross(v1 - v0, v2 - v0)  # Shape: (3,)

    # Compute vector from centroid to base position
    base_vector = base_pos - centroid  # Shape: (3,)

    # Adjust the normal vector to have a consistent direction
    if torch.dot(normal, base_vector) < 0:
        normal = -normal

    # Normalize the normal vector
    normal = normal / torch.norm(normal)

    # Define the X-axis as the normalized vector from v0 to v1
    x_axis = (v1 - v0)
    x_axis = x_axis / torch.norm(x_axis)

    # Define the Y-axis as the cross product of the normal vector and X-axis
    y_axis = torch.cross(normal, x_axis)

    # Project centered neighbor positions onto the plane defined by X and Y axes
    projected = []
    for v in centered_neighbors:
        x = torch.dot(v, x_axis)
        y = torch.dot(v, y_axis)
        projected.append((x, y))

    # Compute angles with respect to the X-axis
    angles = [torch.atan2(y, x).item() for x, y in projected]

    # Normalize angles to [0, 2π)
    angles = [(angle + 2 * torch.pi) % (2 * torch.pi) for angle in angles]

    # Determine ordering based on angles
    ordering = sorted(range(3), key=lambda i: angles[i])

    return ordering

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