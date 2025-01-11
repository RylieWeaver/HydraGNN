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

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool, Sequential
from .Base import Base


class PNAStack(Base):
    def __init__(
        self,
        input_args,
        conv_args,
        deg: list,
        edge_dim: int,
        *args,
        **kwargs,
    ):

        # Add effect of pos to input dim
        args = list(args)
        args[0] = int(args[0]) + 3
        args = tuple(args)
        
        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        self.deg = torch.Tensor(deg)
        self.edge_dim = edge_dim

        super().__init__(input_args, conv_args, *args, **kwargs)
        
    def _embedding(self, data):
        if not hasattr(data, "edge_shifts"):
            data.edge_shifts = torch.zeros(
                (data.edge_index.size(1), 3), device=data.edge_index.device
            )
        conv_args = {"edge_index": data.edge_index.to(torch.long)}
        if self.use_edge_attr:
            assert (
                data.edge_attr is not None
            ), "Data must have edge attributes if use_edge_attributes is set."
            conv_args.update({"edge_attr": data.edge_shifts})
        return torch.cat((data.x, data.pos), dim=-1), data.pos, conv_args
        # return data.x, data.pos, conv_args

    def get_conv(self, input_dim, output_dim):
        pna = PNAConv(
            in_channels=input_dim,
            out_channels=output_dim,
            aggregators=self.aggregators,
            scalers=self.scalers,
            deg=self.deg,
            edge_dim=self.edge_dim,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        return Sequential(
            self.input_args,
            [
                (pna, self.conv_args + " -> inv_node_feat"),
                (
                    lambda inv_node_feat, equiv_node_feat: [
                        inv_node_feat,
                        equiv_node_feat,
                    ],
                    "inv_node_feat, equiv_node_feat -> inv_node_feat, equiv_node_feat",
                ),
            ],
        )

    def __str__(self):
        return "PNAStack"

