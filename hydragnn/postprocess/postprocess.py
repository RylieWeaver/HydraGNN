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


def output_denormalize(y_minmax, true_values, predicted_values):
    for ihead in range(len(y_minmax)):
        for isample in range(len(predicted_values[0])):
            for iatom in range(len(predicted_values[ihead][0])):
                ymin = y_minmax[ihead][0]
                ymax = y_minmax[ihead][1]

                predicted_values[ihead][isample][iatom] = (
                    predicted_values[ihead][isample][iatom] * (ymax - ymin) + ymin
                )
                true_values[ihead][isample][iatom] = (
                    true_values[ihead][isample][iatom] * (ymax - ymin) + ymin
                )

    return true_values, predicted_values


def unscale_features_by_num_nodes(datasets_list, scaled_index_list, nodes_num_list):
    """datasets_list: list of datasets to scale back, e.g., [true_values, predicted_values];
    true_values and predicted_values[num_heads][num_samples][num_atoms]
    """
    for dataset in datasets_list:
        for scaled_index in scaled_index_list:
            head_value = dataset[scaled_index]
            for isample in range(len(nodes_num_list)):
                for iatom in range(len(head_value[isample])):
                    dataset[scaled_index][isample][iatom] *= nodes_num_list[isample]
    return datasets_list
