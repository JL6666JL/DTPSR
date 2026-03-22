"""
Modified from SPADE (https://arxiv.org/abs/1903.07291)

Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn

# Function to easily switch between GroupNorm and BatchNorm
def create_normalization(norm_type, channels, num_groups=32):
    if norm_type == 'group':
        return nn.GroupNorm(num_groups, channels)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


class SAFT(nn.Module):
    '''
    |norm_nc|: the #channels of the normalized activations, hence the output dim of SAFT
    |label_nc|: the #channels of the input semantic map, hence the input dim of SAFT
    '''
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        # kernel size is set to 1 to encode segmentation map
        ks = 1
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU(),
            nn.Conv2d(nhidden, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x_normalized, cond):
        # Produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x_normalized * (1 + gamma) + beta

        return out

class GFM(nn.Module):
    def __init__(self, input_nc, seg_nc, text_nc, norm_type='group'):
        super().__init__()
        # Choose the appropriate normalization (GroupNorm or BatchNorm)
        self.param_free_norm = create_normalization(norm_type, input_nc)

        # Image SFT (SAFT block with Segmentation Mask)
        # self.image_sft = SAFT(input_nc, seg_nc)

        # Text SFT (SAFT block with Segmentation-CLIP map)
        self.text_hf_sft = SAFT(input_nc, text_nc)
        self.text_lf_sft = SAFT(input_nc, text_nc)


    def forward(self, input_feat, scm_hf, scm_lf, lf_ratio):

        # Normalize the input feature
        input_feat_normalized = self.param_free_norm(input_feat)
        
        # # Apply the Image SAFT
        # image_sft_out = self.image_sft(input_feat_normalized, seg_mask)

        # # Apply the Text SAFT
        # text_sft_out = self.text_sft(input_feat_normalized, scm)

        text_hf_sft_out = self.text_hf_sft(input_feat_normalized, scm_hf)
        text_lf_sft_out = self.text_lf_sft(input_feat_normalized, scm_lf)
        lf_ratio = lf_ratio.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Sum the outputs from both Image and Text SAFT blocks
        fused_output = (1-lf_ratio)*text_hf_sft_out + lf_ratio*text_lf_sft_out
        
        return fused_output

