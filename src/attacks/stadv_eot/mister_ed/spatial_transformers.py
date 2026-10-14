# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/spatial_transformers.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" File that contains various parameterizations for spatial transformation
    styles. At its simplest, spatial transforms can be affine grids,
    parameterized  by 6 values. At their most complex, for a CxHxW type image
    grids can be parameterized by CxHxWx2 parameters.

    This file will define subclasses of nn.Module that will have parameters
    corresponding to the transformation parameters and will take in an image
    and output a transformed image.

    Further we'll also want a method to initialize each set to be the identity
    initially
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import pytorch_utils as utils
from torch.autograd import Variable


##############################################################################
#                                                                            #
#                               SKELETON CLASS                               #
#                                                                            #
##############################################################################

class ParameterizedTransformation(nn.Module):
    """ General class of transformations.
    All subclasses need the following methods:
    - norm: no args -> scalar variable
    - identity_params: shape -> TENSOR : takes an input shape and outputs
                       the subclass-specific parameter for the identity
                       transformation
    - forward : Variable -> Variable - is the transformation
    """

    def __init__(self, **kwargs):
        super(ParameterizedTransformation, self).__init__()

        if kwargs.get('manual_gpu', None) is not None:
            self.use_gpu = kwargs['manual_gpu']
        else:
            self.use_gpu = utils.use_gpu()


    def merge_xform(self, other, self_mask):
        """ Takes in an other instance of this same class with the same
            shape of parameters (NxSHAPE) and a self_mask bytetensor of length
            N and outputs the merge between self's parameters for the indices
            of 1s in the self_mask and other's parameters for the indices of 0's
        ARGS:
            other: instance of same class as self with params of shape NxSHAPE -
                   the thing we merge with this one
            self_mask : ByteTensor (length N) - which indices of parameters we
                        keep from self, and which we keep from other
        RETURNS:
            New instance of this class that's merged between the self and other
            (same shaped params)
        """

        # JUST DO ASSERTS IN THE SKELETON CLASS
        assert self.__class__ == other.__class__

        self_params = self.xform_params.data
        other_params = other.xform_params.data
        assert self_params.shape == other_params.shape
        assert self_params.shape[0] == self_mask.shape[0]
        assert other_params.shape[0] == self_mask.shape[0]

        new_xform = self.__class__(shape=self.img_shape)

        new_params = utils.fold_mask(self.xform_params.data,
                                     other.xform_params.data, self_mask)
        new_xform.xform_params = nn.Parameter(new_params)
        new_xform.use_gpu = self.use_gpu
        return new_xform


###############################################################################
#                                                                             #
#                  FULLY PARAMETERIZED SPATIAL TRANSFORMATION NETWORK         #
#                                                                             #
###############################################################################

class FullSpatial(ParameterizedTransformation):
    def __init__(self, *args, **kwargs):
        """ FullSpatial just has parameters that are the grid themselves.
            Forward then will just call grid sample using these params directly
        """

        super(FullSpatial, self).__init__(**kwargs)
        img_shape = kwargs['shape']
        self.img_shape = img_shape
        self.xform_params = nn.Parameter(self.identity_params(img_shape))


    def identity_params(self, shape):
        """ Returns some grid parameters such that the minibatch of images isn't
            changed when forward is called on it
        ARGS:
            shape: torch.Size - shape of the minibatch of images we'll be
                   transforming. First index should be num examples
        RETURNS:
            torch TENSOR (not variable!!!)
            if shape arg has shape NxCxHxW, this has shape NxCxHxWx2
        """

        # Work smarter not harder -- use idenity affine transforms here
        num_examples = shape[0]
        identity_affine_transform = torch.zeros(num_examples, 2, 3)
        if self.use_gpu:
            identity_affine_transform = identity_affine_transform.cuda()

        identity_affine_transform[:,0,0] = 1
        identity_affine_transform[:,1,1] = 1

        return F.affine_grid(identity_affine_transform, shape).data


    def stAdv_norm(self):
        """ Computes the norm used in
           "Spatially Transformed Adversarial Examples"
        """

        # ONLY WORKS FOR SQUARE MATRICES
        dtype = self.xform_params.data.type()
        num_examples, height, width = tuple(self.xform_params.shape[0:3])
        assert height == width
        ######################################################################
        #   Build permutation matrices                                       #
        ######################################################################

        def id_builder():
            x = torch.zeros(height, width).type(dtype)
            for i in range(height):
                x[i,i] = 1
            return x

        col_permuts = []
        row_permuts = []
        # torch.matmul(foo, col_permut)
        for col in ['left', 'right']:
            col_val = {'left': -1, 'right': 1}[col]
            idx = ((torch.arange(width) - col_val) % width)
            idx = idx.type(dtype).type(torch.LongTensor)
            if self.xform_params.is_cuda:
                idx = idx.cuda()

            col_permut = torch.zeros(height, width).index_copy_(1, idx.cpu(),
                                                                id_builder().cpu())
            col_permut = col_permut.type(dtype)

            if col == 'left':
                col_permut[-1][0] = 0
                col_permut[0][0] = 1
            else:
                col_permut[0][-1] = 0
                col_permut[-1][-1] = 1
            col_permut = Variable(col_permut)
            col_permuts.append(col_permut)
            row_permuts.append(col_permut.transpose(0, 1))

        ######################################################################
        #   Build delta_u, delta_v grids                                     #
        ######################################################################
        id_params = Variable(self.identity_params(self.img_shape))
        delta_grids = self.xform_params - id_params
        delta_grids = delta_grids.permute(0, 3, 1, 2)

        ######################################################################
        #   Compute the norm                                                 #
        ######################################################################
        output = Variable(torch.zeros(num_examples).type(dtype))

        for row_or_col, permutes in zip(['row', 'col'],
                                        [row_permuts, col_permuts]):
            for permute in permutes:
                if row_or_col == 'row':
                    temp = delta_grids - torch.matmul(permute, delta_grids)
                else:
                    temp = delta_grids - torch.matmul(delta_grids, permute)
                temp = temp.pow(2)
                temp = temp.sum(1)
                temp = (temp + 1e-10).pow(0.5)
                output.add_(temp.sum((1, 2)))
        return output


    def clip_params(self):
        """ Clips the parameters to be between -1 and 1 as required for
            grid_sample
        """
        clamp_params = torch.clamp(self.xform_params, -1, 1).data
        change_in_params = clamp_params - self.xform_params.data
        self.xform_params.data.add_(change_in_params)


    def merge_xform(self, other, self_mask):
        """ Takes in an other instance of this same class with the same
            shape of parameters (NxSHAPE) and a self_mask bytetensor of length
            N and outputs the merge between self's parameters for the indices
            of 1s in the self_mask and other's parameters for the indices of 0's
        """
        super(FullSpatial, self).merge_xform(other, self_mask)

        new_xform = FullSpatial(shape=self.img_shape,
                                manual_gpu=self.use_gpu)

        new_params = utils.fold_mask(self.xform_params.data,
                                     other.xform_params.data, self_mask)
        new_xform.xform_params = nn.Parameter(new_params)

        return new_xform


    def project_params(self, lp, lp_bound):
        """ Projects the params to be within lp_bound (according to an lp)
            of the identity map. First thing we do is clip the params to be
            valid, too
        ARGS:
            lp : int or 'inf' - which LP norm we use. Must be an int or the
                 string 'inf'
            lp_bound : float - how far we're allowed to go in LP land
        RETURNS:
            None, but modifies self.xform_params
        """

        assert isinstance(lp, int) or lp == 'inf'

        # clip first
        self.clip_params()

        # then project back

        if lp == 'inf':
            identity_params = self.identity_params(self.img_shape)
            clamp_params = utils.clamp_ref(self.xform_params.data,
                                               identity_params, lp_bound)
            change_in_params = clamp_params - self.xform_params.data
            self.xform_params.data.add_(change_in_params)
        else:
            raise NotImplementedError("Only L-infinity bounds working for now ")


    def forward(self, x):
        # usual forward technique
        return F.grid_sample(x, self.xform_params)
