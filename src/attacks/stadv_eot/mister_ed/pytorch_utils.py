# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/utils/pytorch_utils.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" Utilities for general pytorch helpfulness """

import torch
import numpy as np
import torch.cuda as cuda
import os
import warnings
from torch.autograd import Variable


###############################################################################
#                                                                             #
#                                     SAFETY DANCE                            #
#                                                                             #
###############################################################################
# aka things for safer pytorch usage


def use_gpu():
    """ The shortcut to retrieve the environment variable 'MISTER_ED_GPU'"""
    try:
        str_val = os.environ['MISTER_ED_GPU']
    except:
        set_global_gpu()
        str_val = os.environ['MISTER_ED_GPU']
    assert str_val in ['True', 'False']
    return str_val == 'True'


def set_global_gpu(manual=None):
    """ Sets the environment variable 'MISTER_ED_GPU'. Defaults to using gpu
        if cuda is available
    ARGS:
        manual : bool - we set the 'MISTER_ED_GPU' environment var to the string
                 of whatever this is
    RETURNS
        None
    """
    if manual is None:
        val = cuda.is_available()
    else:
        val = manual
    os.environ['MISTER_ED_GPU'] = str(val)


def safe_tensor(entity):
    """ Returns a tensor of an entity, which may or may not already be a
        tensor
    """
    warnings.warn("As of >=pytorch0.4.0 this is no longer necessary",
                  DeprecationWarning)
    if isinstance(entity, Variable):
        return entity.data
    elif isinstance(entity, torch.tensor._TensorBase):
        return entity
    elif isinstance(entity, np.ndarray):
        return torch.Tensor(entity)  # UNSAFE CUDA CASTING
    else:
        raise Exception("Can't cast %s to a Variable" %
                        entity.__class__.__name__)


##############################################################################
#                                                                            #
#                               CONVENIENCE STORE                            #
#                                                                            #
##############################################################################
# aka convenient things that are not builtin to pytorch


def clamp_ref(x, y, l_inf):
    """ Clamps each element of x to be within l_inf of each element of y """
    return torch.clamp(x - y, -l_inf, l_inf) + y


def fold_mask(x, y, mask):
    """ Creates a new tensor that's the result of masking between x and y
    ARGS:
        x : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 1
        y : Tensor or Variable (NxSHAPE) - tensor that we're selecting where the
            masked values are 0
        mask: ByteTensor (N) - masked values. Is only one dimensional: we expand
              it in the creation of this
    RETURNS:
        new object of the same shape/type as x and y
    """
    assert x.shape == y.shape
    assert mask.shape == (x.shape[0],)
    assert type(x) == type(y)
    is_var = isinstance(x, Variable)
    if is_var:
        assert isinstance(mask, Variable)

    make_broadcastable = lambda m: m.view(-1, *tuple([1] * (x.dim() - 1)))

    broadcast_mask = make_broadcastable(mask)
    broadcast_not_mask = make_broadcastable(1 - safe_tensor(mask))
    if is_var:
        broadcast_not_mask = Variable(broadcast_not_mask)

    output = torch.zeros_like(x)
    output.add_(x * (broadcast_mask.type(x.type())))
    output.add_(y * (broadcast_not_mask.type(y.type())))

    return output
