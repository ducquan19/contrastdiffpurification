# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/adversarial_perturbations.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" File that holds adversarial perturbations as torch.nn.Modules.
    An adversarial perturbation is an example-specific
"""

import torch.nn as nn
from . import spatial_transformers as st
from . import pytorch_utils as utils
import functools


# assert initialized decorator
def initialized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.initialized, ("Parameters not initialized yet. "
                                  "Call .forward(...) first")
        return func(self, *args, **kwargs)
    return wrapper

##############################################################################
#                                                                            #
#                                   SKELETON CLASS                           #
#                                                                            #
##############################################################################

class AdversarialPerturbation(nn.Module):
    """ Skeleton class to hold adversarial perturbations FOR A SINGLE MINIBATCH.
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class

        All subclasses need the following:
        - perturbation_norm() : no args -> scalar Variable
        - self.parameters() needs to iterate over params we want to optimize
        - constrain_params() : no args -> no return,
             modifies the parameters such that this is still a valid image
        - forward : no args -> Variable - applies the adversarial perturbation
                    the originals and outputs a Variable of how we got there
        - adversarial_tensors() : applies the adversarial transform to the
                                  originals and outputs TENSORS that are the
                                  adversarial images
    """

    def __init__(self, threat_model, perturbation_params):

        super(AdversarialPerturbation, self).__init__()
        self.threat_model = threat_model
        self.initialized = False
        self.perturbation_params = perturbation_params

        if isinstance(perturbation_params, tuple):
            self.use_gpu = perturbation_params[1].use_gpu or utils.use_gpu()
        else:
            self.use_gpu = perturbation_params.use_gpu or utils.use_gpu()
        # Stores parameters of the adversarial perturbation and hyperparams
        # to compute total perturbation norm here


    def __call__(self, x):
        return self.forward(x)


    def setup(self, x):
        """ This is the standard setup technique and should be used to
            initialize a perturbation (i.e. sets up parameters and unlocks
            all other methods)
        ARGS:
            x : Variable or Tensor (NxCxHxW) - the images this perturbation is
                intended for
        """
        self.num_examples = x.shape[0]


    @initialized
    def attach_attr(self, attr_name, attr):
        """ Special method to set an attribute if it doesn't exist in this
            object yet. throws error if this attr already exists
        ARGS:
            attr_name : string - name of attribute we're attaching
            attr: object - attribute we're attaching
        RETURNS:
            None
        """
        if hasattr(self, attr_name):
            raise Exception("%s already has attribute %s" % (self, attr_name))
        else:
            setattr(self, attr_name, attr)


    @initialized
    def attach_originals(self, originals):
        """ Little helper method to tack on the original images to self to
            pass around the (images, perturbation) in a single object
        """
        self.attach_attr('originals', originals)


    @initialized
    def merge_perturbation(self, other, self_mask):
        """ Special technique to merge this perturbation with another
            perturbation of the same threat model.
            This will return a new perturbation object that, for each parameter
            will return the parameters of self for self_mask, and the
            perturbation of other for NOT(self_mask)

        ARGS:
            other: AdversarialPerturbation Object - instance of other
                   adversarial perturbation that is instantiated with the
                   same threat model as self
            self_indices: ByteTensor [N] : bytetensor indicating which
                          parameters to include from self and which to include
                          from other
        """

        # this parent class just does the shared asserts such that this is a
        # valid thing
        assert self.__class__ == other.__class__
        assert self.threat_model == other.threat_model
        assert self.num_examples == other.num_examples
        assert self.perturbation_params == other.perturbation_params
        assert other.initialized


class PerturbationParameters(dict):
    """ Object that stores parameters like a dictionary.
        This allows perturbation classes to be only partially instantiated and
        then fed various 'originals' later.
    Implementation taken from : https://stackoverflow.com/a/14620633/3837607
    (and then modified with the getattribute trick to return none instead of
     error for missing attributes)
    """
    def __init__(self, *args, **kwargs):
        super(PerturbationParameters, self).__init__(*args, **kwargs)
        if kwargs.get('manual_gpu') is not None:
            self.use_gpu = kwargs['manual_gpu']
        else:
            self.use_gpu = utils.use_gpu()
        self.__dict__ = self

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None


class ThreatModel(object):
    def __init__(self, perturbation_class, param_kwargs, *other_args):
        """ Factory class to generate per_minibatch instances of Adversarial
            perturbations.
        ARGS:
            perturbation_class : class - subclass of Adversarial Perturbations
            param_kwargs : dict - dict containing named kwargs to instantiate
                           the class in perturbation class
        """
        assert issubclass(perturbation_class, AdversarialPerturbation)
        self.perturbation_class = perturbation_class
        if isinstance(param_kwargs, dict):
            param_kwargs = PerturbationParameters(**param_kwargs)
        self.param_kwargs = param_kwargs
        self.other_args = other_args


    def __call__(self, *args):
        if args == ():
            return self.perturbation_obj()
        else:
            perturbation_obj = self.perturbation_obj()
            perturbation_obj.setup(*args)
            return perturbation_obj


    def perturbation_obj(self):
        return self.perturbation_class(self, self.param_kwargs, *self.other_args)


##############################################################################
#                                                                            #
#                               SPATIAL PARAMETERS                           #
#                                                                            #
##############################################################################

class ParameterizedXformAdv(AdversarialPerturbation):

    def __init__(self, threat_model, perturbation_params, *other_args):
        super(ParameterizedXformAdv, self).__init__(threat_model,
                                                    perturbation_params)
        assert issubclass(perturbation_params.xform_class,
                          st.ParameterizedTransformation)

        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound


    def _merge_setup(self, num_examples, new_xform):
        """ DANGEROUS TO BE CALLED OUTSIDE OF THIS FILE!!!"""
        self.num_examples = num_examples
        self.xform = new_xform
        self.initialized = True

    def setup(self, originals):
        super(ParameterizedXformAdv, self).setup(originals)
        self.xform = self.perturbation_params.xform_class(shape=originals.shape,
                                                        manual_gpu=self.use_gpu)
        self.initialized = True

    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        # flow smoothness loss of stAdv (Xiao et al., 2018)
        assert isinstance(self.xform, st.FullSpatial)
        return self.xform.stAdv_norm()

    @initialized
    def constrain_params(self, x=None):
        # Do lp projections
        if isinstance(self.lp_style, int) or self.lp_style == 'inf':
            self.xform.project_params(self.lp_style, self.lp_bound)


    @initialized
    def merge_perturbation(self, other, self_mask):
        super(ParameterizedXformAdv, self).merge_perturbation(other, self_mask)
        new_perturbation = ParameterizedXformAdv(self.threat_model,
                                                 self.perturbation_params)

        new_xform = self.xform.merge_xform(other.xform, self_mask)
        new_perturbation._merge_setup(self.num_examples, new_xform)

        return new_perturbation


    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.constrain_params()
        return self.xform.forward(x)
