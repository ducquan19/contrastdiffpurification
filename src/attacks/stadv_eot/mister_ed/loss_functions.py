# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/loss_functions.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch
from numbers import Number
from . import adversarial_perturbations as ap

""" Loss function building blocks """


##############################################################################
#                                                                            #
#                        LOSS FUNCTION WRAPPER                               #
#                                                                            #
##############################################################################

class RegularizedLoss(object):
    """ Wrapper for multiple PartialLoss objects where we combine with
        regularization constants """

    def __init__(self, losses, scalars, negate=False):
        """
        ARGS:
            losses : dict - dictionary of partialLoss objects, each is keyed
                            with a nice identifying name
            scalars : dict - dictionary of scalars, each is keyed with the
                             same identifying name as is in self.losses
            negate : bool - if True, we negate the whole thing at the end
        """

        assert sorted(losses.keys()) == sorted(scalars.keys())

        self.losses = losses
        self.scalars = scalars
        self.negate = negate

    def forward(self, examples, labels, *args, **kwargs):

        output = None
        output_per_example = kwargs.get('output_per_example', False)
        for k in self.losses:
            loss = self.losses[k]
            scalar = self.scalars[k]

            loss_val = loss.forward(examples, labels, *args, **kwargs)
            # assert scalar is either a...
            assert (isinstance(scalar, float) or  # number
                    scalar.numel() == 1 or  # tf wrapping of a number
                    scalar.shape == loss_val.shape)  # same as the loss_val

            addendum = loss_val * scalar
            if addendum.numel() > 1:
                if not output_per_example:
                    addendum = torch.sum(addendum)

            if output is None:
                output = addendum
            else:
                output = output + addendum
        if self.negate:
            return output * -1
        else:
            return output

    def setup_attack_batch(self, fix_im):
        """ Setup before calling loss on a new minibatch: zeros all grads
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        for loss in self.losses.values():
            loss.zero_grad()

    def cleanup_attack_batch(self):
        """ Does some cleanup stuff after we finish on a minibatch:
        - zeros grads
        - clears example-based scalars (i.e. scalars that depend on which
          example we're using)
        """
        for loss in self.losses.values():
            loss.zero_grad()

        for key, scalar in self.scalars.items():
            if not isinstance(scalar, Number):
                self.scalars[key] = None


class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """

    def __init__(self):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()


##############################################################################
#                           Carlini Wagner loss functions                    #
##############################################################################

class CWLossF6(PartialLoss):
    def __init__(self, classifier, normalizer=None, kappa=0.0):
        super(CWLossF6, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)
        self.kappa = kappa

    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get target logits
        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))

        # get largest non-target logits
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max

        # untargeted: push the true-class logit below the best other class
        f6 = torch.clamp(target_logits - max_other, min=-1 * self.kappa)

        return f6.squeeze()


##############################################################################
#                                                                            #
#                           SPATIAL LOSS FUNCTIONS                           #
#                                                                            #
##############################################################################


class PerturbationNormLoss(PartialLoss):

    def __init__(self, **kwargs):
        super(PerturbationNormLoss, self).__init__()

        lp = kwargs.get('lp', 2)
        assert lp in [1, 2, 'inf']
        self.lp = lp

    def forward(self, examples, *args, **kwargs):
        """ Computes perturbation norm and multiplies by scale
        There better be a kwarg with key 'perturbation' which is a perturbation
        object with a 'perturbation_norm' method that takes 'lp_style' as a
        kwarg
        """

        perturbation = kwargs['perturbation']
        assert isinstance(perturbation, ap.AdversarialPerturbation)

        return perturbation.perturbation_norm(lp_style=self.lp)
