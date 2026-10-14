# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/adversarial_attacks.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" Holds the various attacks we can do """

import logging

import torch
from torch.autograd import Variable
from torch import optim

from . import pytorch_utils as utils

logger = logging.getLogger(__name__)



###############################################################################
#                                                                             #
#                      PARENT CLASS FOR ADVERSARIAL ATTACKS                   #
#                                                                             #
###############################################################################

class AdversarialAttack(object):
    """ Wrapper for adversarial attacks. Is helpful for when subsidiary methods
        are needed.
    """

    def __init__(self, classifier_net, normalizer, threat_model,
                 manual_gpu=None):
        """ Initializes things to hold to perform a single batch of
            adversarial attacks
        ARGS:
            classifier_net : nn.Module subclass - neural net that is the
                             classifier we're attacking
            normalizer : DifferentiableNormalize object - object to convert
                         input data to mean-zero, unit-var examples
            threat_model : ThreatModel object - object that allows us to create
                           per-minibatch adversarial examples
            manual_gpu : None or boolean - if not None, we override the
                         environment variable 'MISTER_ED_GPU' for how we use
                         the GPU in this object

        """
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()
        self.threat_model = threat_model


##############################################################################
#                                                                            #
#                           PGD/FGSM^k/BIM                                   #
#                                                                            #
##############################################################################
# This goes by a lot of different names in the literature
# The key idea here is that we take many small steps of FGSM
# I'll call it PGD though

class PGD(AdversarialAttack):

    def __init__(self, classifier_net, normalizer, threat_model, loss_fxn,
                 manual_gpu=None):
        super(PGD, self).__init__(classifier_net, normalizer, threat_model,
                                  manual_gpu=manual_gpu)
        self.loss_fxn = loss_fxn  # WE MAXIMIZE THIS!!!

    def attack(self, examples, labels, num_iterations=20,
               optimizer=None, optimizer_kwargs=None,
               loss_convergence=0.999, keep_best=True, eot_iter=1):
        """ Builds PGD examples for the given examples with l_inf bound and
            given step size. Is almost identical to the BIM attack, except
            we take steps that are proportional to gradient value instead of
            just their sign.

        ARGS:
            examples: NxCxHxW tensor - for N examples, is NOT NORMALIZED
                      (i.e., all values are in between 0.0 and 1.0)
            labels: N longTensor - single dimension tensor with labels of
                    examples (in same order as examples)
            num_iterations: int or pair of ints - how many iterations we take.
                            If pair of ints, is of form (lo, hi), where we run
                            at least 'lo' iterations, at most 'hi' iterations
                            and we quit early if loss has stabilized.
            keep_best : bool - if True, we keep track of the best adversarial
                               perturbations per example (in terms of maximal
                               loss) in the minibatch. The output is the best of
                               each of these then
        RETURNS:
            AdversarialPerturbation object with correct parameters.
            Calling perturbation() gets Variable of output and
            calling perturbation().data gets tensor of output
        """

        ######################################################################
        #   Setups and assertions                                            #
        ######################################################################

        self.classifier_net.eval()

        perturbation = self.threat_model(examples)

        num_examples = examples.shape[0]
        var_examples = Variable(examples, requires_grad=True)
        var_labels = Variable(labels, requires_grad=False)

        if isinstance(num_iterations, int):
            min_iterations = num_iterations
            max_iterations = num_iterations
        elif isinstance(num_iterations, tuple):
            min_iterations, max_iterations = num_iterations

        best_perturbation = None
        if keep_best:
            best_loss_per_example = {i: None for i in range(num_examples)}

        prev_loss = None

        ######################################################################
        #   Loop through iterations                                          #
        ######################################################################

        self.loss_fxn.setup_attack_batch(var_examples)

        # Build the optimizer over the perturbation parameters
        optimizer = optimizer or optim.Adam
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.0001}
        optimizer = optimizer(perturbation.parameters(), **optimizer_kwargs)

        param_list = list(perturbation.parameters())
        assert len(param_list) == 1, len(param_list)
        param = param_list[0]
        logger.debug('PGD attack: eot_iter %d, max_iterations %d', eot_iter, max_iterations)
        for iter_no in range(max_iterations):
            perturbation.zero_grad()

            grad = torch.zeros_like(param)
            loss_per_example_ave = 0
            for i in range(eot_iter):
                loss_per_example = self.loss_fxn.forward(perturbation(var_examples), var_labels,
                                                         perturbation=perturbation,
                                                         output_per_example=keep_best)

                loss_per_example_ave += loss_per_example.detach().clone()
                loss = -1 * loss_per_example.sum()

                loss.backward()
                grad += param.grad.data.detach()
                param.grad.data.zero_()

            grad /= float(eot_iter)
            loss_per_example_ave /= float(eot_iter)

            assert len(param_list) == 1, len(param_list)
            param.grad.data = grad.clone()

            optimizer.step()

            if keep_best:
                mask_val = torch.zeros(num_examples, dtype=torch.uint8)
                for i, el in enumerate(loss_per_example_ave):
                    this_best_loss = best_loss_per_example[i]
                    if this_best_loss is None or this_best_loss[1] < float(el):
                        mask_val[i] = 1
                        best_loss_per_example[i] = (iter_no, float(el))

                if best_perturbation is None:
                    best_perturbation = self.threat_model(examples)

                best_perturbation = perturbation.merge_perturbation(
                    best_perturbation,
                    mask_val)

            # Stop early if loss didn't go down too much
            if (iter_no >= min_iterations and
                    float(loss) >= loss_convergence * prev_loss):
                break
            prev_loss = float(loss)

        perturbation.zero_grad()
        self.loss_fxn.cleanup_attack_batch()
        perturbation.attach_originals(examples)
        return perturbation
