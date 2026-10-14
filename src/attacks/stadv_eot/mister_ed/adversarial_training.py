# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/adversarial_training.py
#
# The license for the original version of this file can be
# found in `src/attacks/stadv_eot/LICENSE_RECOLORADV`.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" Parameters of an adversarial attack and a helper that runs it on a minibatch """

from torch.autograd import Variable

import random


##############################################################################
#                                                                            #
#                               ATTACK PARAMETERS OBJECT                     #
#                                                                            #
##############################################################################

class AdversarialAttackParameters(object):
    """ Wrapper to store an adversarial attack object as well as some extra
        parameters for how to use it in training
    """

    def __init__(self, adv_attack_obj, proportion_attacked=1.0,
                 attack_specific_params=None):
        """ Stores params for how to use adversarial attacks in training
        ARGS:
            adv_attack_obj : AdversarialAttack subclass -
                             thing that actually does the attack
            proportion_attacked: float between [0.0, 1.0] - what proportion of
                                 the minibatch we build adv examples for
            attack_specific_params: possibly None dict, but possibly dict with
                                    specific parameters for attacks

        """
        self.adv_attack_obj = adv_attack_obj
        self.proportion_attacked = proportion_attacked

        attack_specific_params = attack_specific_params or {}
        self.attack_specific_params = attack_specific_params
        self.attack_kwargs = attack_specific_params.get('attack_kwargs', {})

    def set_gpu(self, use_gpu):
        """ Propagates changes of the 'use_gpu' parameter down to the attack
        ARGS:
            use_gpu : bool - if True, the attack uses the GPU, ow it doesn't
        RETURNS:
            None
        """
        self.adv_attack_obj.use_gpu = use_gpu

    def attack(self, inputs, labels):
        """ Builds some adversarial examples given real inputs and labels
        ARGS:
            inputs : torch.Tensor (NxCxHxW) - tensor with examples needed
            labels : torch.Tensor (N) - tensor with the examples needed
        RETURNS:
            some sample of (self.proportion_attacked * N ) examples that are
            adversarial, and the corresponding NONADVERSARIAL LABELS

            output is a tuple with three tensors:
             (adv_examples, pre_adv_labels, selected_idxs, coupled )
             adv_examples: Tensor with shape (N'xCxHxW) [the perturbed outputs]
             pre_adv_labels: Tensor with shape (N') [original labels]
             selected_idxs : Tensor with shape (N') [idxs selected]
             adv_inputs : Tensor with shape (N') [examples used to make advs]
             perturbation: Adversarial Perturbation Object
        """
        num_elements = inputs.shape[0]

        selected_idxs = sorted(random.sample(list(range(num_elements)),
                                             int(self.proportion_attacked * num_elements)))

        selected_idxs = inputs.new(selected_idxs).long()
        if selected_idxs.numel() == 0:
            return (None, None, None)

        adv_inputs = Variable(inputs.index_select(0, selected_idxs))
        pre_adv_labels = labels.index_select(0, selected_idxs)

        perturbation = self.adv_attack_obj.attack(adv_inputs.data,
                                                  pre_adv_labels,
                                                  **self.attack_kwargs)
        adv_examples = perturbation(adv_inputs)

        return (adv_examples, pre_adv_labels, selected_idxs, adv_inputs,
                perturbation)
