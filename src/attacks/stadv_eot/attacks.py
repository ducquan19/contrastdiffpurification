# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from torch import nn
from torch import optim

# mister_ed
from .mister_ed import loss_functions as lf
from .mister_ed import adversarial_training as advtrain
from .mister_ed import adversarial_perturbations as ap
from .mister_ed import adversarial_attacks as aa
from .mister_ed import spatial_transformers as st


PGD_ITERS = 20



class MisterEdAttack(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, perturbation_norm_loss=False, lr=0.001, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.perturbation_norm_loss = perturbation_norm_loss
        self.attack_kwargs = kwargs
        self.lr = lr

        self.attack = None

    def _setup_attack(self):
        cw_loss = lf.CWLossF6(self.model, self.normalizer, kappa=float('inf'))
        perturbation_loss = lf.PerturbationNormLoss(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = lf.RegularizedLoss({
            'cw': cw_loss,
            'pert': perturbation_loss,
        }, {
            'cw': 1.0,
            'pert': pert_factor,
        }, negate=True)

        self.pgd_attack = aa.PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'num_iterations': PGD_ITERS,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = advtrain.AdversarialAttackParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        return self.attack.attack(inputs, labels)[0]


class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )
