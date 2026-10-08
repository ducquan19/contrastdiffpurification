import time

import numpy as np
import torch

from .autopgd_base import APGDAttack, APGDAttack_targeted
from .fab import FABAttack
from .other_utils import Logger
from .square import SquareAttack
from . import checks
from .state import EvaluationState
import warnings
import pathlib


class AutoAttack():
    """AutoAttack ``standard`` (APGD-CE, APGD-T, FAB-T, Square) or ``rand`` (APGD-CE/DLR + EOT).

    Linf and L2 threat models; the model is a PyTorch module returning logits.
    """

    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.device = device
        self.logger = Logger(log_path)

        if version in ['standard', 'rand'] and attacks_to_run != []:
            warnings.warn("attacks_to_run will be overridden unless you use version='custom'")

        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
            device=self.device, logger=self.logger)

        self.fab = FABAttack(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False, device=self.device)

        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)

        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            logger=self.logger)

        if version in ['standard', 'rand']:
            self.set_version(version)

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self,
                                x_orig,
                                y_orig,
                                bs=250,
                                return_labels=False,
                                state_path=None):
        if isinstance(state_path, str):
            state_path = pathlib.Path(state_path)
        new_state = False
        if state_path is not None and state_path.exists():
            try:
                state = EvaluationState.from_disk(state_path)
                if set(self.attacks_to_run) != state.attacks_to_run:
                    warnings.warn("The state was created with a different set of attacks "
                                    "to run. You are probably using the wrong state file. start from begining.")
                    new_state=True
                elif self.verbose:
                    self.logger.log("Restored state from {}".format(state_path))
                    self.logger.log("Since the state has been restored, **only** "
                                    "the adversarial examples from the current run "
                                    "are going to be returned.")
            except Exception:
                new_state = True
        else:
            new_state = True
        
        if new_state:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log("Created state in {}".format(state_path))                                

        attacks_to_run = list(filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run))
        if self.verbose:
            self.logger.log('using {} version including {}.'.format(self.version,
                  ', '.join(attacks_to_run)))
            if state.run_attacks:
                self.logger.log('{} was/were already run.'.format(', '.join(state.run_attacks)))

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                    correct_batch = y.eq(output)
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': robust_accuracy}
                state.clean_accuracy = robust_accuracy
                
                if self.verbose:
                    self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
            else:
                # copy before;
                # fix  UnboundLocalError: local variable 'y_adv' referenced before assignment
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                # end of copy

                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': state.clean_accuracy}
                if self.verbose:
                    self.logger.log('initial clean accuracy: {:.2%}'.format(state.clean_accuracy))
                    self.logger.log('robust accuracy at the time of restoring the state: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            state.to_disk(force=True)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))

        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self.apgd.n_restarts = 1
            self.apgd_targeted.n_target_classes = 9
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.square.n_queries = 5000

        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20
