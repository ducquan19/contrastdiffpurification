# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""Attack evaluation for classifiers and diffusion purifiers.

Every ``eval_*`` function attacks the bare classifier first and then the purified ``model``,
prints clean / robust accuracy and saves the adversarial examples as
``x_adv_{resnet,sde}_sd<seed>.pt`` in ``log_dir``. ``model`` may be wrapped in
``nn.DataParallel``; its counter and tag are reset through the unwrapped module.
"""

import time

import torch
import torchattacks
from src.attacks.autoattack import AutoAttack

from src.evaluation.metrics import get_accuracy
from src.evaluation.purified_model import UndefendedClassifier
from src.attacks.bpda_eot.bpda_eot_attack import BPDA_EOT_Attack
from src.attacks.stadv_eot.attacks import StAdvAttack
from src.models.classifiers import get_image_classifier


def _unwrap(model):
    """The module inside ``nn.DataParallel`` (multi-GPU), else ``model`` itself."""
    return model.module if torch.cuda.device_count() > 1 else model


def _save_adv(x_adv, y, log_dir, target, seed):
    print(f'x_adv_{target} shape: {x_adv.shape}')
    torch.save([x_adv, y], f'{log_dir}/x_adv_{target}_sd{seed}.pt')


def eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """AutoAttack: ``standard`` (APGD-CE, APGD-T, FAB-T, Square) or ``rand`` (APGD-CE/DLR with EOT).

    Progress is checkpointed to ``adv_*_eval_state.json`` so an interrupted run can resume.
    """
    if args.attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif args.attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    else:
        raise NotImplementedError(f'Unknown attack version: {args.attack_version}!')

    print(f"attack_version: {args.attack_version}", f"attack_list: {attack_list}")

    def run(target_model, name, log_name, state_name):
        adversary = AutoAttack(
            target_model,
            norm=args.lp_norm,
            eps=args.adv_eps,
            version=args.attack_version,
            attacks_to_run=attack_list,
            log_path=f'{log_dir}/{log_name}',
            device=config.device)

        if args.attack_version == 'rand':
            adversary.apgd.eot_iter = args.eot_iter
            print(f'[{name}] rand version with eot_iter: {adversary.apgd.eot_iter}')

        print(f'{args.lp_norm}, epsilon: {args.adv_eps}')
        return adversary.run_standard_evaluation(
            x_val, y_val, bs=adv_batch_size, state_path=f"{log_dir}/{state_name}"
        )

    print(f'apply the attack to classifier [{args.lp_norm}]...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)
    x_adv = run(classifier, 'classifier', 'log_resnet.txt', 'adv_resnet_eval_state.json')
    _save_adv(x_adv, y_val, log_dir, 'resnet', args.seed)

    print(f'apply the attack to sde_adv [{args.lp_norm}]...')
    _unwrap(model).reset_counter()
    x_adv = run(model, 'adv_sde', 'log_sde_adv.txt', 'adv_sde_eval_state.json')
    _save_adv(x_adv, y_val, log_dir, 'sde', args.seed)


def _eval_with_attack(args, config, model, x_val, y_val, adv_batch_size, log_dir, name, build_attack):
    """Attack the classifier, then the purified model, with ``build_attack(target)(x, y)``.

    Clean and robust accuracy are measured with ``get_accuracy``; the purified model's batches
    are tagged ``no_adv`` / (none) / ``sde_adv`` for the clean pass, the attack and the re-check.
    """
    model_ = _unwrap(model)
    x_val, y_val = x_val.to(config.device), y_val.to(config.device)
    print(f'bound: {args.adv_eps}')

    print(f'apply the {name} attack to resnet...')
    resnet = get_image_classifier(args.classifier_name).to(config.device)

    start_time = time.time()
    init_acc = get_accuracy(resnet, x_val, y_val, bs=adv_batch_size)
    print(f'initial accuracy: {init_acc:.2%}', f'time elapsed: {time.time() - start_time:.2f}s')

    adversary = build_attack(resnet)
    start_time = time.time()
    x_adv = adversary(x_val, y_val)
    robust_acc = get_accuracy(resnet, x_adv, y_val, bs=adv_batch_size)
    print(f'robust accuracy: {robust_acc:.2%}', f'time elapsed: {time.time() - start_time:.2f}s')
    _save_adv(x_adv, y_val, log_dir, 'resnet', args.seed)

    print(f'apply the {name} attack to sde_adv...')
    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    print(f'initial accuracy: {init_acc:.2%}', f'time elapsed: {time.time() - start_time:.2f}s')

    adversary = build_attack(model)
    start_time = time.time()
    model_.reset_counter()
    model_.set_tag()
    x_adv = adversary(x_val, y_val)

    model_.reset_counter()
    model_.set_tag('sde_adv')
    robust_acc = get_accuracy(model, x_adv, y_val, bs=adv_batch_size)
    print(f'robust accuracy: {robust_acc:.2%}', f'time elapsed: {time.time() - start_time:.2f}s')
    _save_adv(x_adv, y_val, log_dir, 'sde', args.seed)


def eval_stadv(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """Spatially transformed adversarial examples (StAdv, 100 iterations, EOT)."""
    _eval_with_attack(
        args, config, model, x_val, y_val, adv_batch_size, log_dir, 'stadv',
        lambda target: StAdvAttack(target, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter))


def eval_eotpgd(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """EOT-PGD from torchattacks (10 steps, step size 2/255)."""
    _eval_with_attack(
        args, config, model, x_val, y_val, adv_batch_size, log_dir, 'eotpgd',
        lambda target: torchattacks.EOTPGD(
            target, eps=args.adv_eps, alpha=2.0/255, steps=10, eot_iter=args.eot_iter))


def eval_apgd(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """APGD-CE from torchattacks (10 steps, EOT)."""
    _eval_with_attack(
        args, config, model, x_val, y_val, adv_batch_size, log_dir, 'apgd',
        lambda target: torchattacks.APGD(
            target, eps=args.adv_eps, steps=10, eot_iter=args.eot_iter, verbose=True))


def eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    """BPDA+EOT against the undefended classifier, then against the purified ``model``."""
    ngpus = torch.cuda.device_count()
    model_ = _unwrap(model)
    x_val, y_val = x_val.to(config.device), y_val.to(config.device)

    def run(target_model, target):
        adversary = BPDA_EOT_Attack(
            target_model,
            adv_eps=args.adv_eps,
            eot_defense_reps=args.eot_defense_reps,
            eot_attack_reps=args.eot_attack_reps
        )
        start_time = time.time()
        class_batch, ims_adv_batch = adversary.attack_all(x_val, y_val, batch_size=adv_batch_size)
        # row 0: correct before the attack, last row: correct after it
        init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
        robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
        print(
            f"init acc: {init_acc:.2%}",
            f"robust acc: {robust_acc:.2%}",
            f"time elapsed: {time.time() - start_time:.2f}s"
        )
        _save_adv(ims_adv_batch, y_val, log_dir, target, args.seed)

    print('apply the bpda attack to resnet...')
    resnet_bpda = UndefendedClassifier(args, config)
    if ngpus > 1:
        resnet_bpda = torch.nn.DataParallel(resnet_bpda)

    start_time = time.time()
    init_acc = get_accuracy(resnet_bpda, x_val, y_val, bs=adv_batch_size)
    print(f'initial accuracy: {init_acc:.2%}', f'time elapsed: {time.time() - start_time:.2f}s')
    run(resnet_bpda, 'resnet')

    print('apply the bpda attack to sde_adv...')
    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    print(f"initial accuracy: {init_acc:.2%}", f"time elapsed: {time.time() - start_time:.2f}s")

    model_.reset_counter()
    model_.set_tag()
    run(model, 'sde')
