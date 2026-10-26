"""Certified l2 robustness with diffusion denoised smoothing (docs/wadds.md).

* ``smoothing``  -- Cohen et al. (2019) CERTIFY with a Clopper-Pearson bound, metrics;
* ``denoisers``  -- the base-classifier denoisers: Carlini et al. one-shot, multi-step DDIM and
  WA-DDS (Wiener-anchored sampling);
* ``spectrum``   -- the dataset power spectrum P(f) and the Wiener gain W = P / (P + sigma_d^2).

Nothing here imports ``score_sde`` or ``robustbench``: the score network enters through ``eps_fn``
and the classifier through a callable, so the package is testable on CPU (tests/test_certify.py).
"""
