# Confidence-routed cascade DDS (training-free)

## Motivation

* Deterministic multi-step DDS (DDIM, eta = 0) beats one-shot DDS, but it is not new: it was
  observed by Carlini et al. (2023) and certified as the "PF-ODE EDM" baseline in Consistency
  Purification (NeurIPS 2024).
* Its gain depends on the classifier. With a standard WRN-28-10 (CIFAR-10 only), multi-step
  improves on one-shot by 11–19 points (our main table). With the ViT-B/16 (ImageNet-21k) used
  by Carlini et al. and Consistency Purification, the gain is only 1–2 points. The bottleneck is
  the classifier's covariate shift on denoised, low-pass images.
* Multi-step costs S times more score-network evaluations (NFE) for every one of the N
  Monte-Carlo samples.

## Method

The base classifier of randomized smoothing, for ONE noisy input `x~ = x + sigma xi`:

1. `m = onestep(x~)` (1 NFE); `p = head(m)`.
2. If `max_c p_c >= tau`: answer `argmax p`. Otherwise continue the SAME deterministic DDIM
   trajectory (it reuses the score evaluation behind `m`), `S - 1` more NFE, and answer
   `argmax head(multistep(x~))`.

The head is a set of off-the-shelf classifiers, with no training:

* `single`;
* `ens` (mean softmax);
* `route`: the most confident member (`heads.ConfidenceRouter`). The candidates are a standard
  WRN, l2-adversarially-trained RobustBench WRNs (low-frequency-biased, so plausibly less
  shifted by the denoiser's blur) and the ViT-B/16.

Mean cost: `1 + (S - 1) * P[conf < tau]` NFE per sample.

**Validity.** `h` is a deterministic function of `x~` (eta = 0), and the classifiers and `tau` are
fixed before certification. Cohen et al.'s theorem therefore applies unchanged. `tau` and the head
are chosen on the dev split only. `tau <= 0` gives exactly one-shot and `tau > 1` exactly
multi-step (tests/test_cascade.py).

## Protocol

1. `fetch` (login node): install transformers, download the checkpoints, and report the clean
   dev accuracy of each classifier (sanity check of the weights and the label order).
2. `screen` (1 GPU job): `scripts/screen_cascade.py`. 200 dev images, 200 draws, one-shot plus
   S = 5 and S = 10 on the same draws, probabilities of all 4 classifiers saved.
3. `analyze` (CPU): `scripts/analyze_cascade.py`. It scores every head and every cascade
   offline, with paired bootstrap CIs against multi-step + standard WRN. `tau` is 2-fold
   cross-fitted over images: the fewest-NFE `tau` that keeps 99% of the head's own multi-step
   acc/draw.
4. **Gate.** Two criteria:
   * accuracy: the best head in multi-step beats the reference by at least 3 points acc/draw,
     with CI > 0;
   * cost: the best cascade keeps at least 98% of its multi-step acc at no more than 50% of the
     NFE.
5. `main` (CERTIFY, 500 test images, N = 10k, alpha = 0.001):
   * ViT-B/16 one-shot and multi-step, for the fair comparison with the published ViT numbers;
   * the chosen head as multi-step and as cascade.

   Report certified accuracy, ACR and the mean NFE.

## Comparison rules

Compare only at equal classifier and with the protocol stated:

* N, number of images, alpha;
* the diffusion model (ours: score_sde VP; Carlini: improved-diffusion 50M; CP: EDM);
* extra data of the classifiers (ViT: ImageNet-21k; Wang2023Better / Rebuffi2021: generated data).
