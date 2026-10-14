# Kiến trúc nghiên cứu

Code hiện thực Contrastive DiffPure (thanh lọc bằng diffusion có contrastive guidance) như thế nào.
Mô tả dưới đây theo đúng mã nguồn; các link trỏ tới cấu trúc hiện tại.

## 1. Pipeline

```
experiments/<nhóm>/*.sh             đúng cấu hình đã công bố (CIFAR-10)
   │  python -m scripts.eval_robustness | scripts.eval_bpda  --guidance {none,contrastive}
   ▼
src/evaluation/robustness.py        parse tham số → setup(): config YAML, logging, seed, thư mục output
   ▼
src/data/loaders.py::load_data      subset ngẫu nhiên cố định (num_sub ảnh, chọn theo data_seed) của tập test, pixel trong [0,1]
   ▼
src/evaluation/purified_model.py    PurifiedClassifier.forward(x):
   │   classifier  = get_image_classifier(--classifier_name)
   │   runner      = SDEPurifier (sde) | ODEPurifier (ode)
   │   [contrastive] runner.model ← ContrastiveGuidedDiffusion(runner.model, …)
   │   x ∈ [0,1] → [-1,1] → thêm nhiễu tới t* → giải reverse SDE/ODE → classifier
   ▼
src/evaluation/attacks.py           AutoAttack (rand / standard) · EOT-PGD · APGD · StAdv · BPDA+EOT
   │                                mỗi attack chạy trên classifier trần, rồi trên purifier + classifier
   │                                LƯU Ý: gradient của purifier bị che khỏi attacker
   │                                (straight-through); xem "Threat model" ở §7
   ▼
outputs/<exp>/<i>/<classifier>/<diffusion>_<attack>/seed<S>/data<D>/
                                    log.txt, log_resnet.txt, log_sde_adv.txt, x_adv_*.pt, ảnh đã thanh lọc
```

Phương pháp **không có giai đoạn huấn luyện**. Mạng embedding của contrastive được khởi tạo lại và
tối ưu trực tuyến trong mỗi lần thanh lọc (§3). Các script huấn luyện trong `scripts/` chỉ tạo ra
diffusion model và classifier cho CIFAR-100 / GTSRB. CIFAR-10 dùng checkpoint công khai.

## 2. Thanh lọc (DiffPure)

| `--diffusion_type` | Lớp | File | Solver |
|---|---|---|---|
| `sde` | `SDEPurifier` + `RevVPSDE` | [src/models/diffusion/sde.py](../src/models/diffusion/sde.py) | `torchsde.sdeint_adjoint`, Euler |
| `ode` | `ODEPurifier` + `VPODE` | [src/models/diffusion/ode.py](../src/models/diffusion/ode.py) | `torchdiffeq.odeint_adjoint`, Euler (`--step_size`) |

`image_editing_sample(img)`:
1. Thêm nhiễu xuôi tới `t = --t` (trên thang 1000): `x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε`.
2. Tích phân ngược VP-SDE (hoặc probability-flow ODE) về `t ≈ 0` bằng adjoint solver. Adjoint
   solver *có thể* cung cấp gradient chính xác cho attack thích nghi, nhưng phần đánh giá ở đây
   không dùng khả năng đó — xem "Threat model" ở §7.
3. Lặp `--sample_step` lần rồi nối kết quả lại.

Score network là NCSN++ của `score_sde`, với score type `score_sde`. Checkpoint được chọn theo
`config.data.dataset`:

| dataset | checkpoint (trong `pretrained/`) |
|---|---|
| CIFAR10 | `score_sde/checkpoint_8.pth` (score_sde_pytorch `vp/cifar10_ddpmpp_deep_continuous`, dùng trọng số EMA) |
| CIFAR100 | `cifar100/cifar100_score_sde.pt` (từ `scripts/train_score_sde.py`) |
| GTSRB | `gtsrb/gtsrb_score_sde.pt` (từ `scripts/train_score_sde.py`) |

## 3. Phương pháp: contrastive guidance

[src/guidance/contrastive.py](../src/guidance/contrastive.py) định nghĩa
`ContrastiveGuidedDiffusion(nn.Module)`, lớp bọc quanh score network.

| thuộc tính | vai trò |
|---|---|
| `model` | score network gốc (NCSN++) |
| `classifier` | classifier đóng băng ở chế độ eval (`--contrastive_classifier`) |
| `embed_net` | `MLP(n_classes → 2·n_classes → n_classes)`, 3 block gated residual — chính là F(·) |
| `optimizer` | Adam trên `embed_net` (`embed_lr`) |
| `contrastive_loss` | `ContrastiveLoss(tau_minus, beta, temperature, estimator)`: InfoNCE debiased / hard-negative |
| `xt_cache`, `drift_counter` | `x` của lần gọi solver trước; số lần gọi kể từ lúc reset |

`forward(x, t)`:
1. `out = model(x, t)`.
2. Guidance bị bỏ qua ở lần gọi đầu và khi ở ngoài khoảng `[drift_min, drift_max)`.
   Ngược lại:
   * `z0 = embed_net(classifier(xt_cache))` là anchor, lấy từ bước solver trước;
   * `zd = embed_net(classifier(x + δ))`, với δ = 0 và gradient được bật;
   * `loss = ContrastiveLoss(z0, zd)`; negative là các sample khác trong cùng batch;
   * backward, rồi **một bước Adam trên `embed_net`**;
   * `delta = ∂loss/∂δ · guidance_strength`.
3. `out[:, :C] += delta`, chỉ C kênh đầu bị thay đổi (để xử lý trường hợp đầu ra có learned sigma).
4. Lưu `x` vào cache và tăng `drift_counter`.

Sau mỗi batch đã thanh lọc, `PurifiedClassifier.forward` gọi `reset_drift_counter()`, hàm này khởi tạo
lại `embed_net` và optimizer của nó.

Lớp gắn kết: `build_contrastive_guidance` trong
[src/guidance/build.py](../src/guidance/build.py) tạo lớp bọc; `runner.set_score_model` thay
`runner.model` và model của solver (`runner.dynamics.model`).

### Đối chiếu với bài báo

| Bài báo (Algorithm 1, Eq. 8–10) | Code | Khóa config / flag |
|---|---|---|
| `t*` mức nhiễu forward | `--t` | CLI; 100 / 125 / 75 tùy threat model (Bảng 7) |
| `lambda` guidance strength | `guidance_strength` | `guidance_strength` (`--contrastive_guidance_strength`) |
| `tau` temperature | `temperature` | `temperature` (`--contrastive_temperature`) |
| `tau^-` | `tau_minus` | `tau_minus` (`--contrastive_tau_minus`) |
| `beta` độ "hard" của negative mining | `beta` | `beta` (`--contrastive_beta`) |
| HNM Eq. (10) / InfoNCE Eq. (9) | `Estimator.hard` / `.soft` | `estimator` |
| `F(.)` feature extractor | `embed_net`, một MLP trên logits của classifier | `embed_lr`, `--contrastive_classifier` |
| anchor `x(t)_a`, positive `x(t)_p = x(t+dt)` | lần gọi hiện tại so với `xt_cache` | – |
| negative = phần còn lại của minibatch | mask trong `ContrastiveLoss` | `--adv_batch_size` (là siêu tham số của phương pháp) |
| `m` kích thước minibatch | batch truyền vào `forward` | `--adv_batch_size` |
| – (không có trong bài báo) | cửa sổ áp guidance | `drift_min`, `drift_max` |

Bài báo nêu tên các siêu tham số nhưng không cho giá trị số, nên
[configs/method/contrastive.yaml](../configs/method/contrastive.yaml) lưu giá trị của các script
CIFAR-10 gốc và dòng lệnh có thể ghi đè. Mỗi lần chạy ghi kết quả hợp nhất vào
`resolved_config.json`. Giá trị của từng thí nghiệm được liệt kê trong
[experiments/README.md](../experiments/README.md).

## 4. Classifier và dữ liệu

`get_image_classifier(name)` trong [src/models/classifiers/__init__.py](../src/models/classifiers/__init__.py):

| tên | nguồn |
|---|---|
| `cifar10-wideresnet-28-10` | RobustBench `Standard` (tự tải về `pretrained/robustbench/`) |
| `cifar10-wrn-70-16-dropout` | `cifar10_resnet.wide_resnet_70_16_dropout` + `pretrained/cifar10/wrn-70-16-dropout/weights.pt` |
| `cifar10-resnet-50` | `cifar10_resnet.resnet50` + `pretrained/cifar10/resnet-50/weights.pt` |
| `cifar100-wideresnet-28-10` | `pretrained/cifar100/cifar100_wrn_28-10_standard.pt` (từ `scripts/train_classifier.py`) |
| `gtsrb-wrideresnet-28-10`, `gtsrb-resnet18` | `pretrained/gtsrb/…` (từ `scripts/train_classifier.py`) |

Dữ liệu được mô tả một lần duy nhất trong [src/data/specs.py](../src/data/specs.py): mỗi dataset có
một `DatasetSpec` giữ số lớp, kích thước ảnh, shape ảnh, thư mục dữ liệu và file config. Nhà máy
classifier, plug-in contrastive (`img_shape`) và các bộ nạp dữ liệu đều đọc từ đó, thay vì lặp lại
các dữ kiện này.

`load_eval_subset(dataset, num_sub, data_seed)` trong
[src/data/loaders.py](../src/data/loaders.py) trả về các ảnh được đánh giá dưới dạng một tensor
trong `[0, 1]`, kèm theo chỉ số của chúng:

* subset là `np.random.RandomState(data_seed).choice(len(test_split), num_sub, replace=False)`,
  được rút ở một chỗ duy nhất (`select_subset_indices`) và không đổi so với code gốc;
* các chỉ số được ghi vào `eval_indices.json` trong thư mục của lần chạy, nên một lần chạy hoàn
  tất có lưu đúng những ảnh nào đã được chấm điểm;
* tiền xử lý là `Resize((32, 32)) + ToTensor()` cho mọi dataset, tương đương về mặt số học với ba
  pipeline khác nhau mà bản gốc dùng (xem docstring của module và `tests/test_data.py`).

Thư mục dữ liệu: `data/cifar10`, `data/cifar100`, `data/gtsrb`.

Config được nạp và kiểm tra bởi [src/config.py](../src/config.py); xem
[configs/README.md](../configs/README.md).

## 5. Các attack

| `--attack_version` | hiện thực | file |
|---|---|---|
| `rand` | AutoAttack apgd-ce + apgd-dlr kèm EOT (`--eot_iter`) | `external/auto-attack` |
| `standard` | AutoAttack apgd-ce, apgd-t, fab-t, square | như trên |
| `eotpgd` | `torchattacks.EOTPGD(eps, alpha=2/255, steps=10)` | `eval_eotpgd` |
| `apgd` | `torchattacks.APGD(steps=10)` | `eval_apgd` |
| `stadv` | attack biến đổi không gian | `src/attacks/stadv_eot` |
| BPDA+EOT | `scripts.eval_bpda` | `src/attacks/bpda_eot` |

## 6. Sơ đồ code

```
src/
├── paths.py                 thư mục gốc của data / pretrained / outputs / configs (đổi được qua biến môi trường)
├── config.py                schema config score-SDE + loader strict (dùng chung cho eval và training)
├── method.py                siêu tham số contrastive: configs/method/ + hợp nhất với CLI
├── utils.py                 setup_logging, StdoutTee (stdout/stderr → log.txt), str2bool
├── guidance/                contrastive.py (★ phương pháp), build.py (bọc score network)
├── models/diffusion/        base.py (phần dùng chung), sde.py, ode.py (bộ thanh lọc DiffPure trên score_sde)
├── models/classifiers/      get_image_classifier, cifar10_resnet.py
├── attacks/                 attack thượng nguồn: bpda_eot/, stadv_eot/
├── data/                    specs.py (registry dataset), loaders.py (subset đánh giá), gtsrb.py
├── evaluation/              purified_model.py, attacks.py, metrics.py, robustness.py
├── training/                score_sde_module.py, classifier_module.py
external/                    package thượng nguồn cài bằng pip: auto-attack/ (autoattack), score_sde/
```

Chiều import: `scripts → src.evaluation → {src.models, src.attacks, src.guidance, src.data} → thư viện ngoài`.
Không có gì trong `src/` import từ `scripts/` hay `experiments/`.

## 7. Các vấn đề đã biết trong code nghiên cứu

Còn mở:

| ID | Vấn đề | Ảnh hưởng |
|---|---|---|
| **Threat model** | Cả hai nhánh đánh giá đều che gradient của purifier khỏi attacker: `PurifiedClassifier.forward` chạy diffusion trong `torch.no_grad()` rồi truyền `input_x + (x_re - input_x)` vào classifier, nên d(out)/d(input_x) là identity; nhánh BPDA thì `.detach()` tường minh. Code của chính DiffPure không có `no_grad` và truyền gradient ngược qua reverse SDE bằng adjoint solver. | Những gì repo này đo ứng với **Bảng 7** của bài báo ("without accesses to the gradients of diffusion-based purifiers"), không phải cấu hình gradient-on mà các bảng chính mô tả. Cần xác nhận với tác giả trước khi so sánh với các bảng chính |
| A5 (rò rỉ trạng thái) | `reset_drift_counter()` không xoá `xt_cache`, nên bước guidance đầu tiên của một lần thanh lọc dùng trạng thái cuối của lần thanh lọc **trước** (vòng EOT trước, bước attack trước, batch trước) làm positive view | kết quả của một batch phụ thuộc vào lịch sử gọi. Cờ sửa: `--contrastive_fix_cache true` (mặc định tắt để giữ đúng bản phát hành); test: `test_released_reset_keeps_the_cached_view_and_the_fix_drops_it` |
| A6 (guidance bị che) | `acquisition_grad` gọi `.detach()` trên `x`, nên kể cả khi bỏ straight-through, attacker vẫn không thấy đường gradient qua guidance | đánh giá full-gradient của ContrastDiffPure cần `--diffusion_type particle --contrastive_differentiable true`; adjoint solver không dùng được với wrapper có trạng thái (`--grad_mode full` + contrastive bị chặn) |
| A4 (phụ thuộc batch) | negative của loss là các ảnh khác trong batch, và `pos` là cả ma trận B×B | kết quả thanh lọc của một ảnh thay đổi theo các ảnh còn lại trong batch (`test_contrastive_guidance_couples_the_batch`); đây là động lực của PCP, xem [docs/pcp.md](pcp.md) |
| L3 | Hai script ghi đè `drift_min` thành 40 (`contrastive/run_cifar_bpda_eot.sh`, `ablation_t/contrastive_run_cifar_rand_inf.sh`) trong khi config chuẩn ghi 0 | các script là hồ sơ thí nghiệm nên được giữ nguyên như đã viết; cần chốt trước khi báo cáo số |
| L8 | `get_accuracy` bỏ batch cuối nếu không đủ | giữ `num_sub` là bội số của `adv_batch_size` |
| – | `--contrastive_estimator soft` lỗi (sai shape trong `ContrastiveLoss`, thiếu `keepdim=True`) | chỉ `hard` chạy được |

Đã giải quyết trong lúc chuẩn hóa data, config và phương pháp:

| ID | Vấn đề | Cách giải quyết |
|---|---|---|
| B2 | Plug-in contrastive có bản đồ `domain -> (3, H, W)` riêng và từ chối `gtsrb` | `img_shape` chưa bao giờ được phương pháp đọc, đã bỏ khỏi signature |
| B1 | Plug-in BPDA contrastive thiếu hai tham số bắt buộc `batch_size` / `img_shape`, nên mọi lần chạy BPDA contrastive đều báo `TypeError` | cả hai tham số đều không được dùng và đã bị bỏ; một builder dùng chung (`src/guidance/build.py`) giờ phục vụ cả hai nhánh đánh giá, nên `experiments/contrastive/run_cifar_bpda_eot.sh` chạy được |
| L2 | Default của contrastive khác nhau giữa hai entry point (tau 1 so với 10, lambda 500 so với 5000) | một method config chuẩn duy nhất, lấy từ script CIFAR-10 (tau=1, lambda=500). **Script BPDA contrastive do đó thay đổi**, nhưng nó chưa từng chạy được nên không con số đã công bố nào bị ảnh hưởng |
| – | Nhánh GTSRB của `load_data` truyền `batch_size=adv_batch_size`, trả về `adv_batch_size` ảnh thay vì `num_sub` (BPDA: 10 thay vì 200) | mọi dataset giờ đều trả về `num_sub` ảnh. **Số liệu GTSRB không so sánh được với code gốc**; CIFAR-10 không bị ảnh hưởng |
| – | `training.likelihood_weighting` thiếu trong mọi config và lấy từ default của dataclass | đã ghi tường minh trong cả ba config (cùng giá trị, `False`) |
| – | `delta_grad_mul` bị comment là "not used anymore" dù vẫn được dùng | đã đổi tên thành `guidance_strength` (đúng lambda của bài báo) và sửa comment |
