# Hướng dẫn đọc code

Lộ trình đọc xuyên repo và mô tả từng hàm quan trọng làm gì. Viết cho người sẽ **sửa** code này,
không chỉ chạy nó.

Toàn bộ code đang hoạt động khoảng 1950 dòng, nên đọc hết là hoàn toàn khả thi. Hãy đọc theo
**một lần chạy cụ thể**, đừng đọc lần lượt theo thư mục.

Tài liệu đi kèm: [research-architecture.md](research-architecture.md) (phương pháp là gì),
[configs/README.md](../configs/README.md) (những gì cấu hình được),
[reproduction.md](reproduction.md) (cách chạy).

---

## 1. Lộ trình đọc

### Giai đoạn 0 — định hướng (15 phút)

1. [research-architecture.md](research-architecture.md) §3 và §7 — phương pháp, đối chiếu với ký
   hiệu trong bài báo, và các vấn đề đã biết.
2. [configs/README.md](../configs/README.md) — hai lớp config.

### Giai đoạn 1 — đi theo một lần chạy (~400 dòng, 1 giờ)

| # | File | Đọc gì | Cần nắm điều gì |
|---|---|---|---|
| 1 | [experiments/contrastive/run_cifar_rand_inf.sh](../experiments/contrastive/run_cifar_rand_inf.sh) | cả 30 dòng | một thí nghiệm là gì |
| 2 | [scripts/eval_robustness.py](../scripts/eval_robustness.py) | cả 17 dòng | điểm vào được làm mỏng có chủ ý |
| 3 | [src/evaluation/robustness.py](../src/evaluation/robustness.py) | `setup`, rồi `robustness_eval` | config → dữ liệu → model → attack |
| 4 | [src/data/loaders.py](../src/data/loaders.py) | `load_eval_subset` | 64 ảnh nào, và vì sao tái lập được |
| 5 | [src/evaluation/purified_model.py](../src/evaluation/purified_model.py) | `PurifiedClassifier.__init__`, rồi `forward` | phương pháp được gắn vào đâu, và đường gradient |
| 6 | [src/models/diffusion/sde.py](../src/models/diffusion/sde.py) | `image_editing_sample`, rồi `RevVPSDE.f` / `g` | thêm nhiễu tới `t*`, rồi giải reverse SDE |
| 7 | [src/evaluation/attacks.py](../src/evaluation/attacks.py) | chỉ `eval_autoattack` | mỗi attack chạy hai lần: classifier trần, rồi purifier + classifier |

Các attack khác có cùng khuôn với `eval_autoattack`.

### Giai đoạn 2 — phương pháp (275 dòng, 1–2 giờ)

Đọc [src/guidance/contrastive.py](../src/guidance/contrastive.py) từ ngoài vào trong:

1. docstring của module — đối chiếu với Algorithm 1 và Eq. (8);
2. `ContrastiveGuidedDiffusion.forward` — chính là Eq. (8);
3. `acquisition_grad` — 20 dòng, trái tim của phương pháp;
4. `ContrastiveLoss.forward` — Eq. (9) và (10);
5. `MLP` / `MLPBlock` — feature extractor F(·).

Rồi đọc [src/guidance/build.py](../src/guidance/build.py) và [src/method.py](../src/method.py) để
thấy siêu tham số đi tới đó bằng cách nào.

Hãy mở bài báo song song: Algorithm 1 và Eq. (8) ở trang 5, Eq. (9)–(10) ở trang 6.

### Giai đoạn 3 — phần còn lại, khi cần

Classifier, các schema config, phần huấn luyện, các thư viện attack đi kèm.

---

## 2. Tra cứu hàm

Chữ ký hàm được viết rút gọn. "→" là giá trị trả về.

### 2.1 Điểm vào và driver

**[scripts/eval_robustness.py](../scripts/eval_robustness.py)**

* `parse_args_and_config()` → `(args, config)`. Dựng parser rồi gọi `setup`. Cả file tồn tại chỉ
  để tách CLI khỏi phần thư viện.

**[src/evaluation/robustness.py](../src/evaluation/robustness.py)**

* `build_robustness_parser()` → `ArgumentParser`. Toàn bộ flag cho nhánh
  AutoAttack / StAdv / EOT-PGD / APGD. Các flag contrastive có default là `None` để flag không
  truyền sẽ lấy giá trị từ method config. Tên flag là `--contrastive_<trường>`, sinh tự động từ
  `ContrastiveConfig`.
* `build_bpda_parser()` → tương tự cho BPDA+EOT, thêm `--eot_defense_reps` /
  `--eot_attack_reps`.
* `setup(args)` → `(args, config)`. Nạp và kiểm tra config diffusion, resolve method config vào
  `args.method_config` (chỉ khi `--guidance contrastive`), cấu hình logging, đặt `--exp` tương đối
  vào dưới `outputs/`, tạo thư mục output, và gieo seed cho `torch`, `random`, `numpy`. Hàm này
  cũng đặt `torch.backends.cudnn.benchmark = True` — đây là lý do kết quả không tái lập được từng
  bit giữa các lần chạy.
* `record_run_config(args, log_dir)`. Ghi `resolved_config.json`: dòng lệnh, tên các file config,
  và siêu tham số phương pháp sau khi hợp nhất file với các override. Đây là bản ghi cho biết một
  con số được báo cáo đã sinh ra từ cấu hình nào.
* `robustness_eval(args, config)`. Driver chính. Dựng đường dẫn output
  `<exp>/<i>/<classifier>/<diffusion>_<attack>/seed<S>/data<D>`, chuyển stdout vào `log.txt` qua
  `utils.StdoutTee`, nạp dữ liệu và model, rồi dispatch theo `--attack_version`.
  **Thứ tự "nạp dữ liệu" và "dựng model" khác nhau giữa `--guidance none` và `contrastive`**, và
  được giữ nguyên có chủ ý: việc lặp qua một `DataLoader` tiêu thụ RNG toàn cục của torch, nên đổi
  thứ tự sẽ đổi nhiễu khuếch tán với cùng một `--seed`.
* `bpda_robustness_eval(args, config)`. Tương tự, cho BPDA+EOT; luôn dựng model trước.

### 2.2 Dữ liệu

**[src/data/specs.py](../src/data/specs.py)** — nguồn duy nhất cho mọi dữ kiện về dataset.

* `DatasetSpec` — một dataclass frozen: `name`, `num_classes`, `image_size`, `num_channels`,
  `config_file`, `root_subdir`.
  * `.root` → thư mục dữ liệu nằm dưới `data/`.
  * `.img_shape` → `(C, H, W)`; đây là thứ mà plug-in contrastive trước kia khai báo trùng lặp.
  * `.eval_transform()` → `Resize((S, S)) + ToTensor()`, cho tensor float trong `[0, 1]`. Một
    pipeline duy nhất cho mọi dataset; nó giống hệt từng bit với ba pipeline khác nhau mà code gốc
    dùng (`tests/test_data.py` liên tục kiểm tra điều đó).
  * `.build_split(train, transform=None, root=None)` → toàn bộ tập train hoặc test (torchvision
    CIFAR-10/100, hoặc `GTSRBDatasetTrain`/`GTSRBDatasetTest` với `enable_crop=False`). Dùng chung
    cho đánh giá (`train=False`) và hai script huấn luyện.
* `get_dataset_spec(name)` → spec, tra theo tên chính xác. Không so khớp chuỗi con, nên `cifar100`
  không còn bị `cifar10` che mất.

**[src/data/loaders.py](../src/data/loaders.py)**

* `select_subset_indices(n_total, num_sub, data_seed)` → mảng chỉ số.
  `RandomState(data_seed).choice(n_total, num_sub, replace=False)`, không đổi so với bản gốc. Chỉ
  phụ thuộc ba tham số của nó, nên tái lập được giữa các máy và không phụ thuộc RNG toàn cục.
  **Thứ tự không được sắp xếp** — nó quyết định ảnh nào vào batch attack nào.
* `load_eval_subset(dataset, num_sub, data_seed, log_dir=None)` → `(x, y, indices)`.
  Dựng tập test, rút subset, và trả về dưới dạng **một tensor duy nhất**, vì các attack nhận cả
  tensor chứ không nhận loader. `x` nằm trong `[0, 1]`, shape `(num_sub, C, H, W)`, có bật
  `requires_grad_(True)` cho attack. Ghi `eval_indices.json` khi có `log_dir`.
  `num_workers=4` được giữ đúng giá trị gốc: với workers > 0, một `DataLoader` rút một base seed từ
  RNG toàn cục của torch, nên đổi số này sẽ làm lệch chuỗi số ngẫu nhiên.
* `load_data(args, log_dir=None)` → `(x, y)`. Như trên, nhưng lấy tham số từ dòng lệnh đã parse.

### 2.3 Lắp ghép model

**[src/evaluation/purified_model.py](../src/evaluation/purified_model.py)**

* `PurifiedClassifier.__init__(args, config)`. Dựng classifier, rồi dựng purifier
  (`SDEPurifier` cho `sde`, `ODEPurifier` cho `ode`). Với `--guidance contrastive`,
  nó bọc score network bằng `runner.set_score_model(...)`, hàm này thay cả `runner.model` lẫn
  tham chiếu riêng của solver (`runner.dynamics.model`).
* `PurifiedClassifier.forward(x)` (mode mặc định `purify_and_classify`) → logits. `x` trong `[0, 1]`.
  1. toàn bộ quá trình thanh lọc chạy trong `torch.no_grad()`;
  2. `x_re = runner.image_editing_sample((x - 0.5) * 2, ...)` — đưa về `[-1, 1]` rồi thanh lọc;
  3. `_w = x_re - input_x`, một **hằng số**;
  4. `out = classifier(((input_x + _w) + 1) * 0.5)`.
  Về mặt toán học `input_x + _w == x_re`, nhưng đạo hàm theo `input_x` là identity: attacker thấy
  một straight-through estimator và không bao giờ thấy gradient thật của purifier. Xem bẫy số 1.
  Cuối cùng hàm gọi `reset_drift_counter()` trên lớp bọc, nên mỗi batch đã thanh lọc bắt đầu với
  một F(·) mới.
* `reset_counter()`, `set_tag(tag)`. `counter` điều khiển việc log "mỗi 5 NFE" và tên thư mục nơi
  ảnh mẫu được ghi ra; `tag` dùng để gắn nhãn các thư mục đó (`no_adv`, `sde_adv`).

Cũng trong file này, cho nhánh BPDA:

* `UndefendedClassifier` — classifier với `purify` là hàm đồng nhất, tức là baseline không phòng thủ
  trong thí nghiệm BPDA.
* `PurifiedClassifier.purify(x)` → `x` đã thanh lọc, trong `[0, 1]`; `forward(x, mode=...)` chọn
  `purify`, `classify` hoặc cả hai — đây là giao diện mà `BPDA_EOT_Attack` cần.

### 2.4 Bộ thanh lọc

**[src/models/diffusion/base.py](../src/models/diffusion/base.py)** — phần dùng chung cho SDE và ODE

* `restore_checkpoint(ckpt_dir, state, device)`. Nạp dict checkpoint của score_sde
  (`optimizer`, `model`, `ema`, `step`); phần model được nạp với `strict=False`.
* `load_score_model(config, device)`. Chọn checkpoint theo `config.data.dataset`, dựng NCSN++
  bằng `mutils.create_model(config)`, và với CIFAR-10 thì **copy trọng số EMA vào model**
  (`ema.copy_to`) — trọng số EMA mới là trọng số được dùng.
* `VPDynamics` — lịch β của VP-SDE và `score(t, x)`; `RevVPSDE` (sde.py) và `VPODE` (ode.py)
  kế thừa nó.
* `DiffusionPurifier.__init__(args, config, device)`. Nạp score model, dựng `dynamics`
  (`RevVPSDE` hoặc `VPODE`) và giữ `betas` cho bước thêm nhiễu xuôi. `SDEPurifier` (sde.py) và
  `ODEPurifier` (ode.py) chỉ cài đặt bước tích phân `_integrate`.
* `DiffusionPurifier.image_editing_sample(img, bs_id=0, tag=None)` → batch đã thanh lọc.
  `img` nằm trong `[-1, 1]`.
  1. thêm nhiễu xuôi trong thời gian **rời rạc**: `x = x0·√ᾱ[t-1] + ε·√(1-ᾱ[t-1])` với `t = --t`
     (`--rand_t` làm `t` dao động `±t_delta`);
  2. tích phân reverse SDE trong thời gian **liên tục** từ `t0 = 1 - t/1000` đến `t1 = 1 - 1e-5`
     bằng `torchsde.sdeint_adjoint(..., method='euler')`; solver tự chia nhỏ khoảng đó, nên số lần
     gọi drift lớn hơn 2 rất nhiều;
  3. lặp `--sample_step` lần, mỗi lần thêm nhiễu lại kết quả đã thanh lọc, rồi nối theo chiều
     batch. Mọi script đều dùng `sample_step=1`; giá trị lớn hơn sẽ trả về nhiều ảnh hơn số nhận
     vào, điều mà lệnh gọi classifier phía sau không lường tới.
  Với `bs_id < 2`, hàm còn ghi `original_input.png`, `init_*.png` và `samples_*.png/.pth`.
* `RevVPSDE.vpsde_fn(t, x)` → `(drift, diffusion)` của VP-SDE **xuôi**:
  `β(t) = β₀ + t(β₁-β₀)`, drift `-½β(t)x`, diffusion `√β(t)`.
* `RevVPSDE.rvpsde_fn(t, x, return_type)` → drift hoặc diffusion của SDE **ngược**.
  Với drift, hàm reshape trạng thái phẳng về dạng ảnh, lấy score bằng
  `mutils.get_score_fn(..., continuous=True)` (hàm này chuyển dự đoán ε thành score), rồi trả về
  `drift - diffusion²·score`.
* `RevVPSDE.f(t, x)`, `g(t, x)` — giao diện mà `torchsde` yêu cầu, dưới phép đổi biến `t' = 1 - t`:
  `f` tính drift ngược tại `1-t` rồi lấy dấu âm, `g` broadcast diffusion vô hướng ra shape của
  trạng thái. Trạng thái ở dạng phẳng `(batch, C·H·W)` vì `sdeint` chỉ nhận tensor 2 chiều.

[src/models/diffusion/ode.py](../src/models/diffusion/ode.py) giống như trên, với
`VPODE.ode_fn` trả về `drift - ½·diffusion²·score` (probability-flow ODE) và
`torchdiffeq.odeint_adjoint`, trong đó `--step_size` là bước Euler.

### 2.5 Phương pháp

**[src/guidance/contrastive.py](../src/guidance/contrastive.py)**

* `MLPBlock(input_size, hidden_size)` — một block gated residual: `d3(silu(d1(x)) * d2(x))`, không
  bias. `MLP` xếp ba block như vậy với kết nối residual và cung cấp `reset_parameters()`. Đây chính
  là F(·), và nó nhận **logits của classifier**, không nhận ảnh: nó được dựng bằng
  `MLP(n_classes, 2*n_classes)`.
* `ContrastiveLoss.negative_mask(batch_size)` → mask boolean `(2m, 2m)`, loại bỏ với mỗi hàng trong
  `2m` hàng: chính nó và bản sao của nó ở view còn lại.
* `ContrastiveLoss.forward(x_1, x_2)` → loss vô hướng. `x_1`, `x_2` là embedding `(m, d)` của hai
  bước reverse liên tiếp.
  * `out = cat([x_1, x_2])`; `neg = exp(cos_sim(out, out)/τ)` được mask xuống `(2m, 2m-2)`;
  * `pos = exp(cos_sim(x_1, x_2)/τ)` — lưu ý đây là **ma trận `(m, m)` đầy đủ**, rồi xếp thành
    `(2m, m)`; vì vậy loss lấy trung bình trên *mọi* cặp, không chỉ các cặp khớp nhau — đây là
    điểm khác so với InfoNCE trong sách;
  * `estimator='hard'` (Eq. 10): `imp = neg^β`, `reweight_neg = Σ(imp·neg)/mean(imp)`,
    `Ng = (-τ⁻·N·pos + reweight_neg)/(1-τ⁻)` với `N = max(2m-2, 2)`, chặn dưới bởi `N·e^(-1/τ)`;
  * `estimator='soft'` (Eq. 9): `Ng = neg.sum(dim=-1)`, câu này **thiếu `keepdim=True`** nên không
    broadcast được với `pos` `(2m, m)` — đó chính là toàn bộ nguyên nhân của lỗi "soft bị hỏng";
  * `loss = -log(pos/(pos+Ng)).mean()`, và báo lỗi nếu gặp NaN.
* `ContrastiveGuidedDiffusion.__init__(model, classifier, n_classes, ...)`. Giữ score network, một
  classifier đóng băng ở `eval()`, một `MLP` mới, optimizer Adam của nó, hàm loss, cửa sổ guidance
  và `guidance_strength` (λ). `xt_cache` khởi đầu là `None`.
* `ContrastiveGuidedDiffusion.acquisition_grad(x, x0)` → `λ·∂loss/∂x`. Trái tim của phương pháp:
  1. `x0` là **bước reverse trước đó** (không phải ảnh sạch, dù tên gợi ý như vậy); logits của nó
     được tính trong `no_grad`;
  2. `delta = zeros_like(x, requires_grad=True)`, và loss được tính tại `classifier(x + delta)`.
     Vì `delta` bằng 0, giá trị không thay đổi gì, nhưng `delta.grad` cuối cùng chứa `∂loss/∂x` —
     đây là mẹo để lấy gradient theo trạng thái hiện tại;
  3. `loss = contrastive_loss(embed(logits(x0)), embed(logits(x + delta)))`;
  4. `zero_grad(); backward(); step()` — **một bước Adam trên F(·) mỗi lần gọi**, nên feature
     extractor được huấn luyện trực tuyến, đúng như bài báo nói "F(·) requires training on each
     minibatch";
  5. trả về `delta.grad · guidance_strength`.
  Lưu ý: tham số của classifier không bị đóng băng bằng `requires_grad_(False)`, nên buffer `.grad`
  của chúng tích lũy qua các lần gọi. Không có gì đọc chúng, nhưng chúng tốn bộ nhớ.
* `ContrastiveGuidedDiffusion.forward(x, t)` → đầu ra của score network đã cộng
  guidance. Đây chính là Eq. (8).
  1. `out = model(x, t)` — luôn được tính;
  2. nếu shape của batch thay đổi, cache bị bỏ và counter được reset — lần gọi hiện tại khi đó lại
     được tính là lần đầu, nên nó cũng không có guidance;
  3. không có guidance khi `xt_cache is None` (lần gọi đầu) hoặc khi ở ngoài khoảng
     `[drift_min, drift_max)`;
  4. ngược lại `out[:, :C] += acquisition_grad(x, xt_cache)` — chỉ C kênh đầu, để xử lý trường hợp
     đầu ra có learned sigma `(mean, var)`; với NCSN++ thì phép cắt này bao trọn đầu ra;
  5. lưu `x` vào cache, tăng `drift_counter`.
  Quy ước dấu khớp đúng với Eq. (8) (cộng vào dự đoán ε); việc chuyển từ ε sang score, bao gồm hệ
  số âm của nó, diễn ra sau đó trong `RevVPSDE.rvpsde_fn`.
* `ContrastiveGuidedDiffusion.reset_drift_counter()`. Đưa counter về 0, **khởi tạo lại F(·)** và
  dựng lại optimizer. Được gọi sau mỗi batch đã thanh lọc, nên không có thông tin rò rỉ giữa các
  batch.

**[src/guidance/build.py](../src/guidance/build.py)**

* `build_contrastive_guidance(config, model, device)` → score network đã được bọc. Một builder duy
  nhất cho cả hai nhánh đánh giá; hai bản sao mà nó thay thế đã lệch nhau, và đó chính là nguyên
  nhân khiến mọi lần chạy BPDA contrastive bị crash.

### 2.6 Attack và chỉ số

**[src/evaluation/attacks.py](../src/evaluation/attacks.py)** — mọi hàm đều cùng một khuôn: tấn
công classifier trần, rồi tấn công purifier + classifier, in ra độ chính xác và lưu `x_adv_*.pt`.

* `eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir)`.
  `--attack_version rand` chạy `apgd-ce` + `apgd-dlr` với `eot_iter` lần lặp (đây là phiên bản mọi
  script dùng); `standard` chạy `apgd-ce`, `apgd-t`, `fab-t`, `square`. Hàm ghi `log_resnet.txt` và
  `log_sde_adv.txt`, và truyền `state_path=` để một lần chạy có thể tiếp tục khi bị ngắt.
* `eval_eotpgd` / `eval_apgd` — `torchattacks.EOTPGD(eps, alpha=2/255, steps=10, eot_iter)` và
  `torchattacks.APGD(steps=10, eot_iter)`. Cả hai gọi `model_.set_tag('no_adv')` trước lượt chạy
  sạch và `'sde_adv'` sau khi attack; tag này chỉ dùng để đặt tên thư mục ảnh được ghi ra.
* `eval_stadv` — attack biến đổi không gian, `bound=--adv_eps`, 100 vòng lặp.
* `eval_bpda` — `BPDA_EOT_Attack(model, adv_eps, eot_defense_reps, eot_attack_reps)`; báo cáo hàng
  đầu và hàng cuối của `class_batch` làm độ chính xác ban đầu và độ chính xác robust.

**[src/evaluation/metrics.py](../src/evaluation/metrics.py)**

* `get_accuracy(model, x, y, bs, device)` → độ chính xác. Hàm lặp `n_batches = N // bs`, nên phần
  dư bị bỏ âm thầm; hãy giữ `num_sub` là bội số của `adv_batch_size`.

### 2.7 Config và classifier

**[src/config.py](../src/config.py)**

* `load_config(name_or_path)` → `ConfigMain`. Strict: mọi khóa mà code đọc đều bắt buộc, và các
  khóa thiếu, không hợp lệ hoặc sai chính tả đều được báo cùng lúc, kèm tên file và tên mục.
  `ConfigMain.device` do `setup` điền vào, không phải do YAML.

**[src/method.py](../src/method.py)**

* `load_method_config(name)` → `ContrastiveConfig`, từ `configs/method/`. Phần kiểm tra nằm trong
  `__post_init__` (tên estimator, `drift_max > drift_min`, temperature dương).
* `resolve_contrastive(args)` → `ContrastiveConfig` có hiệu lực: lấy từ file, rồi áp các override
  từ dòng lệnh, rồi lấy `n_classes` từ dataset spec nếu chưa được truyền. Báo lỗi nếu chưa chọn
  classifier.

**[src/models/classifiers/__init__.py](../src/models/classifiers/__init__.py)**

* `get_image_classifier(name)` → một `nn.Module` nhận ảnh trong `[0, 1]`. Hàm dispatch theo chuỗi
  con của `name` (`cifar100` được kiểm tra trước `cifar10`, nên thứ tự có ý nghĩa). Model
  RobustBench được tải về `pretrained/robustbench`; các model khác nạp từ `.pt` cục bộ.
  `ImageNetNormalized` thêm chuẩn hóa theo ImageNet — xem bẫy số 6.
* `update_state_dict(state_dict, idx_start)` — bỏ tiền tố kiểu `module.0.` do checkpoint lưu bằng
  `DataParallel` để lại.

---

## 3. Bảy chỗ dễ hiểu sai

1. **Attacker không bao giờ nhận được gradient của purifier.** Cả hai nhánh đều straight-through:
   `PurifiedClassifier.forward` bọc quá trình diffusion trong `torch.no_grad()` rồi đưa
   `input_x + (x_re - input_x)` vào classifier, còn `BPDA_EOT_Attack.purify_and_predict` gọi
   `.detach()` tường minh. Code của chính DiffPure không có `no_grad` và truyền gradient ngược qua
   reverse SDE bằng adjoint solver. Nên những gì repo này đo ứng với **Bảng 7** của bài báo
   ("without accesses to the gradients of diffusion-based purifiers"), không phải cấu hình
   gradient-on mà các bảng chính mô tả. Hãy xác nhận với tác giả trước khi so sánh với các bảng
   chính. Một hệ quả: khả năng cho gradient chính xác của adjoint solver chưa bao giờ được dùng ở
   đây, nên các side effect của guidance chỉ chạy trong lượt xuôi.
2. **`drift_counter` đếm số lần solver gọi drift, không phải số bước khuếch tán.** Với `t=100` và
   solver Euler, có rất nhiều lần gọi cho mỗi batch được thanh lọc, nên `drift_min=0` và
   `drift_min=40` là hai cấu hình khác nhau đáng kể. Tham số này hoàn toàn không có trong bài báo.
3. **Lần gọi đầu tiên không bao giờ có guidance**, vì `xt_cache` còn là `None`.
4. **Positive pair là bước reverse trước đó**, đọc từ `xt_cache` — không phải một phép augmentation
   của cùng một ảnh như trong contrastive learning thông thường.
5. **Negative là các sample khác trong cùng minibatch**, nên `--adv_batch_size` là siêu tham số của
   phương pháp, không chỉ là thiết lập hạ tầng.
6. **CIFAR-100 và GTSRB bị chuẩn hóa hai lần** — `DMWideResNet` mang mean/std riêng của nó, rồi
   lại bị bọc thêm chuẩn hóa ImageNet. Lúc huấn luyện cũng làm y như vậy, nên hai bên khớp nhau và
   các checkpoint đã phát hành phụ thuộc vào điều đó. Đừng "sửa" nó.
7. **`estimator: soft` báo lỗi shape.** Chỉ `hard` chạy được; nguyên nhân là thiếu `keepdim=True`
   như mô tả ở §2.5.

### Tự kiểm chứng bẫy số 1

Khẳng định rằng Jacobian của purifier bị bỏ đi là điều nên tự thấy thay vì tin suông. Với một
purifier giả có đạo hàm đã biết, cách viết của repo và cách lấy đạo hàm qua purifier cho hai
gradient khác nhau:

```python
import torch
purifier  = lambda x: 2 * x          # d/dx = 2
classifier = lambda z: (z ** 2).sum()
x0 = torch.tensor([0.3, 0.7])

x = x0.clone().requires_grad_(True)          # cách viết của repo
with torch.no_grad():
    x_re = purifier((x - 0.5) * 2)
    _w = x_re - x
classifier(((x + _w) + 1) * 0.5).backward()
print(x.grad)                                 # tensor([0.1000, 0.9000])

x = x0.clone().requires_grad_(True)          # lấy đạo hàm qua purifier
classifier((purifier((x - 0.5) * 2) + 1) * 0.5).backward()
print(x.grad)                                 # tensor([0.4000, 3.6000])  -- lớn gấp 4
```

---

## 4. Đọc một cách chủ động

Không cần GPU, checkpoint hay dataset:

```bash
python -m pytest tests/test_contrastive.py -v   # phương pháp trên mạng toy
python -m pytest tests/test_data.py -v          # subset và tiền xử lý
python -m pytest tests/test_method_config.py -v # siêu tham số và cơ chế override
```

`tests/test_contrastive.py` là nơi thí nghiệm rẻ nhất: nó dựng một `ContrastiveGuidedDiffusion`
trên một mạng toy hai lớp, nên bạn có thể thêm `print` vào trong `acquisition_grad` và quan sát
loss, `delta.grad` cùng độ lớn của guidance trên số thật.

Muốn xem các shape bên trong loss thì gọi trực tiếp:

```python
import torch
from src.guidance.contrastive import ContrastiveLoss
loss = ContrastiveLoss(temperature=1.0, estimator="hard")
print(loss(torch.randn(8, 10), torch.randn(8, 10)))   # m=8 embedding, chiều rộng 10
print(loss.negative_mask(8).shape, int(loss.negative_mask(8)[0].sum()))  # (16, 16) 14
```

Mọi khẳng định về hành vi trong §2.5 và §3 đều đã được kiểm chứng bằng cách chạy thật: `pos` đúng
là `(m, m)` và `neg` sau khi mask là `(2m, 2m-2)`; lần gọi đầu đúng là trả về đầu ra model không bị
thay đổi; F(·) đúng là thay đổi ở mọi lần gọi; buffer `.grad` của classifier đúng là tích lũy; việc
đổi shape batch đúng là reset counter; và estimator `soft` báo lỗi
"size of tensor a (8) must match the size of tensor b (16)" với `m = 8`, đúng như dự đoán từ việc
thiếu `keepdim=True`.

---

## 5. Đi tiếp từ đâu

* Sửa phương pháp, thêm attack hoặc dataset: [development.md](development.md).
* Chạy các thí nghiệm đã công bố: [reproduction.md](reproduction.md) và
  [experiments/README.md](../experiments/README.md).
* Các câu hỏi còn mở trước khi báo cáo số: [research-architecture.md](research-architecture.md) §7.
