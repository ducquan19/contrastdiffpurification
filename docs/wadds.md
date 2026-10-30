# WA-DDS — Wiener-Anchored Diffusion Denoised Smoothing

Tài liệu này là điểm vào duy nhất của hướng WA-DDS. Đây là hướng certified ℓ2 và không huấn luyện gì: dùng nguyên diffusion model score_sde CIFAR-10 và classifier WRN-28-10 Standard.
Code PCP, SRSG và contrastive cũ được giữ nguyên, nhưng **không** nằm trên đường chạy của hướng này.

## 1. Phương pháp

Base classifier của randomized smoothing nhận **một** ảnh nhiễu $\tilde x = x + \sigma\xi$ (thang [0, 1]):

1. Đặt $\sigma_d = 2\sigma$ và chọn $t_\sigma$ sao cho $\bar\alpha(t_\sigma) = 1/(1+\sigma_d^2)$ (lời giải đóng: `sigma_to_time`). Khi đó $x_t = \sqrt{\bar\alpha}\,(2\tilde x - 1)$, đúng như Carlini et al.
2. Posterior mean một bước: $m = (x_t - \sqrt{1-\bar\alpha}\,\epsilon_\theta(x_t,t_\sigma))/\sqrt{\bar\alpha}$. Đây là baseline **onestep**.
3. Chạy S bước DDIM (tham số $\eta$). Ở mỗi bước, ước lượng Tweedie $\hat x_0$ được thay bằng
   $\tilde x_0 = \mathcal F^{-1}[\gamma W\,\mathcal F(m) + (1-\gamma W)\,\mathcal F(\hat x_0)]$, với $W = P/(P+\sigma_d^2)$, rồi tính lại $\epsilon$ cho nhất quán. Đây là **wadds**.
   - Khi $W\equiv 0$ ta được **multistep** (một lần chạy của DensePure).
   - Khi $W\equiv 1$ ta được onestep. Cả hai giới hạn đều được kiểm tra trong `tests/test_certify.py`.
4. Đầu ra là ước lượng sạch $\tilde x_0$ của bước cuối, kẹp về [0, 1], rồi đưa qua classifier.
5. Chứng nhận bằng Cohen CERTIFY: $n_0$ mẫu để chọn lớp, $n$ mẫu cho cận dưới Clopper–Pearson $\underline{p_A}$, bán kính $R=\sigma\,\Phi^{-1}(\underline{p_A})$.

$P(f)$ là phổ công suất trung bình của **ảnh train** (45.000 ảnh, không gồm val pool), ở thang [−1, 1], FFT `ortho`, tính riêng từng kênh.

**Vì sao chứng nhận hợp lệ.** $m$, $W$ (một thống kê cố định) và nhiễu DDIM chỉ phụ thuộc $\tilde x$ cùng nhiễu nội bộ độc lập với $x$. Vì thế định lý Cohen áp dụng được.

**Số NFE mỗi mẫu:** onestep tốn 1; multistep và wadds tốn S (bước đầu dùng lại lượt tính đã cho ra $m$).

## 2. File

| File | Vai trò |
|---|---|
| `src/certify/smoothing.py` | CERTIFY, Clopper–Pearson, bán kính, certified accuracy, ACR, generator nhiễu theo từng ảnh |
| `src/certify/denoisers.py` | `DDSDenoiser` (onestep / multistep / wadds), `diagnose()` cho G0 |
| `src/certify/spectrum.py` | $P(f)$, hệ số Wiener, bảng SNR |
| `scripts/spectral_stats.py` | Tính `pretrained/spectral/cifar10_power.pt` (CPU) |
| `scripts/certify_dds.py` | `--mode diagnose` (G0) và `--mode certify` |
| `scripts/compare_dds.py` | So sánh ghép cặp theo ảnh: biến thể với biến thể (G0), run với run (certify); bootstrap CI và McNemar |
| `experiments/wadds/run_cifar10.sh` | Các stage `stats`, `g0`, `dev`, `tune`, `devbase`, `main [method]`, `summary` trên Slurm |
| `tests/test_certify.py` | 30 test CPU |

## 3. Chạy trên cluster

```bash
source /media02/lhbac34/cdp.sh
git pull
python -m pytest tests/test_certify.py -q            # 30 passed
bash experiments/wadds/run_cifar10.sh stats          # login node, khoảng 1 phút
bash experiments/wadds/run_cifar10.sh g0             # laneA, 1 job
bash experiments/wadds/run_cifar10.sh dev            # laneB, 1 job
bash experiments/wadds/run_cifar10.sh tune           # lưới G0 quanh cấu hình mặc định (laneA)
bash experiments/wadds/run_cifar10.sh devbase        # onestep + multistep trên cùng 100 ảnh dev (laneB)
# đọc kết quả bằng compare_dds (mục 4), chọn cấu hình, rồi mới chạy:
WADDS_ARGS="--steps 10 --eta 1.0 --gamma 1.0" bash experiments/wadds/run_cifar10.sh main   # 24 job
bash experiments/wadds/run_cifar10.sh main multistep # baseline cùng giao thức
bash experiments/wadds/run_cifar10.sh summary        # tính lại mọi summary.json
```

Kết quả nằm trong `outputs/wadds/<run>/`, gồm:
- `certify_shard*.tsv` hoặc `diag_shard*.tsv`, ghi từng ảnh và flush ngay;
- `summary.json`;
- `eval_indices.json` và `resolved_config_shard*.json`.

Job bị hủy thì nộp lại **đúng lệnh cũ**: `--resume` đã có sẵn trong lệnh và sẽ bỏ qua các ảnh đã xong.

Tên run được tạo tự động, ví dụ `diag_s0.5_S10_eta1_g1_dev200_ds0` hoặc `wadds_s0.5_S10_eta1_g1_N1000_dev100_ds0`.

## 4. Đọc kết quả

**G0** (`diag_*/summary.json`). Mỗi biến thể có các chỉ số sau:
- `acc_per_draw`: trung bình theo ảnh của tần suất lớp đúng, tức $p_{\text{true}}$;
- `smoothed_acc`: tỉ lệ ảnh có lớp chiếm đa số là lớp đúng;
- `plugin_acr`: ACR tính bằng tần suất thực nghiệm (lạc quan, chỉ dùng để chẩn đoán);
- `d_acc_per_draw_vs_onestep` kèm `ci95`: chênh lệch có ghép cặp so với onestep, CI bootstrap theo ảnh.

Các biến thể:
- `onestep` ($m$) và `multistep` ($s$): hai baseline, có sẵn miễn phí từ cùng các mẫu nhiễu;
- `posthoc`: tần thấp của $m$ ghép với tần cao của $s$;
- `posthoc_rev`: ngược lại;
- `wadds`: phương pháp đề xuất.

`gate_G0.pass` đúng khi `acc_per_draw` của wadds ≥ max(onestep, multistep) **và** `plugin_acr` của wadds ≥ của onestep.

Nếu `posthoc` thắng còn `posthoc_rev` thua, giả thuyết "dao động nhãn đến từ dải tần thấp" được xác nhận. Nếu ngược lại thì xem mục 5.

**So sánh ghép cặp (bắt buộc trước khi kết luận):**
```bash
python -m scripts.compare_dds diag outputs/wadds/diag_s0.5_*/          # biến thể trong từng run + các run tune
python -m scripts.compare_dds certify outputs/wadds/multistep_s0.5_S10_eta1_N1000_dev100_ds0 \
    outputs/wadds/onestep_s0.5_N1000_dev100_ds0 outputs/wadds/wadds_s0.5_S10_eta1_g1_N1000_dev100_ds0
```
Đối thủ chính là **multistep**, không phải onestep. G0 (26/10) cho thấy phần lớn mức tăng so với onestep đến từ việc lấy mẫu nhiều bước.

**G1** (`wadds_*_dev100_*/summary.json`): `certified_accuracy` tại các bán kính 0 đến 2, `acr`, `abstain_rate`.
Chỉ có thể kết luận khi đặt cạnh onestep và multistep cùng N, cùng ảnh (bạn chạy sau):
```bash
gpu_cmd laneB 4 "python -m scripts.certify_dds --mode certify --method onestep --sigma 0.5 --eval_split dev --num_sub 100 --n 1000 --resume"
```
Ngưỡng GO đã đặt: certified accuracy hơn ít nhất 2 điểm ở ít nhất một bán kính (σ = 0.5 hoặc 1.0) và không thua quá 1 điểm ở bán kính nào, **hoặc** ACR tăng ≥ 0.02.

## 5. Nếu G0 không đạt
1. `--gamma 0.5`: neo một phần.
2. `--anchor_t_min 0.5`: chỉ neo ở nửa đầu quỹ đạo.
3. `--power_mode pooled`: một gain chung cho cả ba kênh.
4. `--eta 0`: DDIM deterministic.

Mỗi lần thử G0 trên 200 ảnh chỉ tốn vài phút GPU.

## 6. Chi phí (đo ngày 26/10)

| | Tốc độ | Peak GPU mem |
|---|---|---|
| gpu02 (A100), diagnose | khoảng 860 NFE/s; 2.2 s/ảnh (100 mẫu × 19 NFE) | 1.6 GiB (batch 125) |
| gpu04 (V100), certify N=1000 | khoảng 330 NFE/s; 32.9 s/ảnh | 2.3 GiB (batch 125) |

Thời gian thực tế:
- `g0`: 22 phút trên A100;
- `dev`: 2 giờ 46 phút trên V100;
- `main`: 10.100 × 10 NFE mỗi ảnh. Khoảng 117 s/ảnh trên A100 và khoảng 300 s/ảnh trên V100, nên chia 8 shard khoảng 63 ảnh, `--time 8`, `--batch 500`.

## 7. Thử nghiệm phương pháp thay thế (stage `screen`, 27/10)

Kết quả tune: khi dùng η = 0, phần neo phổ gần như không còn hơn DDIM deterministic. Cấu hình mặc định giờ là η = 0. Stage `screen` thử hai hướng, đều không huấn luyện gì và đều giữ chứng nhận hợp lệ:

- **A. Classifier head** (`src/certify/heads.py`; cờ `--ens_classifiers a,b`, `--tta_flip`): trung bình softmax của nhiều classifier có sẵn, trên ảnh gốc và ảnh lật ngang. Áp lên biến thể `multistep` và `guided`, tạo các biến thể `+flip`, `+ens`, `+ens+flip`.
- **B. Guidance theo tính nhất quán của classifier** (`DDSDenoiser.set_guidance`; các cờ `--guide_*`):
  - chỉ áp ở các bước có $t \le$ `t_frac` $\cdot t_\sigma$;
  - cập nhật $x_0 \leftarrow x_0 + \lambda\, g/\|g\|$, với $g$ là gradient của $\log p_f(\hat y\mid x_0)$;
  - tùy chọn low-pass gradient (`--guide_lowpass`, đơn vị chu kỳ/ảnh) để bỏ phần tần cao mang tính đối kháng;
  - tùy chọn cổng theo độ tự tin (`--guide_conf_min`).

  Thành phần này tạo biến thể `guided`.

Đọc kết quả: `compare_dds diag` tự so mỗi biến thể mới với `multistep`, và với biến thể gốc của nó, ví dụ `guided+ens` so với `guided`.
