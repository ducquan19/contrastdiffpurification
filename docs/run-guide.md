# Hướng dẫn chạy từ đầu đến cuối (PCP và các baseline full-gradient)

Tài liệu này đi theo đúng thứ tự thực hiện: đưa code lên server, dựng môi trường, tải dữ liệu và
checkpoint, kiểm tra, rồi chạy các thí nghiệm theo các pha của kế hoạch (pilot → tune → bảng chính
→ bảng bổ sung → tổng hợp). Mọi lệnh chạy **từ thư mục gốc của repo**. Cụm Slurm là cụm được mô tả
trong [experiments/slurm/README.md](../experiments/slurm/README.md): code, dữ liệu, môi trường và
cache đều nằm dưới `/media/<user>` (hoặc `/media02/<user>`).

```
 0. đưa code lên server
 1. môi trường  ── setup_env.sh (login) ── build_score_sde.sh (GPU)
 2. dữ liệu     ── prepare_assets.sh (login) + tải encoder PCP (login)
 3. kiểm tra    ── pytest ── smoke.sbatch
 4. calibrate   ── đo throughput, ước lượng GPU-giờ
 5. bank        ── build_prototype_bank (GPU, một lần cho mỗi encoder)
 6. pilot       ── 5 job kiểm tra attack  ──► đạt thì đi tiếp
 7. tune        ── lưới siêu tham số trên tập val  ──► go / no-go
 8. bảng chính  ── 5 phương pháp × 3 seed × 4 shard, 512 ảnh test
 9. bổ sung     ── ℓ2, DLR, encoder đơn lẻ, ablation
10. tổng hợp    ── summarize_pgd_full + aggregate_pgd_full
```

---

## 0. Đưa code lên server

Trên máy của bạn, commit rồi push phần thay đổi:

```bash
git add -A && git commit -m "PCP: full-gradient evaluation, val split, run guide" && git push
```

Trên login node:

```bash
ssh <user>@172.29.74.80
cd /media/<user>                                  # hoặc /media02/<user>; khi đó: export CDP_WORK=/media02/<user>
git clone <url của repo> ContrastDiffPurification
cd ContrastDiffPurification
git checkout main                                 # PCP nằm trên main (đã merge refactor/structure)
mkdir -p logs/slurm
```

`experiments/slurm/env.sh` sẽ từ chối chạy nếu repo không nằm dưới `/media/*` hoặc `/media02/*`.

## 1. Môi trường

```bash
# 1a. Login node không có driver NVIDIA (không có nvidia-smi), và node GPU không khai báo feature
#     driver trong Slurm. Không cần chờ một job GPU để xem: wheel PyTorch tự mang CUDA runtime,
#     và mọi bản cu12x chạy trên driver dòng CUDA 12 (minor version compatibility).
#     Job GPU đầu tiên (smoke.sbatch, bước 3) sẽ in nvidia-smi và xác nhận.

# 1b. venv + PyTorch + các package (login node, một lần), dùng bản cu12x thấp nhất
TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash experiments/slurm/setup_env.sh
source experiments/slurm/env.sh && cdp_activate
python -c "import torch; print(torch.__version__, 'CUDA', torch.version.cuda)"   # phải là 12.x

# 1c. build score_sde NGAY TRÊN LOGIN NODE (không cần GPU). Dùng CUDA 12.3 tại
#     /usr/local/cuda-12.3 (cùng major 12 với torch cu121). LƯU Ý: /usr/local/cuda trên login01 là
#     CUDA 11.8, không dùng được với torch cu12x. Biên dịch cho V100 (7.0) và A100 (8.0).
/usr/local/cuda-12.3/bin/nvcc --version | tail -2                      # phải là release 12.3
CUDA_HOME=/usr/local/cuda-12.3 FORCE_CUDA=1 bash experiments/slurm/build_score_sde.sh
```

Nếu `smoke.sbatch` ở bước 3 báo driver quá cũ cho CUDA 12 (tức node GPU dùng driver dòng CUDA 11),
quay lại dùng bộ CUDA 11.8: cài lại torch bản `cu118`, rồi build lại score_sde bằng module
CUDA 11.8 của cụm:

```bash
source experiments/slurm/env.sh && cdp_activate
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
module load cuda-11.8.0-gcc-11.4.0-cuusula
CUDA_HOME=$(dirname $(dirname $(which nvcc))) FORCE_CUDA=1 bash experiments/slurm/build_score_sde.sh
```

Các job Slurm chỉ cần CUDA runtime đi kèm torch, nên không cần đặt `CDP_MODULES` cho chúng.

`setup_env.sh` đã chạy `pytest tests`. Nếu có test của PCP lỗi ở bước này, dừng lại và sửa trước.
Nếu `build_score_sde.sh` in cảnh báo "`import score_sde` failed here (no GPU driver?)" thì không sao:
phần build đã xong, và `smoke.sbatch` sẽ kiểm tra lại trên node GPU.

## 2. Dữ liệu và checkpoint

### 2a. Tải tự động (login node, cần internet)

```bash
bash experiments/slurm/prepare_assets.sh          # CIFAR-10 + WRN-28-10 RobustBench "Standard" + score model
```

Script tải:

| Thứ | Nguồn | Đặt tại |
|---|---|---|
| CIFAR-10 (train **và** test, cùng một file nén) | torchvision | `data/cifar10/cifar-10-batches-py/` |
| Classifier `cifar10-wideresnet-28-10` | RobustBench `Standard`, Linf | `pretrained/robustbench/cifar10/Linf/Standard.pt` |
| Score model CIFAR-10 (`vp/cifar10_ddpmpp_deep_continuous`) | Google Drive của DiffPure | `pretrained/score_sde/checkpoint_8.pth` |

Nếu `gdown` báo hết quota Google Drive, script sẽ in ra dòng `MANUAL: ...`. Khi đó tải file bằng
trình duyệt từ
`https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif` rồi chép lên server:

```bash
scp checkpoint_8.pth <user>@172.29.74.80:/media/<user>/ContrastDiffPurification/pretrained/score_sde/
```

Nếu đã có sẵn CIFAR-10 dạng `cifar-10-batches-py/`, chỉ cần đặt vào `data/cifar10/`; torchvision
kiểm tra checksum và không tải lại.

### 2b. Encoder của PCP (login node)

`prepare_assets.sh` **không** tải encoder robust của PCP. Node GPU có thể không có internet, nên
hãy tải trước ở login node. Lệnh dưới đây đồng thời kiểm tra tên model trong
`configs/method/pcp.yaml` có tồn tại trong RobustBench không:

```bash
source experiments/slurm/env.sh && cdp_activate
python - <<'EOF'
from robustbench import load_model
from src.method_pcp import load_pcp_config
from src.models.classifiers import ROBUSTBENCH_MODEL_DIR
name = load_pcp_config().encoder.split(":")[1]
load_model(model_name=name, model_dir=ROBUSTBENCH_MODEL_DIR, dataset="cifar10", threat_model="Linf")
print("encoder ready:", name)
EOF
```

Nếu RobustBench báo không có model này, chọn một model ℓ∞ CIFAR-10 khác (ResNet-18 hoặc
WRN-28-10) trong danh sách của RobustBench, sửa dòng `encoder:` và `bank:` trong
`configs/method/pcp.yaml`, rồi chạy lại lệnh trên.

### 2c. Cấu trúc thư mục sau khi xong

```
ContrastDiffPurification/
├── data/
│   └── cifar10/cifar-10-batches-py/      data_batch_1..5, test_batch, batches.meta
├── pretrained/
│   ├── score_sde/checkpoint_8.pth
│   ├── robustbench/cifar10/Linf/Standard.pt
│   ├── robustbench/cifar10/Linf/<encoder PCP>.pt
│   └── pcp_banks/cifar10_<encoder>_M500.pt     ← tạo ở bước 5
├── outputs/                              ← kết quả (tạo tự động)
└── logs/slurm/                           ← log của job
```

Ba thư mục gốc có thể chuyển sang chỗ khác (ví dụ một ổ chung) bằng biến môi trường
`CDP_DATA_ROOT`, `CDP_PRETRAINED_ROOT`, `CDP_OUTPUT_ROOT` (xem `src/paths.py`). Nếu dùng, đặt chúng
trong `~/.bashrc` để mọi job cùng thấy.

Cách chia dữ liệu mà các thí nghiệm dùng (cố định trong code, không cần thao tác):

| Phần | Ảnh | Dùng cho |
|---|---|---|
| Train 0 – 44 999 | 45 000 ảnh train | bank prototype (lấy ngẫu nhiên 20 000 ảnh) |
| Train 45 000 – 49 999 | 5 000 ảnh train giữ riêng | tune siêu tham số (`--eval_split val`) |
| Test | 10 000 ảnh test | kết quả báo cáo: 512 ảnh chọn theo `data_seed` (như DiffPure) |

## 3. Kiểm tra

```bash
source experiments/slurm/env.sh && cdp_activate
python -m pytest tests -q                         # CPU; phải qua hết
sbatch experiments/slurm/smoke.sbatch             # baseline + contrastive bản gốc trên 4 ảnh
cat logs/slurm/cdp-smoke-<job_id>.out
```

## 4. Đo throughput

```bash
source experiments/pcp/gpu_job.sh
gpu_cmd calibrate 1 python -m scripts.calibrate_throughput --config cifar10.yml --batch_sizes 64,128,256
```

Log in ra số ảnh·NFE/giây và bảng GPU-ngày ước tính cho các cấu hình. Dùng bảng này để đặt
`--time` cho các bước sau, và để chọn số shard của bảng chính (bước 8).

## 5. Bank prototype (một lần cho mỗi encoder)

```bash
gpu_cmd build-bank 4 python -m scripts.build_prototype_bank --config cifar10.yml --domain cifar10 \
    --n_prototypes 500 --t_values 0,0.025,0.05,0.075,0.1 --n_images 20000 --seed 0
ls -lh pretrained/pcp_banks/                      # file .pt phải có mặt
```

Mặc định, encoder và đường dẫn ra lấy từ `configs/method/pcp.yaml`. Với encoder khác, thêm
`--encoder <tên> --out pretrained/pcp_banks/<tên file>.pt`; khi chạy thì trỏ tới file đó bằng
`--pcp_encoder <tên> --pcp_bank pcp_banks/<tên file>.pt`.

## 6. Pilot: kiểm tra attack (bắt buộc)

64 ảnh test, `data_seed 0`. Năm job độc lập, có thể nộp cùng lúc:

```bash
source experiments/pcp/gpu_job.sh
BASE="--config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 \
      --diffusion_type particle --attack_version pgd_full --lp_norm Linf --seed 121 --data_seed 0 \
      --num_sub 64 --adv_batch_size 32 --exp ./pilot"
EPS=0.031373

# --time from the A100-40GB calibration (attack throughput ~327 img*NFE/s at large batch,
# assumed ~230 at an effective batch of 32), plus margin
gpu_job p-diffpure   4 $BASE --guidance none --pcp_n_particles 1 --adv_eps $EPS --pgd_steps 200 --pgd_eot_iter 4 -i diffpure_sp
gpu_job p-eps1       2 $BASE --guidance none --pcp_n_particles 1 --adv_eps 1    --pgd_steps 50  --pgd_eot_iter 4 -i diffpure_sp_eps1
gpu_job p-steps100   2 $BASE --guidance none --pcp_n_particles 1 --adv_eps $EPS --pgd_steps 100 --pgd_eot_iter 4 -i diffpure_sp_steps100
gpu_job p-eot8       6 $BASE --guidance none --pcp_n_particles 1 --adv_eps $EPS --pgd_steps 200 --pgd_eot_iter 8 -i diffpure_sp_eot8

# PCP probe: 8 images, 10 iterations; measures seconds per batch and peak GPU memory, from which
# the PCP batch size and the --time of the PCP pilot, tuning and main-table jobs are set
gpu_job p-pcp-probe  2 ${BASE/--num_sub 64 --adv_batch_size 32/--num_sub 8 --adv_batch_size 8} \
    --guidance pcp --pcp_n_particles 8 --adv_eps $EPS --pgd_steps 10 --pgd_eot_iter 4 -i pcp_probe

# tham chiếu: đánh giá straight-through của bản phát hành, trên đúng 64 ảnh đó
SEEDS="121" DATA_SEEDS="0" sbatch --array=0 experiments/slurm/run_array.sbatch experiments/baseline/run_cifar_eotpgd.sh
```

Đọc kết quả:

```bash
python -m scripts.aggregate_pgd_full outputs/pilot
```

Chỉ đi tiếp khi **cả bốn** điều kiện sau đều đạt:

1. `diffpure_sp`: robust accuracy cùng khoảng với Lee & Kim (46.84% khi backprop trực tiếp; sampler
   20 bước ở đây không trùng khớp hoàn toàn), và thấp hơn rõ rệt so với log của
   `run_cifar_eotpgd.sh` (straight-through);
2. `diffpure_sp_eps1`: robust accuracy ≈ 0%;
3. `steps100` → `diffpure_sp` (200 vòng): robust accuracy không tăng;
4. `eot8`: robust accuracy không cao hơn `diffpure_sp` (EOT 4).

Nếu không đạt, attack đang yếu (gradient bị che). Gửi log để tìm nguyên nhân trước khi tốn GPU cho
các bước sau. Job `pcp_probe` đo thời gian mỗi batch và bộ nhớ GPU đỉnh của PCP (dòng `batch 0: ..., peak GPU mem`).
Từ đó chọn `--adv_batch_size` cho PCP (số ảnh × K particle phải vừa 40 GB) và `--time` cho các
bước 7–8. Với K = 8, DiffPure-MV và contrastive-MV nên dùng `--adv_batch_size 16` (batch hiệu dụng 128,
khoảng 17 GiB khi đo ở bước 4).

## 7. Tune trên tập validation

256 ảnh từ tập val, attack rẻ (50 vòng, EOT 4). Không bao giờ dùng tập test ở bước này.

```bash
source experiments/pcp/gpu_job.sh
TB="--config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 --diffusion_type particle --attack_version pgd_full --lp_norm Linf --adv_eps 0.031373 --pgd_steps 50 --pgd_eot_iter 4 --eval_split val --num_sub 256 --seed 121 --data_seed 0 --exp ./tune --score_reps 8 --resume --pcp_n_particles 8"

# two lanes (at most 2 running jobs); one-factor-at-a-time around the default (sem 0.1, rep 0.1)
export SBATCH_DEPENDENCY=singleton
gpu_job laneA 22 $TB --adv_batch_size 16 --guidance none -i diffpure_mv
gpu_job laneB 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0.1  --pcp_rep_strength 0.1 -i pcp_s0.1_r0.1
gpu_job laneA 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0.2  --pcp_rep_strength 0.1 -i pcp_s0.2_r0.1
gpu_job laneB 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0.05 --pcp_rep_strength 0.1 -i pcp_s0.05_r0.1
gpu_job laneA 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0.1  --pcp_rep_strength 0   -i pcp_s0.1_r0
gpu_job laneB 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0.1  --pcp_rep_strength 0.2 -i pcp_s0.1_r0.2
gpu_job laneA 26 $TB --adv_batch_size 8 --guidance pcp --pcp_sem_strength 0    --pcp_rep_strength 0.1 -i pcp_s0_r0.1
unset SBATCH_DEPENDENCY

# after (or during) the runs: paired comparison on the common images, DiffPure-MV as reference
for d in $(find outputs/tune -name pgd_full_batch000.pt -exec dirname {} \;); do python -m scripts.summarize_pgd_full "$d" > /dev/null; done
python -m scripts.aggregate_pgd_full outputs/tune
python -m scripts.compare_pgd_full outputs/tune/diffpure_mv outputs/tune/pcp_*
```

Vòng thứ hai, quanh cấu hình tốt nhất: `--pcp_n_particles 4`, `--pcp_tau_s 0.2 --pcp_tau_t 0.1`,
và encoder không robust (`--pcp_encoder cifar10-wideresnet-28-10` cùng một bank xây cho encoder đó).

**Go / no-go:** PCP phải hơn `diffpure_mv` khoảng ≥ 3 điểm robust accuracy. Nếu đạt, ghi cấu hình
tốt nhất vào `configs/method/pcp.yaml`, commit, rồi chạy bảng chính với cấu hình đó. Nếu không
đạt, dừng lại trước bước 8 và xem lại hướng đi (phương án dự phòng trong kế hoạch).

## 8. Bảng chính: 5 phương pháp × 3 seed, 512 ảnh test

Mỗi script trong `experiments/pcp/` chạy 512 ảnh. Chia mỗi (phương pháp, seed) thành 4 shard,
mỗi shard là một task Slurm một GPU, tổng cộng 5 × 3 × 4 = 60 task. Đặt `--time` theo bước 4 (cộng
thêm khoảng 20%). **Không** đặt `T_LIST` cho các script này.

```bash
for s in experiments/pcp/run_cifar_pgd_full_*.sh; do
  for i in 0 1 2 3; do
    SHARD=$i/4 SEEDS="121 122 123" DATA_SEEDS="0" \
      sbatch --array=0-2 --time=24:00:00 experiments/slurm/run_array.sbatch "$s"
  done
done
squeue -u $USER
```

Task id 0, 1, 2 tương ứng seed 121, 122, 123. Một task bị hết giờ hoặc bị huỷ thì nộp lại đúng task
đó, cùng `SHARD`. Nhờ `--resume`, các batch đã lưu (`pgd_full_batch<NNN>.pt`) được bỏ qua:

```bash
SHARD=2/4 SEEDS="121 122 123" DATA_SEEDS="0" sbatch --array=1 experiments/slurm/run_array.sbatch \
    experiments/pcp/run_cifar_pgd_full_pcp.sh
```

## 9. Các bảng bổ sung

Dùng `gpu_job` như ở bước 6, với `--exp ./exp_extra` và tên `-i` riêng. Nên chạy 256 ảnh
(`--num_sub 256`), trừ khi còn dư GPU.

```bash
source experiments/pcp/gpu_job.sh
XBASE="--config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 \
       --diffusion_type particle --attack_version pgd_full --pgd_steps 200 --pgd_eot_iter 4 \
       --num_sub 256 --adv_batch_size 32 --seed 121 --data_seed 0 --exp ./exp_extra"

# encoder robust dùng một mình (bắt buộc báo cáo), đọc clf_clean / clf_adv
ENC=$(python -c "from src.method_pcp import load_pcp_config; print(load_pcp_config().encoder)")
gpu_job x-encoder 6 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance none --pcp_n_particles 1 \
    --classifier_name $ENC --attack_defense false --num_sub 512 -i encoder_alone

# l2, eps 0.5
for g in none pcp; do
  gpu_job x-l2-$g 24 $XBASE --lp_norm L2 --adv_eps 0.5 --guidance $g --pcp_n_particles 8 -i l2_${g}_mv
done

# loss DLR
gpu_job x-dlr-pcp 24 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance pcp --pgd_loss dlr -i dlr_pcp

# ablation K (so với DiffPure-MV cùng K)
for k in 1 2 4 16; do
  for g in none pcp; do
    gpu_job x-k$k-$g 24 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance $g --pcp_n_particles $k -i K${k}_${g}
  done
done

# bỏ từng thành phần của PCP
gpu_job x-no-sem 24 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance pcp --pcp_sem_strength 0 -i pcp_no_sem
gpu_job x-no-rep 24 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance pcp --pcp_rep_strength 0 -i pcp_no_rep

# bảng xác nhận gần DiffPure gốc: Euler–Maruyama 100 bước, 128 ảnh (tốn kém)
for g in none pcp; do
  gpu_job x-s100-$g 48 $XBASE --lp_norm Linf --adv_eps 0.031373 --guidance $g --pcp_n_particles 8 \
      --pcp_solver em --pcp_steps 100 --num_sub 128 -i s100_${g}
done
```

Với ablation số prototype hoặc time-bucket: xây bank mới ở bước 5 (`--n_prototypes 100` hoặc
`2000`, `--t_values 0`, cùng `--out` riêng), rồi chạy với `--pcp_bank pcp_banks/<file>.pt`.

## 10. Tổng hợp kết quả

```bash
# gộp shard của từng lần chạy (bảng chính)
for d in outputs/exp_results_pcp/*/cifar10-wideresnet-28-10/*/seed*/data*; do
  python -m scripts.summarize_pgd_full "$d"
done

# bảng trung bình ± độ lệch chuẩn qua seed
mkdir -p results
python -m scripts.aggregate_pgd_full outputs/exp_results_pcp --csv results/main_linf.csv
python -m scripts.aggregate_pgd_full outputs/exp_extra       --csv results/extra.csv
```

Mỗi thư mục chạy có dạng
`outputs/<exp>/<tên -i>/cifar10-wideresnet-28-10/particle_<guidance>_pgd_full/seed<S>/data<D>/`, gồm:

| File | Nội dung |
|---|---|
| `log.txt` / `log_shard<i>of<n>.txt` | log đầy đủ |
| `resolved_config.json`, `eval_indices.json` | cấu hình hợp nhất, chỉ số các ảnh đã chấm |
| `pgd_full_batch<NNN>.pt` | ảnh adversarial, cờ bị phá trong lúc attack, dự đoán của từng particle |
| `pgd_full_summary.json` | `def_clean`, `def_adv` (robust accuracy), `clf_*`, Wilson 95%, `agreement_*` (ρ, `k_eff`) |

Robust accuracy để báo cáo là `def_adv`. Tương quan lỗi giữa các particle (`agreement_adv.error_corr`)
là trục x của hình phân tích cơ chế (robust accuracy theo ρ).

Sao lưu định kỳ vì server có thể dọn dữ liệu:

```bash
rsync -a outputs logs results <máy khác>:/backup/cdp/
```

## 11. Chạy không có Slurm (một máy nhiều GPU)

Bỏ `gpu_job` và `sbatch`; chạy trực tiếp `python -m scripts.eval_robustness ...` với cùng tham số.
Với bảng chính, mỗi GPU một tiến trình:

```bash
for seed in 121 122 123; do
  for i in 0 1 2 3; do bash experiments/pcp/run_cifar_pgd_full_pcp.sh $i $seed 0 $i/4 & done
  wait
done
```

`--diffusion_type particle` không dùng `DataParallel`. Chạy nhiều GPU trong một tiến trình sẽ báo
lỗi; luôn chia bằng `--shard`.

## 12. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `CUDA out of memory` | giảm `--adv_batch_size` (32 → 16 → 8) |
| `prototype bank ... not found` | chạy bước 5; kiểm tra `bank:` trong `pcp.yaml` hoặc `--pcp_bank` |
| RobustBench báo không có model | sửa `encoder:` trong `pcp.yaml` (bước 2b) |
| `--diffusion_type particle runs on one GPU per process` | job thấy nhiều GPU: đặt `CUDA_VISIBLE_DEVICES` về một GPU |
| `pgd_full on the sde/ode purifier needs --grad_mode full` | dùng `--diffusion_type particle`, hoặc thêm `--grad_mode full` |
| `import score_sde` lỗi | build lại ở bước 1c trên node GPU, sau khi `module load` CUDA |
| job biến mất khỏi `squeue` | đọc `logs/slurm/<tên>-<jobid>.out` |
| hết giờ | nộp lại cùng task/shard; `--resume` chạy tiếp từ batch dang dở |

Các giá trị chưa được xác minh cần kiểm tra trước khi báo cáo số: tên encoder (bước 2b) và step
size ℓ∞ mặc định 0.007 (`default_step_size` trong `src/evaluation/attacks_full.py`; có thể đặt tường
minh bằng `--pgd_step_size`).
