# Chạy thí nghiệm trên cụm Slurm

Các script trong thư mục này tuân theo quy định của server: code, dữ liệu, môi trường conda và mọi
cache (kể cả `.cache`) đều nằm dưới `/media/<user>` (hoặc `/media02/<user>`); login node chỉ dùng
để chuẩn bị và nộp job; chạy thử bằng `srun` trước khi nộp `sbatch` dài.

| File | Chạy ở đâu | Việc |
|---|---|---|
| `env.sh` | được các script khác `source` | đường dẫn lưu trữ, chuyển cache khỏi `/home`, kích hoạt conda |
| `setup_env.sh` | login node, một lần | tạo môi trường conda, cài package, chạy test CPU |
| `prepare_assets.sh` | login node, một lần | tải CIFAR-10, classifier RobustBench và checkpoint diffusion |
| `smoke.sbatch` | `sbatch` | chạy thử đầu-cuối trên 4 ảnh, đo thời gian |
| `run_array.sbatch` | `sbatch` | chạy một script thí nghiệm, mỗi task một cặp (seed, data_seed) |

Chỉ cần driver GPU: các config `cifar10/cifar100/gtsrb` đặt `fir: False`, nên CUDA op biên dịch
của score_sde không bao giờ được dùng và không cần `nvcc`.

## 1. Chuẩn bị (login node)

```bash
ssh <user>@172.29.74.80
cd /media/<user>                       # hoặc /media02/<user>: đặt CDP_WORK=/media02/<user>
git clone <repo> ContrastDiffPurification && cd ContrastDiffPurification
git checkout refactor/structure

# xem driver hỗ trợ CUDA bản nào (dòng "CUDA Version"), rồi chọn bản PyTorch không vượt quá nó
srun --mem=4G --gres=gpu:1 nvidia-smi

TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash experiments/slurm/setup_env.sh
bash experiments/slurm/prepare_assets.sh          # thêm --all cho các script *_rn50* / *_70-16-dp*
```

`setup_env.sh` ghi danh sách package vào `logs/env-<ngày>.txt`; giữ file này cùng kết quả.

## 2. Chạy thử

```bash
sbatch experiments/slurm/smoke.sbatch
squeue -u $USER
cat logs/slurm/cdp-smoke-<job_id>.out
```

Mọi `sbatch` phải được nộp **từ thư mục gốc của repo** (log Slurm nằm trong `logs/slurm/`).
Nếu job biến mất khỏi `squeue` ngay sau khi nộp, đọc file log đó trước tiên.

Muốn dò lỗi tương tác thì dùng:

```bash
srun --mem=16G --gres=gpu:1 --cpus-per-task=4 --pty bash
source experiments/slurm/env.sh && cdp_activate
```

## 3. Chạy thí nghiệm

```bash
# giao thức DiffPure: 3 seed x 8 data seed = 24 task (mặc định --array=0-23)
sbatch experiments/slurm/run_array.sbatch experiments/baseline/run_cifar_rand_inf.sh
sbatch experiments/slurm/run_array.sbatch experiments/contrastive/run_cifar_rand_inf.sh

# lưới khác: số task = số seed x số data seed
SEEDS="121" DATA_SEEDS="0 1" sbatch --array=0-1 experiments/slurm/run_array.sbatch <script>

# ablation theo t
T_LIST="50 75 100 125" sbatch experiments/slurm/run_array.sbatch experiments/ablation_t/contrastive_run_cifar_rand_inf.sh
```

Task id ánh xạ thành `seed = SEEDS[id / số data seed]`, `data_seed = DATA_SEEDS[id % số data seed]`.
Mỗi task dùng 1 GPU, 4 CPU, 32 GB RAM.

**Thời gian.** `--time=48:00:00` chỉ là giá trị mở đầu. Lấy thời gian của lần chạy thử (4 ảnh,
EOT 1) hoặc của một task đầy đủ đầu tiên, rồi đặt `--time` lớn hơn khoảng 20%
(`sbatch --time=... experiments/slurm/run_array.sbatch ...`). Xem giới hạn của cluster bằng
`sinfo -o "%P %l"`.

**Job hết giờ.** AutoAttack lưu tiến độ sau mỗi batch vào `adv_*_eval_state.json` trong thư mục
của lần chạy. Nộp lại đúng task đó, nó sẽ chạy tiếp (độ chính xác vẫn đúng; `x_adv_*.pt` khi đó
chỉ chứa các batch của lần chạy sau):

```bash
sbatch --array=<task_id> experiments/slurm/run_array.sbatch <script>
```

Các attack EOT-PGD / APGD / StAdv / BPDA không có trạng thái để chạy tiếp; lần nộp lại sẽ chạy từ
đầu.

## 4. Theo dõi và sao lưu

```bash
squeue -u $USER                 # đang chạy / đang chờ
squeue -u $USER --start         # giờ bắt đầu dự kiến
scontrol show job <job_id>      # lý do pending, bộ nhớ, trạng thái kết thúc
scancel <job_id>                # hủy (cả array) hoặc <job_id>_<task_id>
```

Kết quả nằm trong `outputs/` (cấu trúc thư mục: [docs/reproduction.md](../../docs/reproduction.md) §5).
Server có thể dọn dữ liệu định kỳ: sao lưu `outputs/` và `logs/` khoảng 2 ngày một lần, ví dụ
`rsync -a outputs logs <máy khác>:/backup/cdp/`. Ngắt SSH (và VS Code Remote) khi xong việc.
