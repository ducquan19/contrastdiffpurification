# Tái lập các thí nghiệm CIFAR-10

> **Trạng thái.** Code đã được tái cấu trúc và kiểm chứng là tương đương về mặt cấu trúc với repo
> gốc (§6). **Chưa có thí nghiệm nào được chạy lại sau khi tái cấu trúc**, nên chưa con số
> robustness nào được tái lập. Hãy coi lần chạy đầy đủ đầu tiên chính là bước tái lập.

## 1. Phần cứng và môi trường

* GPU NVIDIA. DiffPure, nền tảng của code này, cho biết họ dùng 1–4 GPU, mỗi GPU 32 GB. Các attack
  thích nghi đi qua adjoint solver rất chậm, nên hãy tính đến hàng giờ cho mỗi tổ hợp
  seed × data-seed với các lần chạy AutoAttack `rand`.
* Python ≥ 3.9 (code dùng `str.removeprefix`).

```bash
conda create -n cdp python=3.10 -y && conda activate cdp
pip install torch torchvision            # chọn bản CUDA khớp driver của bạn
pip install -r requirements.txt
pip install ./external/auto-attack       # AutoAttack đi kèm (hỗ trợ state_path để chạy tiếp)
pip install --no-build-isolation ./external/score_sde   # biên dịch CUDA op của score_sde (trên máy có GPU + nvcc)
pip install -e .                         # không bắt buộc: giúp `import src` chạy từ mọi thư mục
```

Kiểm tra cài đặt:

```bash
python -m pytest tests -q                # chỉ CPU; test điểm vào sẽ chạy khi có đủ dependency
python -c "import score_sde, autoattack, torchsde, torchdiffeq, torchattacks, robustbench; print('ok')"
```

`setup.py` của score_sde chỉ biên dịch CUDA op khi thấy GPU; build trên máy không có GPU sẽ khiến
`import score_sde` lỗi. Trên cụm Slurm, dùng `experiments/slurm/build_score_sde.sh` qua `srun`.

## 2. Checkpoint (`pretrained/`)

Với CIFAR-10 chỉ cần những file sau:

| File | Dùng cho | Nguồn |
|---|---|---|
| `pretrained/score_sde/checkpoint_8.pth` | diffusion model (mọi lần chạy CIFAR-10) | score_sde_pytorch `vp/cifar10_ddpmpp_deep_continuous` ([link từ README của DiffPure](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view?usp=sharing)) |
| `pretrained/robustbench/…` | `cifar10-wideresnet-28-10` (cũng là classifier cho contrastive) | RobustBench tự tải (`Standard`, Linf) |
| `pretrained/cifar10/resnet-50/weights.pt` | các script `*_rn50*` | [link DiffPure](https://drive.google.com/drive/folders/1SEGilIEAnx9OC1JVhmOynreCF3oog7Fi?usp=sharing) |
| `pretrained/cifar10/wrn-70-16-dropout/weights.pt` | các script `*_70-16-dp*` | [link DiffPure](https://drive.google.com/drive/folders/1istqcnPNXJ-TQFdxRqNJuInzOtlYOG1b?usp=sharing) |

Có thể chuyển hướng các đường dẫn này bằng `CDP_PRETRAINED_ROOT=/duong/dan/checkpoint`.

## 3. Dữ liệu (`data/`)

CIFAR-10 được torchvision tự tải vào `data/cifar10/` ở lần dùng đầu tiên (đổi thư mục gốc bằng
`CDP_DATA_ROOT`). Các thư mục dữ liệu là `data/cifar10`, `data/cifar100` và `data/gtsrb`; chúng
được khai báo trong [`src/data/specs.py`](../src/data/specs.py).

Mỗi lần chạy đánh giá `--num_sub` ảnh test được chọn theo `--data_seed`
(`RandomState(data_seed).choice(len(test_split), num_sub, replace=False)`) và ghi các chỉ số đã
chọn vào `eval_indices.json` trong thư mục của lần chạy, nên biết chính xác những ảnh nào đã dùng.

Nếu bạn đã có CIFAR-10 ở `data/cifar-10-batches-py` theo layout cũ, hãy chuyển nó sang
`data/cifar10/cifar-10-batches-py` để không phải tải lại.

## 4. Chạy

Chạy từ đâu cũng được; mỗi script tự `cd` về thư mục gốc của repo.

```bash
# kiểm tra nhanh toàn tuyến: một seed, một subset dữ liệu
bash experiments/baseline/run_cifar_rand_inf.sh    0 121 0
bash experiments/contrastive/run_cifar_rand_inf.sh 0 121 0

# giao thức đầy đủ theo DiffPure: seed 121..123 x data seed 0..7 (mỗi lần 64 ảnh)
bash experiments/contrastive/run_cifar_rand_inf.sh 0 "121 122 123" "0 1 2 3 4 5 6 7"
```

Tham số: `<gpu_id> <các seed> <các data_seed>` (script trong `ablation_t/` nhận thêm tham số thứ
4 là danh sách `t`). [experiments/README.md](../experiments/README.md) liệt kê từng script và cấu
hình của nó.

Trên cụm Slurm, mỗi cặp (seed, data_seed) là một task của job array, chạy trên một GPU riêng:

```bash
sbatch experiments/slurm/run_array.sbatch experiments/contrastive/run_cifar_rand_inf.sh
```

Cài đặt, tải checkpoint, chạy thử và chạy tiếp khi hết giờ: xem
[experiments/slurm/README.md](../experiments/slurm/README.md).

## 5. Kết quả

Mỗi lần chạy ghi vào
`outputs/<--exp>/<-i>/<classifier>/<diffusion_type>_<attack_version>/seed<S>/data<D>/`:

| File | Nội dung |
|---|---|
| `log.txt` | toàn bộ stdout. EOT-PGD / APGD / StAdv / BPDA in `initial accuracy` và `robust accuracy` ở đây |
| `eval_indices.json` | dataset, `num_sub`, `data_seed` và đúng các chỉ số ảnh test đã đánh giá |
| `resolved_config.json` | mọi thiết lập lần chạy đã dùng: dòng lệnh, tên file config và siêu tham số phương pháp sau khi hợp nhất |
| `log_resnet.txt`, `log_sde_adv.txt` | log AutoAttack cho classifier trần và cho purifier + classifier |
| `x_adv_resnet_sd<S>.pt`, `x_adv_sde_sd<S>.pt` | các mẫu đối kháng |
| `bs0_*/`, `bs1_*/` | ảnh đầu vào, ảnh đã thêm nhiễu và ảnh đã thanh lọc, cho hai batch đầu |

Hiện chưa có script tổng hợp. Hãy lấy trung bình robust accuracy trên toàn bộ các thư mục
`seed*/data*`. DiffPure báo cáo giá trị nhỏ nhất giữa hai phiên bản AutoAttack `rand` và `standard`
làm robustness trường hợp xấu nhất. Repo này chỉ kèm script `rand`; muốn `standard` thì dùng
`--attack_version standard`.

## 6. Những gì đã được kiểm chứng

Trên một máy chỉ có CPU và thiếu nhiều dependency (các gói thiếu được thay bằng stub):

| Hạng mục | Kết quả |
|---|---|
| Cú pháp Python của toàn bộ file | OK |
| Mọi module đang hoạt động đều import được (`src.*`, `scripts.*`) | OK, 25/25 |
| 21 script thí nghiệm parse ra cùng bộ tham số như các chương trình gốc (chỉ trừ các flag vô dụng đã xóa và thư mục output, giờ là `outputs/`) | OK, 21/21 |
| Đầu ra của `ContrastiveGuidedDiffusion` trên đầu vào toy cố định, cũ so với mới | giống hệt từng bit |
| Drift / diffusion của reverse SDE và drift của ODE (dùng `score_sde.sde_lib` thật), cũ so với mới | giống hệt từng bit |
| Chỉ số subset đánh giá khớp `RandomState(data_seed).choice(...)`, đúng công thức gốc | OK (`tests/test_data.py`) |
| Pipeline `Resize + ToTensor` dùng chung bằng với từng transform riêng của bản gốc | giống hệt từng bit (`tests/test_data.py`) |
| Cả ba config nạp được dưới schema strict; khóa thiếu / không hợp lệ đều bị báo | OK (`tests/test_config.py`) |
| Method config khớp script CIFAR-10 gốc; CLI ghi đè được | OK (`tests/test_method_config.py`) |
| Siêu tham số contrastive sau khi resolve cho từng script, cũ so với mới | giống hệt với 9/10; script BPDA contrastive thay đổi có chủ ý (xem dưới) |
| `--help` của mọi điểm vào; chuyển đổi checkpoint trên một checkpoint Lightning giả | OK |
| Điểm vào đánh giá chạy tới bước nạp `pretrained/score_sde/checkpoint_8.pth` | tới đúng đường dẫn mong đợi |

**Chưa kiểm chứng:** mọi thứ cần GPU, dependency thật, checkpoint hoặc dữ liệu — tức là bản thân
quá trình thanh lọc, các attack, các script huấn luyện ngoài phần parse tham số, và do đó là
toàn bộ các con số được báo cáo.

## 7. Các vấn đề đã biết ảnh hưởng tới việc tái lập

* `experiments/contrastive/run_cifar_bpda_eot.sh` trước đây crash (vấn đề B1) và giờ chạy được.
  Do siêu tham số của phương pháp đã được thống nhất về bộ giá trị CIFAR-10, nó dùng tau = 1 và
  lambda = 500 thay cho default cũ của entry point là 10 và 5000. Không con số đã công bố nào bị
  ảnh hưởng, vì script này trước đó không thể chạy.
* `drift_min` vẫn khác nhau giữa các script (0 ở script AutoAttack / EOT-PGD, 40 ở script BPDA và
  ablation). Config chuẩn ghi 0; các script được giữ nguyên như bản công bố. Xem
  `experiments/README.md`. Cần chốt giá trị trước khi báo cáo số.
* Kết quả GTSRB **không** so sánh được với code gốc: bộ nạp dữ liệu của nó trả về `adv_batch_size`
  ảnh thay vì `num_sub`, lỗi này đã được sửa. CIFAR-10 không bị ảnh hưởng.
* **Attacker không nhận được gradient của purifier** ở cả hai nhánh đánh giá, nên những gì repo
  này đo ứng với Bảng 7 của bài báo, không phải cấu hình gradient-on của các bảng chính. Xem
  "Threat model" trong [research-architecture.md](research-architecture.md) §7 — đây là điều cần
  xác nhận trước khi đối chiếu với bài báo.
* Kết quả không tái lập được từng bit giữa các lần chạy (`cudnn.benchmark = True`, nhiễu của SDE).
  Hãy chạy cùng một cấu hình hai lần để đo độ biến động trước khi so sánh các phương pháp.
