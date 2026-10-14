# Contrastive DiffPure

Code cho bài báo *Diffusion Models Demand Contrastive Guidance for Adversarial Purification to
Advance* (Bai et al., ICML 2024).

Thanh lọc đối kháng bằng diffusion model (DiffPure): ảnh đầu vào được thêm nhiễu tới một mốc
thời gian khuếch tán nhỏ `t*`, rồi khử nhiễu bằng reverse SDE/ODE trước khi phân loại.
**Contrastive guidance** cộng thêm, tại mỗi bước reverse, gradient của một contrastive loss được
học trực tuyến giữa embedding của classifier ở hai bước liên tiếp vào đầu ra của score network.
MLP embedding được huấn luyện ngay trong lúc chạy cho từng batch, nên phương pháp không cần thêm
giai đoạn huấn luyện nào.

Repo này giới hạn ở các thí nghiệm **CIFAR-10** cho phương pháp và cho baseline DiffPure. Phần
huấn luyện mô hình CIFAR-100 / GTSRB được giữ lại cho công việc sau.

Repo còn chứa phần mở rộng **Particle-Contrastive Purification (PCP)** và một bộ đánh giá
**full-gradient** cho mọi phương pháp (DiffPure, ContrastDiffPure, PCP) với cùng chi phí tính toán.
Cách chạy đầy đủ nằm ở mục [Thí nghiệm PCP](#thí-nghiệm-pcp-đánh-giá-full-gradient); hướng dẫn
từng bước trên cụm Slurm, từ tải dữ liệu đến bảng kết quả, ở [docs/run-guide.md](docs/run-guide.md);
mô tả phương pháp ở [docs/pcp.md](docs/pcp.md).

## Cấu trúc

```
src/                       code nghiên cứu (`import src…`)
├── guidance/contrastive.py      ★ phương pháp: ContrastiveGuidedDiffusion, ContrastiveLoss
├── guidance/particle.py         ★ PCP: guidance giữa K particle của cùng một ảnh (L_sem + L_rep)
├── guidance/memory_bank.py      bank prototype sạch (spherical k-means, theo time-bucket)
├── guidance/encoders.py         encoder F đông cứng cho PCP
├── models/diffusion/            bộ thanh lọc DiffPure: reverse SDE (sde.py), probability-flow ODE (ode.py),
│                                sampler K particle khả vi cho đánh giá full-gradient (sampler.py)
├── models/classifiers/          tập classifier (RobustBench / các model CIFAR-10 của DiffPure)
├── attacks/                     attack từ thượng nguồn: bpda_eot/, stadv_eot/
├── data/                        specs.py (registry dataset), loaders.py (subset đánh giá), gtsrb.py
├── evaluation/                  model đã thanh lọc, các attack, driver đánh giá và CLI
├── training/                    Lightning module để huấn luyện score-SDE và classifier
├── config.py                    schema config cho diffusion model + loader strict
├── method.py                    siêu tham số contrastive (configs/method/) + hợp nhất với CLI
├── method_pcp.py                siêu tham số sampler + PCP (configs/method/pcp.yaml), flag --pcp_<field>
└── paths.py                     vị trí của data/ pretrained/ outputs/ configs/
scripts/                   điểm vào: eval_robustness, eval_bpda, train_*, chuyển checkpoint, tải data
experiments/               mỗi thí nghiệm một shell script (baseline/, contrastive/, ablation_t/, training/); slurm/ để chạy trên cụm
configs/                   config diffusion theo dataset + method/contrastive.yaml (xem configs/README.md)
tests/                     test chạy trên CPU (pytest)
docs/                      kiến trúc, tái lập, hướng dẫn phát triển
external/                  package thượng nguồn cài bằng pip: auto-attack (autoattack), score_sde
data/  pretrained/  outputs/     bị git bỏ qua: dataset, checkpoint, kết quả
```

## Bắt đầu nhanh

```bash
pip install torch torchvision                 # bản CUDA khớp driver của bạn
pip install -r requirements.txt
pip install ./external/auto-attack
pip install --no-build-isolation ./external/score_sde   # biên dịch CUDA op: cần GPU + nvcc
pip install -e .                              # không bắt buộc

# đặt score model CIFAR-10 tại pretrained/score_sde/checkpoint_8.pth (xem docs/reproduction.md)

bash experiments/baseline/run_cifar_rand_inf.sh    0 121 0     # DiffPure,     GPU 0, seed 121, data seed 0
bash experiments/contrastive/run_cifar_rand_inf.sh 0 121 0     # Contrastive DiffPure
```

Hoặc gọi trực tiếp điểm vào từ thư mục gốc của repo:

```bash
python -m scripts.eval_robustness --guidance contrastive --config cifar10.yml \
    --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --diffusion_type sde --t 100 \
    --attack_version rand --adv_eps 0.031373 --num_sub 64 --adv_batch_size 64 \
    --contrastive_classifier cifar10-wideresnet-28-10 --exp ./my_run
```

Kết quả nằm ở `outputs/<exp>/…/seed<S>/data<D>/log*.txt`.

> **Lưu ý về threat model.** Các script `experiments/baseline/` và `experiments/contrastive/` tái
> lập *bản phát hành*: purifier là phép đồng nhất trong backward (straight-through), tương ứng
> Bảng 7 của bài báo. Muốn đo robustness trước attacker có gradient đầy đủ, dùng các thí nghiệm
> dưới đây.

## Thí nghiệm PCP (đánh giá full-gradient)

Mọi phương pháp chạy trên **cùng một sampler** (`--diffusion_type particle`: vòng reverse tường
minh, attack backprop trực tiếp qua mọi bước và qua guidance), cùng `t* = 100`, cùng số bước
(`--pcp_steps`, mặc định 20 bước DDIM η = 1). Chỉ khác nhau ở guidance và số particle K:

| Script (`experiments/pcp/`) | Phương pháp | K |
|---|---|---|
| `run_cifar_pgd_full_diffpure_sp.sh` | DiffPure | 1 |
| `run_cifar_pgd_full_diffpure_mv.sh` | DiffPure + majority vote | 8 |
| `run_cifar_pgd_full_contrastive_sp.sh` | ContrastDiffPure (guidance hiện với attacker, đã sửa rò rỉ cache) | 1 |
| `run_cifar_pgd_full_contrastive_mv.sh` | ContrastDiffPure + majority vote | 8 |
| `run_cifar_pgd_full_pcp.sh` | **PCP** | 8 |

Attack mặc định của các script: PGD+EOT full-gradient, ℓ∞ ε = 8/255, 200 vòng, EOT 4, loss CE
trên soft vote, 512 ảnh test (`--num_sub 512`, cùng cách chọn ảnh theo `data_seed` như DiffPure).
Mỗi lệnh: `bash <script> <gpu> <seed> <data_seed> [shard i/n]`.

### Bước 0. Môi trường, dữ liệu, kiểm thử

```bash
# trên cụm: theo experiments/slurm/README.md (setup_env.sh, prepare_assets.sh, build_score_sde.sh)
python -m pytest tests -q                      # CPU; phải qua hết trước mọi job GPU
python -c "from robustbench.model_zoo import model_dicts"   # sau đó kiểm tra tên encoder bên dưới
```

Trước lần chạy đầu tiên, kiểm tra hai giá trị **chưa được xác minh** trong code:

1. tên model RobustBench `encoder: robustbench:Rebuffi2021Fixing_R18_ddpm` trong
   `configs/method/pcp.yaml` (sửa nếu RobustBench không có tên này);
2. step size ℓ∞ mặc định 0.007 (`default_step_size` trong `src/evaluation/attacks_full.py`):
   đối chiếu với code của Lee & Kim (ICCV 2023), hoặc đặt tường minh bằng `--pgd_step_size`.

### Bước 1. Đo throughput và ước lượng thời gian

```bash
bash experiments/pcp/calibrate.sh 0
```

In số ảnh·NFE/giây của score network và số GPU-ngày ước tính cho từng cấu hình. Dùng con số này
để đặt `--time` cho Slurm và chọn số shard. Nếu hết bộ nhớ GPU, giảm `--adv_batch_size` (mặc định
32 ảnh × K = 8 particle = 256 dòng mỗi lần gọi score network).

### Bước 2. Xây bank prototype (một lần cho mỗi encoder)

```bash
bash experiments/pcp/build_bank.sh 0
# -> pretrained/pcp_banks/cifar10_Rebuffi2021Fixing_R18_ddpm_M500.pt (đường dẫn `bank` trong pcp.yaml)
```

Bank dùng 20 000 ảnh **train**, không bao giờ dùng 5 000 ảnh train cuối (tập validation, bước 4)
hay tập test. Đổi encoder hoặc số prototype thì xây bank mới với `--encoder`, `--n_prototypes` và
`--out`, rồi trỏ tới nó bằng `--pcp_bank`.

### Bước 3. Pilot: kiểm tra attack trước khi tin bất kỳ con số nào

```bash
# DiffPure K = 1 trên 64 ảnh
python -m scripts.eval_robustness --guidance none --diffusion_type particle --pcp_n_particles 1 \
    --config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 \
    --attack_version pgd_full --lp_norm Linf --adv_eps 0.031373 --pgd_steps 200 --pgd_eot_iter 4 \
    --num_sub 64 --adv_batch_size 32 --seed 121 --data_seed 0 --exp ./pilot -i diffpure_sp
```

Điều kiện để đi tiếp:

* robust accuracy của DiffPure cùng khoảng với Lee & Kim (46.84% khi backprop trực tiếp, ℓ∞ 8/255;
  sampler ở đây ít bước hơn nên không kỳ vọng trùng khớp), và **thấp hơn rõ rệt** so với
  `bash experiments/baseline/run_cifar_eotpgd.sh` (straight-through);
* thêm `--adv_eps 1` (không giới hạn): robust accuracy ≈ 0%;
* tăng `--pgd_steps` 50 → 100 → 200 hoặc `--pgd_eot_iter` 2 → 4 → 8: robust accuracy không tăng.

Nếu một điều kiện không đạt, attack đang yếu (gradient bị che) và phải sửa trước khi chạy tiếp.

### Bước 4. Tune siêu tham số PCP trên tập validation (không bao giờ trên tập test)

`--eval_split val` lấy ảnh từ 5 000 ảnh train được giữ riêng (`src/data/loaders.py`, `VAL_SIZE`).
Attack rẻ hơn cho giai đoạn này: 50 vòng, EOT 4, 256 ảnh.

```bash
for sem in 0.05 0.1 0.2; do for rep in 0 0.05 0.1 0.2; do
  python -m scripts.eval_robustness --guidance pcp --diffusion_type particle \
      --config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 \
      --attack_version pgd_full --lp_norm Linf --adv_eps 0.031373 --pgd_steps 50 --pgd_eot_iter 4 \
      --eval_split val --num_sub 256 --adv_batch_size 32 --seed 121 --data_seed 0 \
      --pcp_sem_strength $sem --pcp_rep_strength $rep \
      --exp ./tune -i pcp_sem${sem}_rep${rep}
done; done
# baseline cùng cấu hình attack, cùng tập val:
python -m scripts.eval_robustness --guidance none --diffusion_type particle --pcp_n_particles 8 \
    --config cifar10.yml --domain cifar10 --classifier_name cifar10-wideresnet-28-10 --t 100 \
    --attack_version pgd_full --lp_norm Linf --adv_eps 0.031373 --pgd_steps 50 --pgd_eot_iter 4 \
    --eval_split val --num_sub 256 --adv_batch_size 32 --seed 121 --data_seed 0 \
    --exp ./tune -i diffpure_mv
```

Các trục khác nên thử: `--pcp_n_particles {4,8}`, `--pcp_tau_s {0.1,0.2}` (giữ `tau_t = tau_s/2`),
và encoder (`--pcp_encoder cifar10-wideresnet-28-10 --pcp_encoder_feature penultimate` cùng bank
tương ứng). Chọn cấu hình tốt nhất rồi ghi vào `configs/method/pcp.yaml`.

**Go / no-go:** PCP-MV phải hơn DiffPure-MV khoảng ≥ 3 điểm robust accuracy trên val ở cùng chi
phí. Nếu không đạt, xem phương án dự phòng trong kế hoạch (bài phân tích thay vì bài phương pháp).

### Bước 5. Bảng chính: 5 phương pháp × 3 seed, 512 ảnh test

Trên một máy có nhiều GPU, mỗi tiến trình một GPU, chia 512 ảnh thành các shard:

```bash
for seed in 121 122 123; do
  for i in 0 1 2 3; do bash experiments/pcp/run_cifar_pgd_full_pcp.sh $i $seed 0 $i/4 & done; wait
done
```

Trên Slurm (`run_array.sbatch` cho mỗi task một cặp (seed, data_seed); shard truyền qua biến môi
trường `SHARD`; **không** đặt `T_LIST` cho các script này):

```bash
for s in experiments/pcp/run_cifar_pgd_full_*.sh; do
  for i in 0 1 2 3; do
    SHARD=$i/4 SEEDS="121 122 123" DATA_SEEDS="0" sbatch --array=0-2 --time=24:00:00 \
        experiments/slurm/run_array.sbatch $s
  done
done
```

Job bị ngắt giữa chừng: nộp lại đúng task/shard đó. Các script có `--resume`, nên các batch đã lưu
(`pgd_full_batch<NNN>.pt`) được bỏ qua.

Gom kết quả của một lần chạy (mọi shard):

```bash
python -m scripts.summarize_pgd_full \
    outputs/exp_results_pcp/cifar10-pgdfull-linf-pcp/cifar10-wideresnet-28-10/particle_pcp_pgd_full/seed121/data0
```

`pgd_full_summary.json` chứa `def_clean`, `def_adv` (robust accuracy của phương pháp), `clf_clean`,
`clf_adv` (classifier không có purifier), tỉ lệ ảnh bị phá trong lúc attack, kèm khoảng tin cậy
Wilson 95%. `agreement_clean` / `agreement_adv` chứa tương quan lỗi giữa các particle (ρ, `k_eff`)
cho phân tích cơ chế. Báo cáo trung bình ± độ lệch chuẩn qua 3 seed.

### Bước 6. Các bảng bổ sung

Tất cả là các flag ghi đè trên các script ở bước 5 (thêm vào cuối lệnh `python`, hoặc chạy lệnh
`python -m scripts.eval_robustness` trực tiếp với `-i` riêng):

| Thí nghiệm | Flag |
|---|---|
| ℓ2, ε = 0.5 | `--lp_norm L2 --adv_eps 0.5` |
| Loss DLR | `--pgd_loss dlr` |
| Bảng xác nhận gần DiffPure gốc (128–256 ảnh) | `--pcp_solver em --pcp_steps 100 --num_sub 256` |
| Encoder robust dùng một mình (bắt buộc báo cáo) | `--classifier_name robustbench:Rebuffi2021Fixing_R18_ddpm --attack_defense false --guidance none` → đọc `clf_clean` / `clf_adv` |
| Ablation K | `--pcp_n_particles {1,2,4,8,16}` (so với DiffPure-MV cùng K) |
| Bỏ từng thành phần | `--pcp_sem_strength 0` (chỉ đẩy) · `--pcp_rep_strength 0` (chỉ kéo) |
| Encoder không robust | `--pcp_encoder cifar10-wideresnet-28-10` + bank tương ứng (`--pcp_bank`) |
| Số prototype / time-bucket | bank xây với `--n_prototypes {100,500,2000}`, `--t_values 0` (một bucket) |
| Số bước, t* | `--pcp_steps {10,20,50}` · `--t {75,100,125}` |
| ContrastDiffPure bản phát hành (chưa sửa) | bỏ `--contrastive_fix_cache true` và/hoặc `--contrastive_differentiable true` |

### Chưa được triển khai

Phần sau có trong kế hoạch nhưng **chưa có code**, nên chưa nằm trong quy trình trên: adapter cho
DiffBreak / DiffHammer, các attack thích ứng riêng cho PCP (consensus-hijack, anti-diversity,
encoder-feature), và thí nghiệm batch-coupling attack lên ContrastDiffPure. Reviewer hội nghị lớn
sẽ yêu cầu ít nhất DiffBreak (chế độ majority vote) và một attack thích ứng riêng cho PCP trước
khi chấp nhận kết quả.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/code-tour.md](docs/code-tour.md) | **bắt đầu từ đây khi đọc code**: lộ trình đọc + mô tả từng hàm |
| [docs/research-architecture.md](docs/research-architecture.md) | pipeline, phương pháp, các module, vấn đề đã biết |
| [docs/pcp.md](docs/pcp.md) | PCP, sampler K particle, đánh giá full-gradient, quy tắc không che gradient |
| [docs/run-guide.md](docs/run-guide.md) | **chạy từ đầu đến cuối**: môi trường, dữ liệu/checkpoint, thứ tự các bước, tổng hợp kết quả |
| [docs/reproduction.md](docs/reproduction.md) | môi trường, checkpoint, cách chạy, những gì đã kiểm chứng |
| [docs/development.md](docs/development.md) | cách thêm phương pháp guidance, attack hoặc dataset mới |
| [experiments/README.md](experiments/README.md) | từng script thí nghiệm và cấu hình của nó |
| [configs/README.md](configs/README.md) | mỗi mục config điều khiển gì, và cơ chế kiểm tra |

## Trạng thái

Code đã được tái cấu trúc từ bản phát hành gốc (commit `a1b2138`). Tham số của các thí nghiệm,
module contrastive và drift của SDE/ODE đã được kiểm chứng là giống hệt bản gốc.
**Các thí nghiệm chưa được chạy lại từ đó**, nên những con số trong bài báo vẫn chưa được tái lập
ở đây. Các vấn đề còn mở được liệt kê trong
[docs/research-architecture.md §7](docs/research-architecture.md).

Phần PCP và đánh giá full-gradient đã có code và test CPU, nhưng **chưa được chạy trên GPU**;
chưa có con số nào của PCP.

## Ghi công và giấy phép

Xây dựng trên [DiffPure](https://github.com/NVlabs/DiffPure) (NVIDIA Source Code License,
[LICENSES/DiffPure_LICENSE](LICENSES/DiffPure_LICENSE); áp dụng cho các file có header của NVIDIA),
[Contrastive-Guided-Diffusion-Process](https://github.com/YiDongOuYang/Contrastive-Guided-Diffusion-Process.git)
và [stablediffusion](https://github.com/Stability-AI/stablediffusion.git) (các tham chiếu của bản
phát hành gốc). Code thượng nguồn đi kèm giữ giấy phép riêng:
[score_sde](https://github.com/yang-song/score_sde_pytorch),
[AutoAttack](https://github.com/fra31/auto-attack), BPDA+EOT và ReColorAdv/mister_ed (StAdv).
Classifier lấy từ [RobustBench](https://github.com/RobustBench/robustbench). Xem [LICENSE](LICENSE).
