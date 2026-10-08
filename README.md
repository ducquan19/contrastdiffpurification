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

## Cấu trúc

```
src/                       code nghiên cứu (`import src…`)
├── guidance/contrastive.py      ★ phương pháp: ContrastiveGuidedDiffusion, ContrastiveLoss
├── models/diffusion/            bộ thanh lọc DiffPure: reverse SDE (sde.py), probability-flow ODE (ode.py)
├── models/classifiers/          tập classifier (RobustBench / các model CIFAR-10 của DiffPure)
├── models/score_sde/            score_sde (NCSN++, SDE, loss, sampler) từ thượng nguồn; CUDA op biên dịch khi dùng lần đầu
├── attacks/                     attack từ thượng nguồn: autoattack/, bpda_eot/, stadv_eot/
├── data/                        specs.py (registry dataset), loaders.py (subset đánh giá), gtsrb.py
├── evaluation/                  model đã thanh lọc, các attack, driver đánh giá và CLI
├── training/                    Lightning module để huấn luyện score-SDE và classifier
├── config.py                    schema config cho diffusion model + loader strict
├── method.py                    siêu tham số contrastive (configs/method/) + hợp nhất với CLI
└── paths.py                     vị trí của data/ pretrained/ outputs/ configs/
scripts/                   điểm vào: eval_robustness, eval_bpda, train_*, chuyển checkpoint, tải data
experiments/               mỗi thí nghiệm một shell script (baseline/, contrastive/, ablation_t/, training/); slurm/ để chạy trên cụm
configs/                   config diffusion theo dataset + method/contrastive.yaml (xem configs/README.md)
tests/                     test chạy trên CPU (pytest)
docs/                      kiến trúc, tái lập, hướng dẫn phát triển
data/  pretrained/  outputs/     bị git bỏ qua: dataset, checkpoint, kết quả
```

## Bắt đầu nhanh

```bash
pip install torch torchvision                 # bản CUDA khớp driver của bạn
pip install -r requirements.txt
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

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/code-tour.md](docs/code-tour.md) | **bắt đầu từ đây khi đọc code**: lộ trình đọc + mô tả từng hàm |
| [docs/research-architecture.md](docs/research-architecture.md) | pipeline, phương pháp, các module, vấn đề đã biết |
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

## Ghi công và giấy phép

Xây dựng trên [DiffPure](https://github.com/NVlabs/DiffPure) (NVIDIA Source Code License,
[LICENSES/DiffPure_LICENSE](LICENSES/DiffPure_LICENSE); áp dụng cho các file có header của NVIDIA),
[Contrastive-Guided-Diffusion-Process](https://github.com/YiDongOuYang/Contrastive-Guided-Diffusion-Process.git)
và [stablediffusion](https://github.com/Stability-AI/stablediffusion.git) (các tham chiếu của bản
phát hành gốc). Code thượng nguồn đi kèm giữ giấy phép riêng:
[score_sde](https://github.com/yang-song/score_sde_pytorch),
[AutoAttack](https://github.com/fra31/auto-attack), BPDA+EOT và ReColorAdv/mister_ed (StAdv).
Classifier lấy từ [RobustBench](https://github.com/RobustBench/robustbench). Xem [LICENSE](LICENSE).
