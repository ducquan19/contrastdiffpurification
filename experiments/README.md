# Thí nghiệm (CIFAR-10)

Mỗi script là một thí nghiệm từ repo gốc, với đúng cấu hình của nó. So với bản gốc chỉ có ba thay
đổi:
- điểm vào Python (`python -m scripts.eval_robustness|eval_bpda --guidance none|contrastive`);
- một lệnh `cd` về thư mục gốc của repo;
- kết quả giờ ghi vào `outputs/<--exp>/…` thay vì `src/<--exp>/…`.

```bash
bash experiments/<nhóm>/<script>.sh <gpu_id> "<các seed>" "<các data_seed>"
# ví dụ, cấu hình DiffPure khuyến nghị: 3 seed x 8 subset dữ liệu (mỗi subset 64 ảnh)
bash experiments/contrastive/run_cifar_rand_inf.sh 0 "121 122 123" "0 1 2 3 4 5 6 7"
```

Các script trong `ablation_t/` nhận thêm tham số thứ 4 là danh sách thời gian khuếch tán `t`,
ví dụ `"50 75 100 125"`.

## Danh mục

`t` là thời gian khuếch tán, `eps` là ngân sách của attack, `n` là số ảnh cho mỗi data seed.
Ở cột contrastive, "cls" là `--contrastive_classifier`, `T` là temperature (tau) và `lambda` là
guidance strength. Bất cứ thứ gì script không truyền sẽ lấy từ
[configs/method/contrastive.yaml](../configs/method/contrastive.yaml) và được ghi lại cho từng lần
chạy trong `resolved_config.json`.

| Script | Chương trình | Attack | Classifier | Diffusion | t | eps | n | Cấu hình contrastive |
|---|---|---|---|---|---|---|---|---|
| `baseline/run_cifar_rand_inf.sh` | eval_robustness, none | AutoAttack rand, Linf, EOT 20 | wideresnet-28-10 | sde | 100 | 8/255 | 64 | – |
| `baseline/run_cifar_rand_inf_70-16-dp.sh` | eval_robustness, none | AutoAttack rand, Linf | wrn-70-16-dropout | sde | 100 | 8/255 | 64 | – |
| `baseline/run_cifar_rand_inf_rn50.sh` | eval_robustness, none | AutoAttack rand, Linf | resnet-50 | sde | 125 | 8/255 | 64 | – |
| `baseline/run_cifar_rand_inf_ode.sh` | eval_robustness, none | AutoAttack rand, Linf | wideresnet-28-10 | ode | 100 | 8/255 | 64 | – |
| `baseline/run_cifar_rand_L2.sh` | eval_robustness, none | AutoAttack rand, L2 | wideresnet-28-10 | sde | 75 | 0.5 | 64 | – |
| `baseline/run_cifar_rand_L2_70-16-dp.sh` | eval_robustness, none | AutoAttack rand, L2 | wrn-70-16-dropout | sde | 75 | 0.5 | 64 | – |
| `baseline/run_cifar_rand_L2_rn50_eps1.sh` | eval_robustness, none | AutoAttack rand, L2 | resnet-50 | sde | 125 | 1.0 | 64 | – |
| `baseline/run_cifar_eotpgd.sh` | eval_robustness, none | EOT-PGD (10 bước, EOT 20) | wideresnet-28-10 | sde | 100 | 8/255 | 64 | – |
| `baseline/run_cifar_stadv_rn50.sh` | eval_robustness, none | StAdv | resnet-50 | sde | 125 | 0.05 | 64 | – |
| `baseline/run_cifar_bpda_eot.sh` | eval_bpda, none | BPDA+EOT | wideresnet-28-10 | sde | 100 | 8/255 | 200 | – |
| `contrastive/run_cifar_rand_inf.sh` | eval_robustness, contrastive | AutoAttack rand, Linf | wideresnet-28-10 | sde | 100 | 8/255 | 64 | cls wrn-28-10, drift_min 0, T 1.0, lambda 500 |
| `contrastive/run_cifar_rand_inf_70-16-dp.sh` | eval_robustness, contrastive | AutoAttack rand, Linf | wrn-70-16-dropout | sde | 100 | 8/255 | 64 | như trên |
| `contrastive/run_cifar_rand_inf_rn50.sh` | eval_robustness, contrastive | AutoAttack rand, Linf | resnet-50 | sde | 125 | 8/255 | 64 | như trên |
| `contrastive/run_cifar_rand_inf_ode.sh` | eval_robustness, contrastive | AutoAttack rand, Linf | wideresnet-28-10 | ode | 100 | 8/255 | 64 | như trên |
| `contrastive/run_cifar_rand_L2.sh` | eval_robustness, contrastive | AutoAttack rand, L2 | wideresnet-28-10 | sde | 75 | 0.5 | 64 | như trên |
| `contrastive/run_cifar_rand_L2_70-16-dp.sh` | eval_robustness, contrastive | AutoAttack rand, L2 | wrn-70-16-dropout | sde | 75 | 0.5 | 64 | như trên |
| `contrastive/run_cifar_rand_L2_rn50_eps1.sh` | eval_robustness, contrastive | AutoAttack rand, L2 | resnet-50 | sde | 125 | 1.0 | 64 | như trên |
| `contrastive/run_cifar_eotpgd.sh` | eval_robustness, contrastive | EOT-PGD | wideresnet-28-10 | sde | 100 | 8/255 | 64 | như trên |
| `contrastive/run_cifar_bpda_eot.sh` | eval_bpda, contrastive | BPDA+EOT | wideresnet-28-10 | sde | 100 | 8/255 | 200 | cls wrn-28-10, drift_min 40; T và lambda lấy từ method config (1.0 / 500) |
| `ablation_t/baseline_run_cifar_rand_inf.sh` | eval_robustness, none | AutoAttack rand, Linf | wideresnet-28-10 | sde | `$4` | 8/255 | 64 | – |
| `ablation_t/contrastive_run_cifar_rand_inf.sh` | eval_robustness, contrastive | AutoAttack rand, Linf | wideresnet-28-10 | sde | `$4` | 8/255 | 64 | cls wrn-28-10, drift_min **40**; phần còn lại lấy từ method config |

8/255 được viết là `0.031373` trong script. Tên classifier ở bảng đã viết ngắn; script dùng
`cifar10-wideresnet-28-10`, `cifar10-wrn-70-16-dropout` và `cifar10-resnet-50`.

## Vấn đề đã biết (xem `docs/research-architecture.md` §7)

* **`drift_min` không thống nhất giữa các script**: script AutoAttack / EOT-PGD dùng 0, còn
  `contrastive/run_cifar_bpda_eot.sh` và `ablation_t/contrastive_run_cifar_rand_inf.sh` truyền 40.
  Config chuẩn ghi 0. Các script được giữ nguyên đúng như bản công bố, vì chúng là hồ sơ thí
  nghiệm — bạn cần chốt dùng giá trị nào trước khi báo cáo số. Tham số này không có trong bài báo,
  nơi guidance được áp ở mọi bước reverse.
* **Script BPDA contrastive giờ chạy được** (trước đây crash, vấn đề B1) và dùng giá trị chuẩn
  tau = 1 / lambda = 500 thay cho default cũ của entry point là 10 / 5000.
* Repo gốc không có script StAdv cho nhánh contrastive.

## Huấn luyện (`training/`)

Các script này huấn luyện diffusion model và classifier cho CIFAR-100 / GTSRB. **Chúng không cần
thiết cho các thí nghiệm CIFAR-10 ở trên**, vì những thí nghiệm đó dùng checkpoint công khai
(xem `docs/reproduction.md`).
