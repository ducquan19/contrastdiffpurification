# Configs

Hai lớp, vì chúng trả lời hai câu hỏi khác nhau.

| Lớp | File | Câu hỏi nó trả lời | Schema |
|---|---|---|---|
| diffusion model | `cifar10.yml`, `cifar100.yml`, `gtsrb.yml` | score network nào và noise schedule nào, tức là checkpoint chứa gì | [`src/config.py`](../src/config.py) |
| phương pháp | `method/contrastive.yaml` | siêu tham số của contrastive guidance trong bài báo | [`src/method.py`](../src/method.py) |

Mọi thứ thay đổi giữa các lần chạy — dataset, classifier, attack, epsilon, `t`, số ảnh, seed —
đều nằm ở dòng lệnh và được ghi lại cho từng thí nghiệm trong `experiments/`. Mỗi lần chạy còn
ghi `resolved_config.json` cạnh log, chứa kết quả hợp nhất của file config và dòng lệnh: đó chính
là thứ đã tạo ra con số được báo cáo.

## Diffusion model

Mỗi dataset một file YAML, mô tả **diffusion model score-SDE (NCSN++)** và noise schedule dùng để
thanh lọc dataset đó. Chọn bằng `--config cifar10.yml` (đánh giá) hoặc
`--config_path configs/cifar10.yml` (huấn luyện).

| File | Được dùng bởi |
|---|---|
| `cifar10.yml` | mọi thí nghiệm CIFAR-10 trong `experiments/` |
| `cifar100.yml` | `scripts/train_score_sde.py` (diffusion model CIFAR-100) |
| `gtsrb.yml` | `scripts/train_score_sde.py` (diffusion model GTSRB) |

Ba file giống hệt nhau, chỉ khác `data.dataset` — đây là khóa mà các runner trong
`src/models/diffusion/` so khớp để chọn checkpoint.

## Các mục

| Mục | Điều khiển gì | Được đọc bởi |
|---|---|---|
| `data` | nạp checkpoint nào (`dataset`), kích thước đầu vào, và model có làm việc trên dữ liệu đã scale về `[-1, 1]` hay không (`centered`) | các runner, NCSN++, module huấn luyện |
| `model` | score network (kiến trúc NCSN++) và noise schedule của VP-SDE (`sigma_*`, `beta_*`, `num_scales`), cùng hệ số EMA dùng khi khôi phục trọng số | NCSN++, các runner, module huấn luyện |
| `training` | loại SDE và các tùy chọn loss; chỉ dùng khi huấn luyện diffusion model | `src/training/score_sde_module.py` |
| `optim` | optimizer và warmup; chỉ dùng khi huấn luyện | `score_sde/losses.py` |
| `sampling` | sampler predictor-corrector dùng bởi `train_score_sde.py --mode generate`; **không** dùng cho thanh lọc, vì thanh lọc tích phân trực tiếp reverse SDE/ODE | `score_sde/sampling.py` |

## Cơ chế kiểm tra

Mọi khóa trong schema đều **bắt buộc**. Khóa thiếu, sai chính tả hoặc không hợp lệ sẽ bị báo ngay
khi nạp config, kèm tên file, tên mục và tên khóa:

```
configs/cifar10.yml:
  - section 'model' missing key(s): nf
  - section 'data' unknown key(s): typo
```

Cách này thay cho hai loader trước đây: phần đánh giá đọc YAML thành namespace mà không kiểm tra
gì cả, còn phần huấn luyện dùng dataclass có default âm thầm điền vào bất cứ khóa nào file bỏ
trống (`training.likelihood_weighting` thiếu trong mọi config và lấy từ chính một default như vậy;
giờ nó đã được ghi ra tường minh).

Các khóa không có code nào đọc đã bị xóa, nên những gì còn lại đúng là những gì có tác dụng.
Đã xóa: `data.category`, `data.random_flip`, `data.uniform_dequantization`,
`model.attention_type`, `model.conv_size`, `training.n_iters`. Hai khóa trong config score_sde
thượng nguồn cũng không có mặt, vì chỉ các nhánh code không dùng mới đọc chúng:
`data.channels` và `model.num_classes` (NCSNv2 và nhánh `InstanceNorm++`; các model ở đây là
NCSN++ với GroupNorm).

## Phương pháp: `method/contrastive.yaml`

Các siêu tham số mà Algorithm 1 của bài báo yêu cầu, kèm đối chiếu với ký hiệu trong bài:

| Bài báo | Khóa | Flag CLI | Giá trị |
|---|---|---|---|
| lambda | `guidance_strength` | `--contrastive_guidance_strength` | 500 |
| tau | `temperature` | `--contrastive_temperature` | 1.0 |
| tau^- | `tau_minus` | `--contrastive_tau_minus` | 0.1 |
| beta | `beta` | `--contrastive_beta` | 1.0 |
| Eq. (10) so với Eq. (9) | `estimator` | `--contrastive_estimator` | hard |
| huấn luyện F(.) | `embed_lr` | `--contrastive_embed_lr` | 0.01 |
| – (không có trong bài báo) | `drift_min`, `drift_max` | `--contrastive_drift_*` | 0, 100000 |

Bài báo nêu tên các siêu tham số này nhưng **không cho giá trị số nào**, nên các giá trị ở trên
lấy từ script thí nghiệm CIFAR-10 của bản phát hành gốc — nguồn duy nhất có chúng.

Không nằm trong file này, vì không phải thuộc tính của phương pháp:

* backbone mà F(.) embed — `--contrastive_classifier`, phụ thuộc dataset;
* `n_classes`, lấy từ dataset spec trừ khi truyền `--contrastive_n_classes`;
* `t` (mức nhiễu forward `t*`), thay đổi theo threat model: 100 cho Linf 8/255, 125 cho ResNet-50,
  75 cho L2 0.5 (Bảng 7 của bài báo).

Ghi đè bất kỳ khóa `<f>` nào từ dòng lệnh bằng `--contrastive_<f>` khi cần quét tham số:

```bash
python -m scripts.eval_robustness --guidance contrastive \
    --contrastive_guidance_strength 1000 --contrastive_temperature 0.5 ...
```

Lưu ý rằng `--adv_batch_size` ở đây cũng là một siêu tham số của phương pháp, không chỉ là thiết
lập hạ tầng: các negative pair của contrastive loss chính là những sample khác trong cùng minibatch.

## Những gì *không* nằm ở đây

Các dữ kiện về dataset — số lớp, kích thước ảnh, thư mục dữ liệu — nằm trong
[`src/data/specs.py`](../src/data/specs.py).
