# Phát triển trên nền Contrastive DiffPure

Trang này nói rõ thứ gì nên đặt ở đâu khi bạn mở rộng code nghiên cứu.

## Nguyên tắc nền

* **`src/` là code nghiên cứu.** Import bằng `from src.… import …` (chạy từ thư mục gốc của repo,
  hoặc `pip install -e .`).
* **`external/` và `src/attacks/` là code thượng nguồn.** `external/auto-attack` và
  `external/score_sde` được cài bằng pip thành package `autoattack` / `score_sde` và giữ nguyên như
  bản phát hành (AutoAttack có thêm `state_path`); `src/attacks/` (BPDA, StAdv) đã được rút gọn còn
  phần thực sự dùng. Không sửa chúng thêm. Nếu buộc phải sửa, hãy ghi lại trong file này, vì đó là thay đổi định nghĩa của
  attack hoặc của model.
* **Không dùng đường dẫn phụ thuộc thư mục làm việc.** Dùng `src.paths` (`data_path`,
  `pretrained_path`, `output_path`, `config_path`).
* **Mỗi dữ kiện chỉ có một nơi ở.** Dữ kiện về dataset nằm trong `src/data/specs.py`; diffusion
  model trong `configs/<dataset>.yml` (schema `src/config.py`); siêu tham số của phương pháp trong
  `configs/method/*.yaml` (schema `src/method.py`); thứ gì thay đổi theo từng lần chạy thì ở dòng
  lệnh. Đừng bao giờ lặp lại một dữ kiện ở hai nơi.
* **Đặt tên theo bài báo.** `guidance_strength` là lambda, `tau_minus` là tau^-. Mỗi trường
  `<f>` của `ContrastiveConfig` có đúng một flag CLI là `--contrastive_<f>`; không có alias.
* **Một thí nghiệm = một shell script trong `experiments/`.** Nó lưu đúng các flag của lần chạy đó.
  Thêm một dòng vào `experiments/README.md`.
* **Giữ baseline và phương pháp so sánh được với nhau.** Cùng một điểm vào, cùng `--seed`,
  `--data_seed` và `--num_sub` phải cho cùng tập ảnh và cùng nhiễu khuếch tán. Thứ tự nạp dữ liệu
  và dựng model trong `robustness_eval` là một phần của điều đó (xem comment ở đó).

## Thêm một phương pháp guidance mới

Phương pháp contrastive là một lớp bọc quanh score network:
`forward(x, t) -> model(x, t) + lượng hiệu chỉnh`.

1. Tạo `src/guidance/<tên>.py` với một `nn.Module` nhận score network và cung cấp `forward(x, t)`
   trả về đúng shape như của network.
2. Thêm schema config và `configs/method/<tên>.yaml`, theo mẫu `src/method.py`: khóa bắt buộc,
   kiểm tra trong `__post_init__`, và một hàm `resolve_<tên>(args)` cho phép dòng lệnh ghi đè file.
   Đặt các giá trị đã công bố vào YAML, không đặt vào default của argparse.
3. Thêm một builder cạnh `src/guidance/build.py`, thêm lựa chọn vào `GUIDANCE_CHOICES` trong
   `src/evaluation/robustness.py`, resolve config trong `setup()`, và bọc network trong
   `PurifiedClassifier.__init__` (`src/evaluation/purified_model.py`) đúng tại chỗ lớp bọc contrastive
   được gắn vào, bằng `runner.set_score_model(...)` (thay cả `runner.model` lẫn model của solver).
4. Nếu lớp bọc giữ trạng thái theo từng batch, hãy reset nó sau mỗi batch đã thanh lọc, ở cuối
   `PurifiedClassifier.forward`, cạnh `reset_drift_counter()`.
5. Thêm một test chạy CPU trong `tests/` với mạng toy, theo mẫu `tests/test_contrastive.py`.
6. Thêm script thí nghiệm trong `experiments/<tên>/`, sao từ script contrastive tương ứng và chỉ
   đổi các flag của guidance.

## Thêm một attack

Thêm `eval_<attack>(args, config, model, x_val, y_val, adv_batch_size, log_dir)` vào
`src/evaluation/attacks.py`, theo mẫu `eval_eotpgd`: tấn công classifier trần, rồi tấn công
purifier + classifier, và lưu `x_adv_*.pt`. Sau đó khai báo nó trong phần dispatch của
`robustness_eval` và trong danh sách `choices` của `--attack_version`.

## Thêm một dataset

1. Thêm một `DatasetSpec` vào `DATASETS` trong `src/data/specs.py` (tên, số lớp, kích thước ảnh,
   thư mục dữ liệu, file config) và một nhánh trong `build_test_set` của nó. Đó là tất cả những gì
   `--domain`, bộ nạp subset và plug-in contrastive cần.
2. Thêm config `configs/<dataset>.yml` với mọi khóa mà schema yêu cầu; `data.dataset` là khóa các
   runner so khớp để chọn checkpoint.
3. Thêm nhánh checkpoint trong `src/models/diffusion/sde.py` và `ode.py`, và classifier trong
   `src/models/classifiers/__init__.py`.
4. Huấn luyện các model còn thiếu bằng `scripts/train_score_sde.py` /
   `scripts/train_classifier.py` rồi chuyển đổi bằng `scripts/convert_lightning_checkpoint.py`.
5. Nếu tiền xử lý không thể là `Resize + ToTensor`, hãy ghi đè `eval_transform` trên spec và nói
   rõ lý do trong docstring: tiền xử lý lúc test là một phần của phương pháp.

## Trước khi báo cáo một con số

* Ghim môi trường (`pip freeze > outputs/<exp>/requirements.lock`).
* Kiểm tra `resolved_config.json` trong thư mục của lần chạy: đó là bản ghi hợp nhất của file
  config và dòng lệnh, còn `eval_indices.json` ghi lại những ảnh nào đã được chấm điểm.
* Chạy baseline và phương pháp với cùng bộ seed và data seed.
* Chạy lặp một cấu hình hai lần để đo mức biến động giữa các lần chạy.
* Xem lại các vấn đề còn mở trong `docs/research-architecture.md` §7.
