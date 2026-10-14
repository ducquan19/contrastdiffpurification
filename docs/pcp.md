# Particle-Contrastive Purification (PCP) và đánh giá full-gradient

Trang này mô tả phần code mới: sampler K particle khả vi, guidance PCP, và attack PGD+EOT full-gradient.

## 1. Ý tưởng

ContrastDiffPure lấy negative từ các ảnh khác trong minibatch, nên kết quả thanh lọc của một ảnh phụ thuộc vào batch chứa nó (§7: A4). Theo DiffBreak, chế độ còn giữ được robustness trước gradient chính xác là **majority vote** trên nhiều bản thanh lọc, và robustness đó đến từ việc các bản thanh lọc ít tương quan với nhau.

PCP chuyển phần contrastive vào bên trong K particle của **cùng một ảnh**:

* **L_sem** (kéo, ngữ nghĩa): kiểu SwAV, với target là phân phối gán prototype của đồng thuận leave-one-out từ các particle khác. Prototype là k-means của feature ảnh **train** sạch, lưu trong một bank cố định chia theo time-bucket. Không có negative lấy từ batch.
* **L_rep** (đẩy, chi tiết): kernel RBF giữa các particle trong không gian pixel 16×16, độ rộng kernel chọn theo median heuristic (kiểu particle guidance).
* **Dự đoán**: majority vote trên K particle (`vote: hard`). Attack tối ưu trên soft vote (log của xác suất trung bình).

Cả hai loss được định nghĩa **theo từng ảnh**, nên guidance của ảnh i không phụ thuộc vào các ảnh khác (`test_purification_of_an_image_does_not_depend_on_the_batch`).

## 2. Code

| File | Vai trò |
|---|---|
| `src/models/diffusion/sampler.py` | `ParticleSampler`: vòng reverse tường minh (DDIM-η hoặc Euler–Maruyama), K particle, nhiễu theo seed của từng ảnh, `checkpoint='eps'` |
| `src/guidance/particle.py` | `ParticleContrastiveGuidance` và các gradient giải tích của L_sem, L_rep |
| `src/guidance/memory_bank.py` | `spherical_kmeans`, `PrototypeBank`, `build_prototype_bank` |
| `src/guidance/encoders.py` | encoder F đông cứng (`robustbench:<name>` hoặc tên classifier), lấy penultimate hoặc logits |
| `src/method_pcp.py`, `configs/method/pcp.yaml` | config của sampler và PCP, ghi đè bằng `--pcp_<field>` |
| `src/evaluation/particle_model.py` | `ParticlePurifiedClassifier` (`--diffusion_type particle`, `--guidance none/contrastive/pcp`) |
| `src/evaluation/voting.py` | soft/hard vote; `particle_agreement` (tương quan lỗi ρ, K_eff) |
| `src/evaluation/attacks_full.py` | `--attack_version pgd_full` |
| `scripts/build_prototype_bank.py`, `scripts/calibrate_throughput.py`, `scripts/summarize_pgd_full.py` | công cụ |
| `experiments/pcp/*.sh` | các thí nghiệm, chi phí ngang nhau giữa các phương pháp |

### Quy tắc không che gradient

Trong `sampler.py`, `particle.py`, `memory_bank.py`, `particle_model.py`, `attacks_full.py` và `voting.py`, mọi `detach` / `no_grad` phải có chú thích `# SAFE: <lý do>` trên cùng dòng (`tests/test_no_gradient_masking.py`). Các "stop-gradient" dùng để *định nghĩa* guidance (q và h được coi là hằng số khi lấy gradient của loss) được viết dưới dạng giải tích, không dùng `.detach()`. Nhờ vậy attacker vẫn vi phân được qua cả q và h (`test_full_gradient_through_the_guidance_is_exact`).

### Các baseline cùng chi phí

| Lệnh | Phương pháp |
|---|---|
| `--diffusion_type particle --guidance none --pcp_n_particles 1` | DiffPure, single purification |
| `... --guidance none --pcp_n_particles 8` | DiffPure-MV |
| `... --guidance contrastive --contrastive_differentiable true --contrastive_fix_cache true` | ContrastDiffPure (đã sửa, guidance hiện với attacker) |
| `... --guidance pcp` | PCP |

Tất cả dùng chung `--t`, `--pcp_steps`, `--pcp_solver`, `--pcp_eta`. Mặc định là S = 20 bước DDIM với η = 1. Cấu hình S = 100 Euler–Maruyama tương ứng với DiffPure gốc và chỉ dùng cho bảng xác nhận, vì quá tốn kém để chạy cho mọi ô.

## 3. Quy trình chạy

```bash
bash experiments/pcp/calibrate.sh 0            # throughput -> GPU-ngày ước tính
bash experiments/pcp/build_bank.sh 0           # bank prototype (ảnh train)
bash experiments/pcp/run_cifar_pgd_full_diffpure_sp.sh 0 121 0          # pilot: kiểm tra attack
# chia 512 ảnh cho 4 GPU, mỗi tiến trình một GPU:
for i in 0 1 2 3; do bash experiments/pcp/run_cifar_pgd_full_pcp.sh $i 121 0 $i/4 & done; wait
python -m scripts.summarize_pgd_full outputs/exp_results_pcp/cifar10-pgdfull-linf-pcp/cifar10-wideresnet-28-10/particle_pcp_pgd_full/seed121/data0
```

Mỗi batch được lưu vào `pgd_full_batch<NNN>.pt`, gồm ảnh adversarial, cờ "bị phá trong lúc attack" và dự đoán của từng particle. Với `--resume`, các batch đã xong được bỏ qua.

## 4. Kiểm tra bắt buộc trước khi báo cáo số

1. `pytest tests/` qua (CPU).
2. Pilot DiffPure-SP: robust accuracy dưới `pgd_full` phải cùng khoảng giá trị với Lee & Kim (46.84% khi backprop trực tiếp, ℓ∞ 8/255) và **thấp hơn rõ rệt** so với `--attack_version eotpgd` của bản phát hành (straight-through).
3. Sanity check: `--adv_eps 1` cho robust accuracy ≈ 0%. Tăng `--pgd_steps` hoặc `--pgd_eot_iter` không được làm robust accuracy tăng.
4. Báo cáo cả độ chính xác của chính encoder robust (`encoder` trong `pcp.yaml`) khi dùng một mình.
5. Kiểm tra lại tên model RobustBench trong `pcp.yaml` trước lần chạy đầu tiên.

## 5. Giới hạn hiện tại

* Chưa có: adapter DiffBreak/DiffHammer, các attack thích ứng riêng cho PCP (consensus-hijack, anti-diversity, encoder-feature), batch-coupling attack (P1).
* `--diffusion_type particle` chạy một GPU cho mỗi tiến trình (dùng `--shard`), không hỗ trợ `DataParallel`.
* BPDA (`mode='purify'`) chỉ hỗ trợ K = 1.
* Trong ContrastDiffPure, bước Adam online của F(.) được attacker coi là hằng số trong mỗi bước.
