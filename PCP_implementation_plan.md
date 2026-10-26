# Kế hoạch triển khai: Particle-Contrastive Purification (PCP) trên nền mã ContrastDiffPure

Phiên bản 1, tháng 10/2026. Mã nền là repo `tenghuilee/ContrastDiffPurification` (nhánh `main`). Mọi số dòng nêu dưới đây là số dòng trong repo công khai. Nếu bản bạn đang có khác bản này, hãy đối chiếu lại.

---

## 0. Kết quả rà soát mã gốc (phải biết trước khi viết dòng code nào)

Tôi đã đọc các file `src/contrastive_guided_diffusion.py`, `src/eval_sde_adv_contrastive.py`, `src/eval_sde_adv.py`, `src/runners/diffpure_sde.py` và script `run_scripts_contrastive/cifar10/run_cifar_eotpgd.sh`. Có sáu điểm ảnh hưởng trực tiếp tới thiết kế và tới tính công bằng của baseline.

| # | Phát hiện | Vị trí | Hệ quả |
|---|---|---|---|
| A1 | Toàn bộ quá trình thanh lọc nằm trong `with torch.no_grad()`. Đầu ra được ghép lại thành `classifier(input_x + _w)` với `_w = x_re - input_x`, tức gradient của attacker qua purifier là **ma trận đơn vị** (BPDA-identity). | `eval_sde_adv_contrastive.py` dòng 123–156; baseline `eval_sde_adv.py` dòng 80–119 cũng vậy | Các attack "eotpgd" và AutoAttack trong repo **không phải full-gradient**. Mọi con số tái lập bằng mã này đều là đánh giá yếu. Phải viết wrapper mới. |
| A2 | `torchattacks.EOTPGD(..., steps=10, alpha=2/255, eot_iter=20)`; script dùng `--num_sub 64` và `--adv_batch_size 64` | `eval_sde_adv.py` dòng 275–300; file `.sh` | Chỉ 10 bước PGD trên 64 ảnh. Lee & Kim dùng 200 bước; cần ít nhất 512 ảnh. |
| A3 | Guidance dùng **logits của chính classifier bị tấn công** (`--contrastiveive_classifier cifar10-wideresnet-28-10`), đưa qua một MLP `embed_net` được **khởi tạo lại ngẫu nhiên và huấn luyện online bằng Adam** ở mỗi bước reverse | `contrastive_guided_diffusion.py`: `acquisition_grad`, `reset_drift_counter` | Encoder không robust, nên guidance có thể bị chiếm quyền (ý tưởng 3). Hướng guidance còn có thêm tính ngẫu nhiên do MLP ngẫu nhiên. Chính tính ngẫu nhiên này có thể trông giống robustness. |
| A4 | Positive pair là trạng thái x ở bước trước (`xt_cache`); negative là **các ảnh khác trong batch** (`negative_mask`) | `ContrastiveLoss.forward` | Xác nhận phụ thuộc batch, nền tảng cho thí nghiệm batch-coupling (P1). |
| A5 | `reset_drift_counter()` khởi tạo lại MLP và optimizer nhưng **không xoá `xt_cache`**. Bước guidance đầu tiên của lần gọi mới vì thế dùng trạng thái cuối của **lần gọi trước** làm positive (cùng shape nên không bị reset) | `contrastive_guided_diffusion.py`: `forward`, `reset_drift_counter` | Có rò rỉ trạng thái giữa các lần gọi: giữa các vòng EOT, giữa các bước attack, và giữa các batch. Đây là bug cần sửa khi dùng làm baseline; nên báo cáo cả bản gốc lẫn bản đã sửa. |
| A6 | `acquisition_grad` gọi `.detach()` trên x và x0 rồi trả về `delta.grad.detach() * delta_grad_mul` (mul = 500 trong script) | như trên | Dù có bỏ `no_grad` ở A1, attacker vẫn **không nhìn thấy đường gradient qua guidance**. Đây đúng là lỗ hổng mà DiffBreak chỉ ra. |

**Kết luận:** pha P0 (đánh giá đúng) là điều kiện tiên quyết. Bản thân các phát hiện A1–A6 cũng là nội dung có thể đưa vào bài (mục "Re-evaluating ContrastDiffPure").

Một số thông tin hữu ích lấy từ mã:
- **Score model CIFAR-10:** `score_type=score_sde` (VP-SDE liên tục, β_min = 0.1, β_max = 20).
- **ε được tính** qua `mutils.get_score_fn(..., continuous=True)` (`diffpure_sde.py` dòng 114–120).
- **Forward:** `x = √ā·x0 + √(1−ā)·e` với ā rời rạc tại `t = 100` (dòng 238–239).
- **Reverse:** `torchsde.sdeint_adjoint(..., method='euler')` từ `1 − t/1000` đến `1 − 1e-5` (dòng 244–254). Với `dt` mặc định, quá trình này khoảng 100 bước Euler.

---

## 1. Tổng quan phương pháp cần cài đặt

Đầu vào là x_a ∈ [0,1]^{B×3×H×W}. Đầu ra là nhãn bỏ phiếu đa số trên K particle.

1. **Nhân bản:** mỗi ảnh được nhân K lần, x ∈ [−1,1]^{(B·K)×3×H×W}. Nhiễu của từng particle được sinh **độc lập theo từng mẫu**.
2. **Forward:** x_{t*} = √ā_{t*}·x + √(1−ā_{t*})·ε. Tuỳ chọn lấy t* ngẫu nhiên trong một khoảng.
3. **Reverse:** S bước (DDPM/DDIM-η respaced, hoặc Euler–Maruyama VP-SDE). Ở mỗi bước:
   - ε = ε_θ(x_t, t)
   - x̂₀ = (x_t − √(1−ā_t)·ε)/√ā_t (Tweedie)
   - g = ∇ [λ_s(t)·L_sem + λ_r(t)·L_rep], tính tại x̂₀ rồi chia cho √ā_t
   - ε̃ = ε + √(1−ā_t)·clip(g)
   - x_{t−1} = step(x_t, ε̃)
4. **Phân loại:** f(x̂^{(k)}) cho k = 1..K. Lấy majority vote (hard) khi đánh giá, và soft vote log(mean softmax) cho loss của attack.

Quy ước dấu: cộng √(1−ā)·∇L vào ε tương đương trừ ∇L khỏi score, nên quá trình reverse đi theo hướng **giảm** L. Đây cũng là quy ước của classifier guidance và của ContrastDiffPure (mã gốc cộng `delta` vào đầu ra ε).

**L_sem: prototype-contrastive, không phụ thuộc batch.** Làm theo kiểu "swapped prediction" của SwAV, nên không cần negative trong batch và không có false negative.
- z_k = normalize(h(F(x̂₀^{(k)}))) ∈ ℝ^D, trong đó F là encoder đông cứng và h là projection (identity hoặc head huấn luyện offline).
- P ∈ ℝ^{M×D} là memory bank prototype sạch (k-means, chuẩn hoá L2), có thể chia theo khoảng thời gian.
- Logits ℓ_k = z_k Pᵀ/τ_s. Target q_k = softmax(c_{−k} Pᵀ/τ_t), với c_{−k} = normalize(Σ_{j≠k} z_j) (đồng thuận leave-one-out) và τ_t < τ_s để làm nét phân phối.
- L_sem = −(1/BK) Σ_{b,k} Σ_m q_{k,m} log softmax(ℓ_k)_m

Phân phối q đóng vai positive (các prototype mà nhóm particle cùng đồng ý), các prototype còn lại là negative. Guidance kéo mọi particle về cùng một mode sạch.

**L_rep: lực đẩy kiểu particle guidance.**
- φ(x̂₀) là ảnh hạ phân giải xuống 16×16 (hoặc feature tầng thấp của F).
- D²_{kj} = ‖φ_k − φ_j‖²; độ rộng kernel h = median(D²)/log(K+1).
- L_rep = (1/(B·K(K−1))) Σ_b Σ_{k≠j} exp(−D²_{kj}/h)

Giảm L_rep sẽ đẩy các particle ra xa nhau về chi tiết, tức giảm tương quan giữa chúng.

---

## 2. Cấu trúc thư mục đề xuất (thêm mới, không sửa phá mã gốc)

```
ContrastDiffPurification/
├── src/                          # mã gốc giữ nguyên (chỉ thêm cờ, xem P0)
└── pcp/
    ├── __init__.py
    ├── configs/
    │   ├── base_cifar10.yaml
    │   ├── pcp_cifar10.yaml
    │   ├── baselines/{diffpure,contrastdiffpure,gdmp}.yaml
    │   └── attacks/{pgd_eot_linf,pgd_eot_l2,apgd_rand,diffbreak_mv,bpda,batch_coupling}.yaml
    ├── core/
    │   ├── schedules.py          # ā(t) liên tục + rời rạc, respacing S bước, η
    │   ├── score_wrapper.py      # ε_θ(x,t) thống nhất (score_sde / guided_diffusion), dùng lại get_score_fn
    │   ├── sampler.py            # vòng reverse khả vi (thay sdeint_adjoint), checkpoint theo bước
    │   └── rng.py                # bộ sinh nhiễu độc lập theo mẫu (per-sample Generator)
    ├── guidance/
    │   ├── encoders.py           # F: logits gốc / penultimate WRN / robust AT / robust SSL
    │   ├── memory_bank.py        # xây k-means prototype, theo time-bucket, lưu .pt
    │   ├── losses.py             # L_sem (prototype), L_sem_infonce (ablation), L_rep
    │   └── guidance.py           # tính g, chuẩn hoá/clip, lịch λ(t), chế độ defense/attack
    ├── purifier.py               # PCPurifier (stateless), DiffPurePurifier, ContrastDiffPureAdapter
    ├── voting.py                 # hard/soft vote, logits cho attack
    ├── attacks/
    │   ├── pgd_eot.py            # PGD+EOT full-gradient (ℓ∞, ℓ2), loss trên soft-vote
    │   ├── apgd_wrap.py          # AutoAttack-rand / APGD-EOT
    │   ├── bpda.py               # BPDA+EOT (tham khảo)
    │   ├── adaptive_pcp.py       # consensus-hijack, anti-diversity, encoder-feature attack
    │   ├── batch_coupling.py     # tấn công qua ảnh đồng hành (cho phương pháp phụ thuộc batch)
    │   └── external/             # adapter DiffBreak, DiffHammer
    ├── metrics.py                # acc, Wilson CI, bootstrap, tương quan lỗi giữa particle, LPIPS
    ├── run_eval.py               # CLI chính: --config, --shard i/n, --resume
    ├── build_bank.py             # CLI xây memory bank
    ├── calibrate.py              # đo throughput NFE/s, ước lượng GPU-ngày
    ├── analysis/                 # notebook/script vẽ hình, tổng hợp jsonl
    └── tests/                    # pytest (mục 6)
```

---

## 3. Đặc tả từng module (giao diện và điểm cần chú ý)

### 3.1 `core/schedules.py`
- `alpha_bar_cont(t)`: `exp(−0.5(β₁−β₀)t² − β₀t)`, khớp `diffpure_sde.py` dòng 76.
- `make_timesteps(t_star, S, kind={'uniform','quadratic'})` trả về dãy t giảm dần từ t* về t_min = 1e-5.
- `ddim_coeffs(ab_t, ab_s, eta)` trả về (c_x0, c_eps, σ) với σ² = η²·(1−ā_s)/(1−ā_t)·(1−ā_t/ā_s). η = 1 xấp xỉ DDPM ancestral, η = 0 là DDIM tất định.

### 3.2 `core/score_wrapper.py`
- `class EpsModel(nn.Module): forward(x, t_cont) -> eps`. Bọc `mutils.get_score_fn(sde, model, train=False, continuous=True)` rồi đổi score về ε bằng ε = −score·√(1−ā_t), để **dùng đúng cách scale nhãn thời gian của mã gốc**.
- Kiểm tra: với K = 1 và λ = 0, kết quả phải khớp `RevVPSDE.f` về mặt số học (test T1).

### 3.3 `core/sampler.py`: vòng reverse khả vi
```python
def reverse(x_tstar, eps_model, guidance, ts, eta, rng, ckpt=True, attack_mode=False):
    x = x_tstar
    for t, s in zip(ts[:-1], ts[1:]):                      # t > s
        def one_step(x):
            ab_t, ab_s = alpha_bar(t), alpha_bar(s)
            eps = eps_model(x, t)
            x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            g = guidance(x0_hat, t, create_graph=attack_mode) / ab_t.sqrt()  # ∇_{x_t} xấp xỉ
            eps_t = eps + (1 - ab_t).sqrt() * g
            x0_t = (x - (1 - ab_t).sqrt() * eps_t) / ab_t.sqrt()
            c0, ce, sig = ddim_coeffs(ab_t, ab_s, eta)
            return c0 * x0_t + ce * eps_t + sig * rng.randn_like(x)   # rng per-sample
        x = checkpoint(one_step, x, use_reentrant=False) if ckpt else one_step(x)
    return x
```
- Bỏ `sdeint_adjoint` và dùng backprop trực tiếp kèm checkpoint theo từng bước, vì Lee & Kim cho thấy cách này tạo attack mạnh hơn adjoint.
- **Lưu ý khi dùng checkpoint với nhiễu ngẫu nhiên:** trong pha recompute, nhiễu phải được tái tạo đúng như lần chạy đầu. Hãy sinh trước toàn bộ nhiễu của các bước (tensor [S, B·K, C, H, W], hoặc seed theo từng bước) rồi truyền vào hàm bước, thay vì gọi `randn` bên trong.
- Hỗ trợ `S ∈ {10, 20, 50, 100}`. Với S = 100 và η = 1, sampler gần với DiffPure gốc.

### 3.4 `core/rng.py`
- `PerSampleRNG(seeds: LongTensor[B], K)` sinh nhiễu cho từng (ảnh, particle, bước) từ seed riêng.
- Khi deploy hoặc đánh giá: seed mới, ngẫu nhiên ở mỗi lần gọi; attacker không biết seed.
- Khi test: seed cố định để kiểm tra tính độc lập với batch (T2, T3).

### 3.5 `guidance/encoders.py`
| Tên | Mô tả | Mục đích |
|---|---|---|
| `logits_target` | logits của classifier bị tấn công (như mã gốc) | ablation tái hiện A3 |
| `penult_target` | feature penultimate của WRN-28-10 chuẩn | ablation |
| `robust_sup` | penultimate của một mô hình adversarially-trained (RobustBench, ví dụ ResNet-18 ℓ∞) | **mặc định đề xuất** |
| `robust_ssl` | encoder tự giám sát robust (ACL/RoCL/DeACL nếu có checkpoint) | phiên bản "không nhãn" |

- Đầu vào của F là ảnh trong [0,1], đã chuẩn hoá theo đúng encoder. F để `eval()`, `requires_grad_(False)` cho tham số, nhưng **vẫn giữ graph theo đầu vào**.
- **Báo cáo bắt buộc:** độ chính xác của chính mô hình `robust_sup` khi dùng một mình, để reviewer thấy PCP không chỉ "mượn" robustness của AT.

### 3.6 `guidance/memory_bank.py` và `build_bank.py`
- Lấy N_b ảnh train sạch (ví dụ 20k), tính z = normalize(h(F(·))), chạy k-means với M ∈ {100, 500, 2000} (faiss-gpu hoặc `torch` k-means), rồi lưu `bank_{encoder}_{M}.pt`.
- **Bank theo thời gian (khuyến nghị):** ở mỗi bước, x̂₀ ở t lớn bị mờ, nên feature lệch so với ảnh sạch. Với mỗi time-bucket t_j (ví dụ 5 mốc trong [0, t*]), làm nhiễu ảnh sạch tới t_j, tính x̂₀ bằng ε_θ, rồi k-means trên feature của các x̂₀ đó. Khi chạy, dùng bank của bucket gần nhất.
- Bank được **cố định và không phụ thuộc test batch**. Ghi lại hash của file trong log.

### 3.7 `guidance/losses.py`
```python
def sem_prototype_grad_target(z, P, K, tau_s, tau_t):
    # z: [B,K,D] chuẩn hoá; P: [M,D]
    c = F.normalize(z.sum(1, keepdim=True) - z, dim=-1)        # leave-one-out consensus
    q = torch.softmax(c @ P.T / tau_t, -1)                     # [B,K,M]  GIỮ GRAPH (xem 3.8)
    p = torch.softmax(z @ P.T / tau_s, -1)
    # dL/dz (q coi là hằng số trong ĐỊNH NGHĨA của guidance):
    g_logits = (p - q) / (z.shape[0] * K)                      # [B,K,M]
    return (g_logits @ P) / tau_s                              # [B,K,D] = ∂L_sem/∂z

def rep_loss(x0, K, size=16):
    phi = F.interpolate(x0, size=size, mode='area').flatten(1).view(-1, K, 3*size*size)
    d2 = torch.cdist(phi, phi).pow(2)                          # [B,K,K]
    off = ~torch.eye(K, dtype=torch.bool, device=x0.device)
    h = d2[:, off].median(dim=-1).values.clamp_min(1e-8) / math.log(K + 1)
    kmat = torch.exp(-d2 / h[:, None, None])
    return (kmat * off).sum((1, 2)).mean() / (K * (K - 1))
```
Ablation cần có: `sem_infonce_inbatch` (tái hiện cách lấy negative của ContrastDiffPure) và `sem_infonce_particles` (positive là particle khác, negative là prototype).

### 3.8 `guidance/guidance.py`: quy tắc "không che gradient"
- **Chế độ defense:** g = ∂L/∂x̂₀ được tính bằng autograd **theo x̂₀** chứ không theo x_t, nên không phải backprop qua ε_θ và chi phí rẻ.
  - Đường ngữ nghĩa: `g_x0 = torch.autograd.grad(z, x0_hat, grad_outputs=dLdz, create_graph=attack_mode)`, với `dLdz` lấy từ `sem_prototype_grad_target`.
  - Đường đẩy: `torch.autograd.grad(rep_loss(x0_hat), x0_hat, create_graph=attack_mode)`.
- **Chế độ attack (`attack_mode=True`):** giá trị số giống hệt chế độ defense, nhưng **giữ toàn bộ graph**, để attacker vi phân được qua mọi thứ: q (consensus), độ rộng kernel h, x̂₀ (qua ε_θ) và encoder F (double-backward).
  - Quy tắc code: **cấm `.detach()` hoặc `torch.no_grad()` trên đường dữ liệu** trong `guidance.py`, `sampler.py` và `purifier.py`. Chỗ nào cần "stop-grad" trong định nghĩa guidance thì viết gradient giải tích như trên, không dùng detach.
  - Thêm một linter đơn giản trong test: grep `detach(`/`no_grad` trong các file đó, mỗi lần xuất hiện phải có chú thích `# SAFE: not on data path`.
- **Chuẩn hoá theo mẫu:** g ← γ(t)·g·‖ε‖/(‖g‖ + 1e-8) cho từng particle, với γ = γ_s hoặc γ_r. Cách này thay hằng số `delta_grad_mul = 500` thiếu ổn định của mã gốc. Đặt `clip_ratio` tối đa 0.5.
- **Lịch λ(t):** L_rep mạnh ở nửa đầu reverse (t gần t*) và tắt khi t < t_r; L_sem tăng dần, tắt khi t < t_s_min (ví dụ 0.01). Mặc định chọn tuyến tính theo t/t*.

### 3.9 `purifier.py`
```python
class PCPurifier(nn.Module):
    def __init__(self, eps_model, encoder, bank, cfg): ...
    def forward(self, x01, seeds=None, attack_mode=False):     # x01: [B,3,H,W] ∈ [0,1]
        x = (x01 * 2 - 1).repeat_interleave(self.K, 0)          # [B*K,...]
        t_star = self.sample_tstar()                             # cố định hoặc ngẫu nhiên
        rng = PerSampleRNG(seeds or fresh_seeds(B), self.K)
        x_t = ab(t_star).sqrt() * x + (1 - ab(t_star)).sqrt() * rng.randn_like(x)
        x0 = reverse(x_t, self.eps_model, self.guidance, self.ts(t_star), self.eta, rng,
                     ckpt=self.ckpt, attack_mode=attack_mode)
        return ((x0.clamp(-1, 1) + 1) / 2).view(B, self.K, *x01.shape[1:])
```
- **Không có trạng thái** giữa các lần gọi: không cache, không optimizer online. Đây là khác biệt cốt lõi so với A3–A5.
- `DiffPurePurifier` là cùng lớp với λ_s = λ_r = 0. Dùng cho baseline DiffPure-SP (K=1) và DiffPure-MV (K>1), bảo đảm so sánh **cùng sampler, cùng S, cùng số NFE**.
- `ContrastDiffPureAdapter` bọc `ContrastiveGuidedDiffusion` gốc. Có cờ `fix_cache` (sửa A5) và `diff_guidance` (bỏ detach ở A6, tính lại `delta` bằng `autograd.grad(create_graph=True)`). Bước Adam online của `embed_net` được coi là hằng số trong mỗi bước; ghi rõ đây là giới hạn của attack.

### 3.10 `voting.py`
- `hard_vote(logits[B,K,C])`: argmax theo số phiếu, hoà thì phá bằng tổng softmax.
- `soft_vote_logits = log(mean_k softmax(logits))`. Đây là đầu ra để attack tối ưu (CE/DLR), giống chế độ majority-vote của DiffBreak.
- Lưu thêm `per_particle_pred[B,K]` vào kết quả để phân tích tương quan mà không phải chạy lại.

### 3.11 `attacks/`
- **`pgd_eot.py`:** PGD+EOT full-gradient theo Lee & Kim (200 vòng; EOT 20; step size và khởi tạo lấy đúng theo paper/code của họ). Loss: CE hoặc DLR trên `soft_vote_logits`. Mỗi lần forward của EOT gọi purifier với seed mới. Chế độ thay thế là "EOT ngân sách cố định": tổng số mẫu particle mỗi vòng E·K bằng hằng số. Phải báo cáo cả hai, và mặc định dùng E = 20 cho kết quả chính.
- **`apgd_wrap.py`:** APGD-CE/DLR phiên bản rand. Có thể lấy từ thư mục `auto-attack/` sẵn trong repo (`version='rand'`).
- **`adaptive_pcp.py`:** các attack thiết kế riêng cho PCP.
  1. **consensus-hijack:** cộng thêm vào loss số hạng KL(q ‖ e_{wrong}), đẩy đồng thuận của particle về prototype của lớp sai.
  2. **anti-diversity:** cộng −L_rep vào mục tiêu của attacker, để các particle sụp về cùng một mode sai.
  3. **encoder-feature attack:** cực đại hoá khoảng cách feature F(x_a) tới cụm prototype đúng.
  4. **λ-aware:** attack với t* và lịch λ được attacker biết trước.
- **`batch_coupling.py`** (dùng cho P1): biến tối ưu là r ảnh "đồng hành" trong batch m ảnh; ảnh nạn nhân giữ sạch. Mục tiêu là CE của nạn nhân trên bản `ContrastDiffPureAdapter(diff_guidance=True)`. Có hai biến thể: ảnh đồng hành không bị giới hạn (unbounded) và có ràng buộc ℓ∞ quanh một ảnh tự nhiên.
- **`external/`:** adapter cho DiffBreak (SP/MV, gradient qua guidance, LF) và DiffHammer. Chỉ cần đưa purifier về dạng `nn.Module` nhận [0,1] và trả logits. Cần kiểm tra license và API của từng repo.

### 3.12 `metrics.py`
- **Độ chính xác:** clean và robust, kèm khoảng tin cậy Wilson 95% và bootstrap qua 3 seed.
- **Tương quan lỗi giữa particle (cho H2):** với chỉ báo đúng/sai e_{b,k}, tính ρ̄ = trung bình các tương quan Pearson theo cặp (k, j) trên tập ảnh. Thêm "số phiếu hiệu dụng" K_eff = K/(1+(K−1)ρ̄) và biên phiếu (vote margin).
- **Độ đa dạng:** LPIPS trung bình giữa các particle, và PSNR/SSIM của từng particle so với ảnh sạch.
- **Chi phí:** NFE/ảnh, giây/ảnh, bộ nhớ đỉnh.

### 3.13 `run_eval.py`: hạ tầng chạy dài
- Cấu hình bằng YAML. Đối số `--shard i/n` để chia 512 ảnh thành n phần chạy song song trên nhiều GPU, mỗi phần là một tiến trình riêng (đơn giản hơn DDP).
- Lưu theo từng batch: `adv_x.pt`, `per_particle_logits.pt`, `meta.json` (seed, cấu hình, commit hash). `--resume` bỏ qua các batch đã xong.
- Log dạng jsonl; script `analysis/aggregate.py` gom thành bảng LaTeX/CSV.
- Chỉ số ảnh cố định: dùng cùng `data_seed` như DiffPure để chọn 512 ảnh test. Tập validation để tune là **256 ảnh lấy từ tập train** (đã giữ riêng), không bao giờ dùng ảnh test để tune.

---

## 4. Thí nghiệm theo pha (kèm tiêu chí đạt)

### P0: Hạ tầng và tái lập baseline đúng cách (tuần 1–2)
1. Môi trường: Python 3.10, PyTorch ≥ 2.1, CUDA 11.8/12.x; `robustbench`, `lpips`, `faiss-gpu` (tuỳ chọn), `pytest`. Tải checkpoint score_sde CIFAR-10 và WRN-28-10/70-16 như README gốc.
2. Thêm vào mã gốc cờ `--grad_mode {bpda_identity, full}` (sửa A1) để có thể chạy cả hai cách và chứng minh khác biệt.
3. Viết `core/` và `DiffPurePurifier`. Chạy test T1 (khớp với `RevVPSDE`).
4. **`calibrate.py`:** đo r = số ảnh·NFE/giây (forward, và forward+backward có checkpoint) trên A100 và V100, với S ∈ {20, 100} và batch tối đa vừa bộ nhớ. In ra bảng GPU-ngày cho từng cấu hình ở mục 5.
5. **Pilot:** DiffPure-SP với 128 ảnh val, PGD+EOT full-gradient (S = 100, nếu ngân sách cho phép). **Tiêu chí đạt:** robust accuracy cùng khoảng giá trị với Lee & Kim (46.84% với backprop trực tiếp, ℓ∞ 8/255) và thấp hơn rõ rệt so với chế độ `bpda_identity`. Nếu không đạt, gỡ lỗi attack trước khi đi tiếp.

### P1: Phân tích ContrastDiffPure gốc (tuần 3–4), cho hình 1 và mục "động lực" của bài
1. **Rò rỉ trạng thái (A5):** thanh lọc cùng một batch hai lần sau hai batch "tiền nhiệm" khác nhau, với cùng seed nhiễu. Đo ‖Δx̂‖ và tỉ lệ đổi nhãn, so sánh trước và sau khi sửa.
2. **Độ nhạy theo thành phần batch:** giữ cố định 16 ảnh nạn nhân, thay 48 ảnh còn lại theo các kịch bản: cùng lớp, trộn lớp, một lớp duy nhất khác, ảnh OOD (SVHN), và ảnh nhiễu. Đo độ chính xác sạch của nạn nhân. Thí nghiệm này rẻ và cho kết quả nhanh.
3. **Batch-coupling attack:** với r/m ∈ {1/64, 4/64, 16/64, 32/64}, đo độ chính xác của nạn nhân sạch và nạn nhân đã bị PGD nhẹ.
4. **Đánh giá lại ContrastDiffPure** dưới `grad_mode=full`, có và không có `diff_guidance`.
5. **Tiêu chí:** nếu (2) hoặc (3) làm độ chính xác của nạn nhân giảm đáng kể (ví dụ > 5 điểm), đây là một "finding" đủ mạnh để mở đầu bài. Nếu không, chỉ giữ A1–A6 làm phần phụ.

### P2: Cài đặt PCP (tuần 4–7)
- Thứ tự viết: `losses.py`, `memory_bank.py` + `build_bank.py`, `guidance.py`, `purifier.py`, `voting.py`, `adaptive_pcp.py`.
- **Tiêu chí:** toàn bộ test T1–T9 ở mục 6 đều qua. Với dữ liệu sạch, PCP (K = 8) có clean accuracy ≥ DiffPure-MV (K = 8) trừ 1 điểm.

### P3: Tune trên tập val với attack rẻ (tuần 8–9)
- **Attack rẻ:** PGD+EOT full-gradient 50 vòng, EOT 4, S = 20, ℓ∞ 8/255, trên 256 ảnh val.
- **Tìm kiếm:** K ∈ {4, 8}; γ_s ∈ {0.05, 0.1, 0.2}; γ_r ∈ {0, 0.05, 0.1, 0.2}; τ_s ∈ {0.1, 0.2}, τ_t = τ_s/2; M ∈ {500}; encoder ∈ {penult_target, robust_sup}; t* ∈ {0.1}. Dùng random search khoảng 24 cấu hình, hoặc Optuna với ngân sách cố định.
- **Go/No-go:** PCP-MV phải hơn DiffPure-MV **ở cùng NFE** ít nhất khoảng 3 điểm robust accuracy trên val, và hơn ContrastDiffPure-MV (đã sửa).
  - Nếu chỉ thắng nhờ `robust_sup`: kiểm tra baseline "DiffPure-MV + encoder robust làm guidance kiểu GDMP" để tách đóng góp.
  - Nếu không thắng: chuyển sang phương án B (mục 8).

### P4: Đánh giá đầy đủ (tuần 10–14)
Đây là ma trận thí nghiệm chính. Mọi phương pháp dùng cùng S, cùng t*, cùng 512 ảnh test và 3 seed.

| Nhóm | Phương pháp | K |
|---|---|---|
| Không purifier | WRN-28-10 chuẩn; mô hình `robust_sup` dùng một mình; AT tốt nhất trên RobustBench (số báo cáo) | – |
| Single-purification | DiffPure, GDMP, ContrastDiffPure (gốc, đã sửa) | 1 |
| Majority-vote | DiffPure-MV, ContrastDiffPure-MV, GDMP-MV, (FreqPure-MV nếu có code) | 8 |
| Đề xuất | PCP | 8 (và 16 trong ablation) |

| Attack | Ghi chú |
|---|---|
| PGD+EOT full-gradient ℓ∞ 8/255 và ℓ2 0.5 | chính |
| APGD/AutoAttack-rand | bổ sung |
| DiffBreak (MV, gradient qua guidance, LF) | bắt buộc với reviewer A* |
| DiffHammer | bổ sung |
| Adaptive PCP (4 biến thể ở 3.11) | bắt buộc |
| BPDA+EOT | chỉ để so với số liệu cũ |
| Batch-coupling | cho các phương pháp phụ thuộc batch; PCP phải miễn nhiễm (H3) |

**Kết quả báo cáo:** lấy worst-case trên tất cả các attack cho từng phương pháp (đây là con số chính), kèm từng attack riêng lẻ, clean accuracy, NFE và thời gian.

### P5: Phân tích và ablation (tuần 14–16)
1. **H2:** vẽ robust accuracy theo ρ̄ và K_eff, gồm tất cả cấu hình của DiffPure-MV, ContrastDiffPure-MV và PCP. Đây là hình cơ chế của bài.
2. **Ablation:** K ∈ {1, 2, 4, 8, 16}; bỏ L_sem; bỏ L_rep; L_sem dùng InfoNCE in-batch thay cho prototype; encoder (4 loại); M ∈ {100, 500, 2000}; bank theo thời gian hay một bank; không gian đẩy (pixel hay feature); S ∈ {10, 20, 50, 100}; t* ∈ {0.075, 0.1, 0.125}; t* ngẫu nhiên.
3. **So sánh cùng NFE:** PCP với K particle và S bước, đặt cạnh DiffPure-SP với K·S bước và DiffPure-MV với cùng K và S.
4. **Thí nghiệm lý thuyết trên dữ liệu giả:** mô hình Gaussian mixture d = 1000 của bài gốc (mục 3), K particle Langevin có hoặc không có lực đẩy. Kiểm chứng xác suất vote sai giảm theo K và ρ. Phần này chạy bằng numpy trên CPU.

### P6: Mở rộng và viết bài (tuần 16–20)
- CIFAR-100 (WRN-28-10), và ImageNet 256 trên 512 ảnh với guided-diffusion, S = 20, chỉ dùng PGD+EOT. ImageNet tốn kém nên chỉ chạy cấu hình tốt nhất.
- Hình 1: thí nghiệm batch-coupling và rò rỉ. Hình 2: sơ đồ PCP. Hình 3: robust accuracy theo ρ̄. Bảng 1: worst-case. Bảng 2: ablation.

---

## 5. Ngân sách tính toán (phải đo lại ở P0)

Số NFE ảnh cho một cấu hình attack:

  NFE ≈ N_img × T_iter × E × K × S × c_bwd × c_g

- c_bwd ≈ 3 (forward + recompute do checkpoint + backward).
- c_g ≈ 1.5–2.5 (thêm encoder và double-backward của guidance khi `attack_mode`).
- GPU-ngày = NFE / (r × 86400), với r đo bằng `calibrate.py`.

**Ví dụ minh hoạ (giả định r = 2000 ảnh·NFE/s trên A100, phải đo lại):**

| Cấu hình | N | T | E | K | S | GPU-ngày (ước tính) |
|---|---|---|---|---|---|---|
| Lee & Kim đầy đủ, DiffPure-SP (c_g = 1) | 512 | 200 | 20 | 1 | 100 | ≈ 3.6 |
| PCP đầy đủ ở S = 100 (c_g = 2) | 512 | 200 | 20 | 8 | 100 | ≈ 57 (cho **một** cấu hình × một seed × một attack) |
| PCP, S = 20, E = 4 (E·K = 32) | 512 | 200 | 4 | 8 | 20 | ≈ 2.3 |
| DiffPure-MV, S = 20, E = 4 (c_g = 1) | 512 | 200 | 4 | 8 | 20 | ≈ 1.1 |
| Pilot val ở P3 (mỗi cấu hình) | 256 | 50 | 4 | 8 | 20 | ≈ 0.3 |

Toàn bộ ma trận P4 (khoảng 8 phương pháp × 3 seed × khoảng 5 attack) nhân các con số trên lên khoảng 100 lần. Vì vậy cấu hình S = 100 cho mọi ô là không thực tế.

**Hệ quả cho thiết kế:**
- (i) Thí nghiệm chính dùng **S = 20** cho *mọi* phương pháp. Đó là so sánh công bằng; ADBM và Lee & Kim cũng đã đánh giá với sampler ít bước.
- (ii) Chỉ chạy một bảng xác nhận ở S = 100 cho cấu hình tốt nhất trên 128–256 ảnh.
- (iii) Dùng "EOT ngân sách cố định" E·K ≈ 32 làm thiết lập phụ, nhưng phải chứng minh rằng tăng E không làm robust accuracy giảm thêm (sanity check S3).
- (iv) Chạy song song bằng `--shard` trên nhiều A100. V100 không có bf16/TF32, nên chạy fp32 và dành cho các job rẻ (P1, tune).
- (v) Không dùng AMP fp16 cho score model khi attack, vì dễ gây gradient sai hoặc NaN, trông giống obfuscation.

---

## 6. Kiểm thử bắt buộc (pytest, chạy trước mọi job lớn)

| ID | Nội dung | Tiêu chí |
|---|---|---|
| T1 | `DiffPurePurifier` (K=1, λ=0, S=100, η=1) so với `RevVPSDE` gốc với cùng nhiễu | sai khác ≤ 1e-4 (hoặc clean acc trong CI) |
| T2 | Hoán vị thứ tự ảnh trong batch, seed theo mẫu cố định | đầu ra hoán vị tương ứng, sai khác ≤ 1e-5 |
| T3 | Thanh lọc ảnh i một mình so với trong batch 64 ảnh | giống hệt (PCP); khác (ContrastDiffPure, đo mức chênh) |
| T4 | `gradcheck` (float64) cho đường `attack_mode` với mô hình tí hon, S = 2, encoder nhỏ | qua |
| T5 | Dấu guidance: sau một bước có guidance, L_sem và L_rep giảm so với không guidance (trung bình) | qua |
| T6 | Gradient có đi qua guidance: ∂out/∂x khi λ > 0 khác khi λ = 0, và khác 0 | qua |
| T7 | "Linter" detach/no_grad trên đường dữ liệu (3.8) | không vi phạm |
| T8 | Checkpoint và nhiễu: gradient khi có checkpoint bằng khi không | sai khác ≤ 1e-4 |
| T9 | Bộ nhớ: batch tối đa ở S = 20, K = 8, `attack_mode` trên A100-40GB/80GB | ghi lại |

**Sanity check của attack (chạy trên 64 ảnh, mỗi phương pháp):**
- S1: ε = 0 cho ra clean accuracy; ε = 1 (unbounded) cho ra ≈ 0%.
- S2: robust accuracy giảm (hoặc không tăng) khi số vòng tăng 50 → 100 → 200.
- S3: robust accuracy không tăng khi E tăng 4 → 8 → 20.
- S4: attack transfer (black-box từ WRN không có purifier) yếu hơn attack white-box.
- S5: nếu tắt guidance trong graph của attacker (dùng `.detach()`) thì attack yếu hơn. Điều này chứng minh đường gradient qua guidance có tác dụng.

---

## 7. Siêu tham số mặc định đề xuất (điểm khởi đầu cho P3)

| Tham số | Giá trị khởi đầu |
|---|---|
| t* | 0.1 (tương đương t = 100 trong mã gốc) |
| S, η | 20, 1.0 |
| K | 8 |
| Encoder F | `robust_sup` (ResNet-18 AT ℓ∞, penultimate 512-d), h = identity |
| Bank | M = 500, 5 time-bucket, 20k ảnh train |
| τ_s, τ_t | 0.1, 0.05 |
| γ_s, γ_r (tỉ lệ theo ‖ε‖) | 0.1, 0.1 |
| Lịch | L_rep bật khi t ∈ [0.5t*, t*]; L_sem bật khi t ∈ [0.01, t*] |
| φ cho L_rep | ảnh 16×16 (area) |
| clip_ratio | 0.5 |

---

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| PCP không hơn DiffPure-MV ở cùng NFE | P3 không đạt | Chuyển trọng tâm sang bài phân tích: A1–A6 + batch-coupling + đánh giá lại guided purification dưới MV. Hướng này vẫn đủ cho hội nghị A, hoặc ghép với ý tưởng 3 (lý thuyết encoder). |
| Lợi ích chỉ đến từ encoder robust | ablation encoder | Báo cáo trung thực; dùng `robust_ssl` để tránh câu hỏi "dùng nhãn"; so với "DiffPure-MV + guidance robust kiểu GDMP". |
| Gradient bị che (obfuscation) | S2/S3 thất bại, BPDA mạnh hơn full-gradient | Rà lại T6–T8; tăng E; thử attack trên x̂₀ trung gian (kiểu DiffAttack). |
| Không đủ GPU | `calibrate.py` | S = 10–20, giảm N xuống 256 cho ablation; giữ 512 cho bảng chính. |
| Lực đẩy làm giảm clean accuracy | P2 | Giảm γ_r, chỉ đẩy ở t lớn, đẩy trong không gian feature tầng thấp. |
| Batch-coupling attack không hiệu quả | P1 | Chuyển nó thành phần phụ; động lực của bài vẫn là "majority-vote cần particle ít tương quan". |

---

## 9. Danh sách việc cụ thể (checklist)

- [ ] P0.1 Tạo nhánh git `pcp`; thêm `pcp/` và `requirements_pcp.txt`
- [ ] P0.2 Cờ `--grad_mode` trong `eval_sde_adv*.py` (A1); thêm tuỳ chọn số bước PGD và `num_sub`
- [ ] P0.3 `core/schedules.py`, `score_wrapper.py`, `sampler.py`, `rng.py`; test T1, T8
- [ ] P0.4 `calibrate.py`; điền bảng mục 5 bằng số đo thật
- [ ] P0.5 `attacks/pgd_eot.py`; sanity S1–S3 trên DiffPure; pilot 128 ảnh
- [ ] P1.1 Sửa A5 bằng cờ `fix_cache`; thí nghiệm rò rỉ trạng thái
- [ ] P1.2 Thí nghiệm thành phần batch
- [ ] P1.3 `ContrastDiffPureAdapter(diff_guidance=True)` và `attacks/batch_coupling.py`
- [ ] P2.1 `encoders.py` (4 loại), `memory_bank.py`, `build_bank.py`
- [ ] P2.2 `losses.py`, `guidance.py` (đúng quy tắc 3.8), `purifier.py`, `voting.py`
- [ ] P2.3 Test T2–T7, T9
- [ ] P2.4 `attacks/adaptive_pcp.py`, adapter DiffBreak/DiffHammer
- [ ] P3 Tune trên val, quyết định go/no-go
- [ ] P4 Ma trận đánh giá đầy đủ, 3 seed, lưu per-particle logits
- [ ] P5 Phân tích ρ̄/K_eff, ablation, mô phỏng Gaussian mixture
- [ ] P6 CIFAR-100 / ImageNet, hình và bảng, bản thảo

## Tài liệu tham chiếu khi code
- Mã gốc: `src/contrastive_guided_diffusion.py`, `src/eval_sde_adv_contrastive.py`, `src/runners/diffpure_sde.py`
- Lee & Kim, ICCV 2023 (cấu hình PGD+EOT, surrogate process)
- DiffBreak (Kassis et al., arXiv 2411.16598): chế độ MV, gradient qua guidance, LF attack
- DiffHammer (NeurIPS 2024); DiffAttack (NeurIPS 2023)
- Particle Guidance (Corso et al., ICLR 2024): kernel đẩy giữa các particle
- SwAV (Caron et al., NeurIPS 2020): swapped prediction với prototype
- Distribution Invading Attack (Wu et al., ICML 2023): tiền lệ của batch-coupling attack
