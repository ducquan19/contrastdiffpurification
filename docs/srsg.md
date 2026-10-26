# Soft Randomized Spectral Guidance (SRSG) và đánh giá tần số thấp

Tài liệu đặc tả phương pháp SRSG, baseline FreqPure cài lại trên cùng sampler, và attack PGD tần số
thấp. Rà soát novelty: artifact `novelty_frequency.md`.

## 1. Động lực

FreqPure (Pei et al., ICML 2025) giữ lại phần tần số thấp của ảnh đầu vào ở mỗi bước khuếch tán
ngược: thay cứng biên độ trong bán kính D_A và chiếu pha trong bán kính D_P vào khoảng ±δ quanh pha
của ảnh đầu vào. Tiền đề: attack ℓ∞/ℓ2 thông thường đặt nhiễu chủ yếu ở tần số cao.

Đó là một "vùng an toàn tĩnh" trong miền tần số: một attacker biết cơ chế sẽ đặt nhiễu vào chính
băng được giữ. SRSG thay việc *thay cứng một băng cố định* bằng việc *kéo mềm, một phần, về một băng
ngẫu nhiên*:

* mọi băng tần đều đi qua khuếch tán thuận đẳng hướng (không băng nào được giữ nguyên);
* bán kính băng được neo và mức độ kéo được rút ngẫu nhiên cho mỗi particle, mỗi lần suy luận;
* toàn bộ phép biến đổi khả vi, nên được đánh giá bằng gradient chính xác.

## 2. Phát biểu chung: thay thế một phần trong miền tần số

Ở mỗi bước ngược, từ ước lượng Tweedie `x̂0` (trong [-1, 1]) và ảnh tham chiếu `x_ref` (ảnh đầu vào
của bộ thanh lọc, cũng trong [-1, 1]), tính ảnh hiệu chỉnh `x̃0` rồi đổi ngược thành nhiễu:

    ε̃ = (x_t − sqrt(ᾱ_t) · x̃0) / sqrt(1 − ᾱ_t),        d = ε̃ − ε

`d` được trả về theo giao thức guidance của `ParticleSampler` (nhiễu dự đoán được cộng `d`), nên
bước DDIM dùng nhất quán `x̃0` và `ε̃`.

**SRSG.** Với F là DFT 2 chiều chuẩn hoá trực giao, tính trên từng kênh màu:

    x̃0 = x̂0 − γ · F⁻¹( M_r ⊙ F(x̂0 − x_ref) )

* `M_r(u, v) = σ((r − D(u, v)) / τ)`: mặt nạ hướng tâm mềm, `D(u, v)` là độ lớn tần số (khoảng
  cách tới tâm phổ sau fftshift, đơn vị "chu kỳ trên ảnh"), `τ` là độ mềm.
* `r ~ U[r_min, r_max]`, `γ ~ U[γ_min, γ_max]`: rút **một lần cho mỗi particle mỗi lần thanh lọc**,
  từ bộ sinh ngẫu nhiên seed theo từng ảnh (cùng seed với nhiễu khuếch tán), nên kết quả của một
  ảnh không phụ thuộc vào batch và tái lập được.
* `γ < 1`: chỉ kéo một phần, phần còn lại do mô hình khuếch tán tự tái tạo.
* Biến thể `mode = amp`: chỉ kéo biên độ, giữ pha của `x̂0`:
  `|X̃| = |X̂| − γ M_r (|X̂| − |X_ref|)`, `X̃ = |X̃| e^{i∠X̂}`.

**FreqPure** (baseline, cài lại trên cùng sampler, ký hiệu như bài gốc):

    Â = A_x̂0 (1 − H_A) + A_ref H_A,   P̂ = Π_{P_ref ± δ}(P_x̂0) H_P + P_x̂0 (1 − H_P),   x̃0 = F⁻¹(Â e^{iP̂})

với `H_A`, `H_P` là bộ lọc lý tưởng bán kính D_A, D_P; phép chiếu pha dùng hiệu pha quấn về
(−π, π]. Tham số mặc định theo bài gốc: D_A = 3, D_P = 2, δ = 0,2. FreqPure là trường hợp riêng
"mặt nạ cứng, bán kính cố định, γ = 1, biên độ + chiếu pha" của cùng phát biểu.

**Khác biệt so với FreqPure**

| | FreqPure | SRSG |
|---|---|---|
| Mặt nạ | Lý tưởng (0/1) | Mềm (sigmoid) |
| Bán kính băng được neo | Cố định (D_A = 3, D_P = 2) | Ngẫu nhiên mỗi particle, mỗi lần chạy |
| Mức độ | Thay hoàn toàn (γ = 1) | Kéo một phần, γ ngẫu nhiên < 1 |
| Thành phần | Biên độ + chiếu pha | Phổ phức (mặc định) hoặc chỉ biên độ |
| Dự đoán | 1 bản | 1 bản hoặc bỏ phiếu K bản |
| Đánh giá trong bài gốc | Gradient qua surrogate; không có attack tần số thấp | Gradient chính xác; PGD tần số thấp; ℓ2 |

## 3. Attack PGD tần số thấp (LF-PGD)

Attack thích ứng nhắm vào băng được neo, bị chặn theo chuẩn ℓ∞ như PGD thường (để so được):

    δ ← G_σ * clip_ε(δ + α · sign(∇_δ L)),   x_adv = clip_[0,1](x + δ)

`G_σ` là bộ lọc thông thấp Gauss (nhân dương, tổng bằng 1), nên `‖G_σ * δ‖∞ ≤ ‖δ‖∞ ≤ ε`: nhiễu luôn
nằm trong quả cầu ℓ∞ và tập trung ở tần số thấp. Tham số: độ rộng băng `σ_f` trong miền tần số
(`--pgd_lowpass_sigma`, đơn vị chu kỳ trên ảnh; nhỏ hơn = nhiễu mượt hơn). Mọi thứ còn lại (vòng
lặp, EOT, chọn ảnh đối kháng theo margin, chấm 8 lần) giữ nguyên như `pgd_full`.

## 4. Tham số mặc định (điểm khởi đầu, chưa tune)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Sampler | DDIM η = 1, S = 20, t* = 100 | Chung cho mọi phương pháp |
| K | 1 (SRSG-SP) hoặc 4 (SRSG-MV) | |
| r_min, r_max | 2, 6 | Ảnh 32 × 32: tần số tối đa khoảng 16 |
| τ | 1,0 | Độ mềm của mặt nạ |
| γ_min, γ_max | 0,3; 0,9 | |
| Cửa sổ áp dụng | mọi bước (t ∈ [0; 1] · t*) | Như FreqPure |
| mode | `complex` | `amp` để ablation |

## 5. Dữ liệu chọn tham số: split `dev`

Để chọn tham số mà không chạm tập báo cáo và tránh vấn đề ảnh train (mô hình pretrained đã học),
`--eval_split dev` lấy ảnh từ **tập test** nhưng loại trừ 512 ảnh của tập báo cáo
(`select_subset_indices(10000, 512, 0)`). Tập con dev chọn theo `--data_seed` như các split khác.
Bảng chính luôn dùng `--eval_split test --data_seed 0`.

## 6. Thí nghiệm theo thứ tự

1. **FreqPure và DiffPure dưới PGD thường và LF-PGD** (64 ảnh test, 200 vòng). Kiểm tra giả thuyết
   "giữ tần số thấp là lỗ hổng" trước khi đầu tư vào SRSG.
2. **Chọn tham số SRSG** trên dev (256 ảnh, 50 vòng), so ghép cặp với DiffPure và FreqPure, dưới cả
   hai loại attack.
3. **Bảng chính** trên test: PGD ℓ∞, LF-PGD, ℓ2; R18 AT làm tham chiếu; kiểm tra EOT gấp đôi và
   transfer cho SRSG.
