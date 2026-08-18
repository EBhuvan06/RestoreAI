# Method & Mathematical Formulation
## KLA Image Restoration – SEMICON AI Hackathon 2026

---

## 1. Problem Definition

We are given a noisy low-resolution image \( I_{LR} \in \mathbb{R}^{128 \times 128} \) that has been degraded by a combination of:

- Speckle noise
- Additive Gaussian noise
- Spatial downsampling (factor of 2)

The goal is to recover a clean high-resolution image \( \hat{I}_{HR} \in \mathbb{R}^{256 \times 256} \).

---

## 2. Overall Approach

We solve the problem as a **joint denoising + super-resolution** task using a residual convolutional neural network.

Instead of asking the network to generate the clean image from scratch, we ask it to learn only the missing information (the residual):

\[
\hat{I}_{HR} = \underbrace{\text{Bicubic}(I_{LR})}_{\text{low-frequency base}} + \underbrace{\mathcal{F}_\theta(I_{LR})}_{\text{learned residual}}
\]

**Why this helps (simple words):**  
The bicubic-upsampled image already contains a reasonable low-frequency approximation of the structure. The network only needs to add the high-frequency details and remove the noise. This makes training more stable and preserves overall structure better.

---

## 3. Network Architecture

We designed a compact encoder-decoder network inspired by NAFNet principles.

### 3.1 Building Block – NAFBlock

Each processing unit is a Non-Linear Activation Free (NAF) block:

\[
\begin{aligned}
y &= \text{Norm}(x) \\
y &= \text{Conv}_{1\times1}(y) \\
y &= \text{DepthwiseConv}_{3\times3}(y) \\
y &= \text{SimpleGate}(y) \\
y &= y \odot \text{SCA}(y) \\
y &= \text{Conv}_{1\times1}(y) \\
x &\leftarrow x + \beta \cdot y
\end{aligned}
\]

- **SimpleGate**: splits channels into two halves and multiplies them (replaces ReLU/GELU).
- **SCA**: Simplified Channel Attention.
- **β**: learnable residual scale.

### 3.2 Full Network Flow
Input (1 × 128 × 128)
│
▼
Intro Convolution
│
Encoder Stage 1 ──────────────┐
│                        │
Downsample                    │
│                        │
Encoder Stage 2 ──────────────┤
│                        │
Downsample                    │
│                        │
Encoder Stage 3 ──────────────┤
│                        │
Downsample                    │
│                        │
Bottleneck                 │
│                        │
Upsample + Skip Connection ───┘
│
Decoder Stage 3
│
Upsample + Skip Connection
│
Decoder Stage 2
│
Upsample + Skip Connection
│
Decoder Stage 1
│
Final 2× PixelShuffle
│

Bicubic Upsampled Input     ← residual connection
│
Output (1 × 256 × 256)

text**Model size:** 5.24 Million parameters.

---

## 4. Loss Function

We train the network with a combination of robust losses:

\[
\mathcal{L} = \mathcal{L}_{\text{Charbonnier}} + 0.5 \cdot \mathcal{L}_{1}
\]

### Charbonnier Loss
\[
\mathcal{L}_{\text{Charbonnier}} = \sqrt{(x - y)^{2} + \epsilon^{2}}
\]

**Why we chose it:**  
It behaves like L2 when the error is small and like L1 when the error is large. This makes training more robust to outliers (heavy noise) while still giving good PSNR.

---

## 5. Training Strategy

| Item                    | Choice                                      |
|-------------------------|---------------------------------------------|
| Train / Val split       | 90% / 10% (fixed seed = 42)                 |
| Optimizer               | AdamW                                       |
| Learning rate schedule  | Cosine Annealing                            |
| Mixed Precision         | Yes (AMP)                                   |
| Data Augmentation       | Random flips + 90° rotations                |
| Batch size              | 4                                           |
| Model selection         | Highest validation PSNR                     |

---

## 6. Final Performance

| Metric       | Value              |
|--------------|--------------------|
| **PSNR**     | **29.18 dB**       |
| **SSIM**     | **0.7758**         |
| **LPIPS**    | **0.2697**         |
| Inference    | 29.07 ms / image   |
| Throughput   | 34.4 FPS           |
| Parameters   | 5.24 Million       |

---

## 7. Design Decisions & Observations

1. **Residual learning** around the bicubic baseline was critical for stable convergence.
2. A pure Transformer (Restormer) can reach higher PSNR (~31.8 dB) but is significantly slower and heavier. Our CNN offers a practical speed-quality trade-off suitable for real inspection pipelines.
3. Dense random textures remain the most challenging cases (mild over-smoothing can still appear).
4. The complete solution is fully offline and requires no external downloads or internet access.

---

## 8. How to Run the Solution

```bash
python run.py <input-dir> <output-dir>
