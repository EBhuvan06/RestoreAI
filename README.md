# KLA Image Restoration Solution
**SEMICON AI Hackathon 2026 – Problem Statement 2**

## 1. Problem Statement
The task is to restore semiconductor inspection images that have been degraded by a combination of:
- Speckle noise
- Additive Gaussian noise
- Spatial downsampling (factor of 2)

Given a noisy low-resolution image of size `128 × 128`, the model must produce a clean high-resolution image of size `256 × 256`.

---

## 2. Proposed Method

### 2.1 Core Idea
We formulate the problem as a **joint denoising + super-resolution** task and solve it using a lightweight residual convolutional neural network.

Instead of learning the clean image from scratch, the network learns the **residual** between the bicubic-upsampled input and the ground-truth image:

\[
\hat{I}_{HR} = \text{Bicubic}(I_{LR}) + \mathcal{F}_\theta(I_{LR})
\]

where \(\mathcal{F}_\theta\) is the residual network.

This residual learning strategy significantly stabilizes training and preserves low-frequency structure.

### 2.2 Network Architecture
The model is a compact encoder-decoder network inspired by NAFNet design principles:

- **Encoder**: Progressive feature extraction with downsampling
- **Bottleneck**: Deep feature processing
- **Decoder**: Progressive upsampling with skip connections
- **Final 2× PixelShuffle** layer to reach 256×256 resolution
- Strong residual connection from the bicubic-upsampled input

**Key properties:**
- Parameters: **5.24 Million**
- Input: `1 × 128 × 128`
- Output: `1 × 256 × 256`

### 2.3 Loss Function
We used a combination of robust losses:

\[
\mathcal{L} = \mathcal{L}_{\text{Charbonnier}} + 0.5 \cdot \mathcal{L}_1
\]

The Charbonnier loss is defined as:

\[
\mathcal{L}_{\text{Charbonnier}} = \sqrt{(x - y)^2 + \epsilon^2}
\]

This combination provides robustness to outliers while maintaining good PSNR and visual quality.

### 2.4 Training Strategy
- 90/10 train-validation split (seed = 42)
- Mixed Precision Training (AMP)
- AdamW optimizer
- Cosine Annealing learning rate schedule
- Data augmentation: random flips and 90° rotations
- Best model selected based on validation PSNR

---

## 3. Performance

### Validation Results

| Metric       | Value          |
|--------------|----------------|
| **PSNR**     | **29.18 dB**   |
| **SSIM**     | **0.7797**     |
| **LPIPS**    | **0.2697**     |
| Inference    | ~23 ms / image |
| Throughput   | ~43 FPS        |
| Parameters   | 5.24 M         |

The model achieves a strong balance between restoration quality and computational efficiency.

---

## 4. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```
###Run Inference
```bash
Bashpython run.py <input-dir> <output-dir>
```
###Example:
```bash
Bashpython run.py ./input ./output
