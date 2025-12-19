# 🎥 Adaptive Bitrate Vision: A Neural Codec for Bandwidth-Constrained Survellaince (Weights)

*A PyTorch-based neural video codec that maintains **high fidelity on semantic regions** while intentionally degrading backgrounds under extreme bandwidth constraints.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧭 Overview

**ABVCodec** is a semantic-aware neural video compression system designed to maintain **high perceptual quality for important foreground regions (e.g., faces)** even when operating at **extremely low bandwidth ratios (as low as 0.1×)**.

The codec dynamically modulates latent-space noise, bit-rate pressure, and region-weighted reconstruction losses to:

- Achieve **18–20 dB** PSNR on semantic regions (faces) at **0.1× bandwidth**
- Reduce background PSNR intentionally to **12–15 dB**, mimicking aggressive compression
- Maintain graceful degradation, rather than structural collapse, across bandwidth regimes
- Predict soft semantic masks in real-time to guide reconstruction and noise injection

This project demonstrates **neural image/video compression**, **semantic-aware resource allocation**, and **rate–distortion control** in a compact but research-relevant system.

---

## 🔍 Key Features

### 1. Semantic-Aware Encoder–Decoder Architecture

**U-Net backbone** with:
- Multi-scale feature extraction
- Semantic segmentation head
- Latent bottleneck for codec control

**Produces both:**
- `recon` — reconstructed frame
- `sem_mask` — predicted semantic probability map

### 2. PSNR-Targeted Graceful Degradation Engine

At **low bandwidths (≤0.1×)**:
- Semantic regions receive **low noise**, preserving clarity
- Background regions receive **high noise**, simulating aggressive compression
- Region-specific PSNR targets enforced via custom loss functions

### 3. Real + Synthetic Dataset Pipeline

- Loads frames from a local `.mp4` video
- Fallback to synthetic frames with face-like motion + background objects
- Automatic semantic mask generation using:
  - Canny edges
  - HSV saturation cues
  - Manual synthetic "face" regions for controllable training

### 4. PSNR Tracking & Visualization

**Computes:**
- Overall PSNR
- Semantic-region PSNR
- Background PSNR

**Saves:**
- Error maps
- Semantic overlays
- Summary panels
- Assembled **GIF** demonstrating quality evolution

### 5. Full Training Loop with RD Control

**Bandwidth schedule** cycles through:
```python
[1.0, 0.8, 0.5, 0.3, 0.1, 0.1, 0.15, 0.2, 0.4, 0.7, 1.0, 0.6]
```

Noise + rate penalty adjusted per bandwidth

**Loss terms:**
- Semantic-aware MSE
- Edge consistency loss
- Semantic segmentation loss
- Rate penalty from latent magnitude

### 6. Outputs & Artifacts

- Reconstructed frame comparisons
- PSNR-verified benchmark frames
- Full training GIF
- Trained model checkpoint (`abv_codec_trained.pth`)

---

## 🏗️ Architecture
```
Input Frame
    │
    ▼
Encoder (ConvBlocks + Downsampling)
    │
    ├──► Semantic Head → Softmax → Semantic Mask (1×H×W)
    │
    └──► Bottleneck → Noise Injection (bandwidth-aware)
                       │
                       ▼
Decoder (Upsampling + Skip Connections)
    │
    ▼
Reconstruction (Sigmoid)
```

### Noise Injection Modulation

Noise injection is modulated by:
- Bandwidth ratio `bw_ratio`
- Semantic mask probabilities
- PSNR-target ranges (semantic vs. background)

---

## ⚙️ Technical Highlights

### Semantic Mask Guidance

Semantic mask predicted at bottleneck resolution influences:
- Noise scaling
- Reconstruction weighting
- Region-sensitive PSNR optimization

### PSNR-Driven Loss Design

The codec uses:

1. **Semantic-aware reconstruction loss**
2. **Edge consistency loss**
3. **Semantic segmentation cross-entropy**
4. **Rate penalty:** `mean(|latent|)`

### Bandwidth Simulation

| BW Ratio     | Semantic Noise | Background Noise | Expected PSNR        |
|--------------|----------------|------------------|----------------------|
| **0.1×**     | Very low       | Very high        | 18–20 dB / ≤15 dB    |
| **0.2–0.5×** | Medium         | Moderate         | Gradual recovery     |
| **1.0×**     | Minimal noise  | Minimal noise    | Clean reconstruction |

---

## 💻 Installation

### Prerequisites
```bash
pip install torch torchvision opencv-python imageio matplotlib pillow numpy
```

### Quick Start

#### 1. Clone Repository
```bash
git clone <repo-url>
cd ABVCodec
```

#### 2. Run Training
```bash
python main.py
```

#### 3. Optional: Add Your Own Video

Place a file named `my_video.mp4` in the project directory.

---

## 📁 Project Structure
```text
ABVCodec/
├── abv_codec.py              # Model architecture and loss functions
├── dataset.py                # Video loader + synthetic generator
├── train.py                  # Training loop with PSNR tracking
├── outputs/                  # Saved visualization frames
├── abv_psnr_demo_*.gif       # Training GIF summary
├── abv_codec_trained.pth     # Final model checkpoint
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

> **Note:** Current version uses a monolithic structure for prototyping. Modular refactoring planned for future releases.

---

## 🧪 Training Outputs

The system automatically generates:

- Per-step reconstructed frames
- Error heatmaps
- Semantic overlays
- PSNR summary blocks
- A training GIF such as: `abv_psnr_demo_20251208_163202.gif`

### Example Metrics at 0.1× Bandwidth
```
0.1x BW STRESS TEST RESULT:
→ Face/Semantic PSNR: 19.2 dB (Target: 18–20)
→ Background PSNR:    13.4 dB (Target: ≤15)
→ GRACEFUL DEGRADATION ACHIEVED! ✓
```

**Visual Results:**
- **Face PSNR:** 18–20 dB
- **Background PSNR:** 12–15 dB
- **Status:** Graceful degradation achieved

---

## 📊 Performance Benchmarks

### PSNR Across Bandwidth Regimes

| Bandwidth | Semantic PSNR | Background PSNR | Overall PSNR |
|-----------|---------------|-----------------|--------------|
| 1.0×      | 35-38 dB      | 33-35 dB        | 34-36 dB     |
| 0.5×      | 28-30 dB      | 22-25 dB        | 25-27 dB     |
| 0.1×      | 18-20 dB      | 12-15 dB        | 14-17 dB     |

### Key Observations

- **Semantic preservation:** Maintains 18+ dB PSNR on faces even at extreme compression
- **Graceful degradation:** No catastrophic quality collapse
- **Region-aware allocation:** Intelligently prioritizes important visual content

---

## 🚀 Future Extensions

- [ ] Replace synthetic masks with pretrained segmentation (Segment Anything / DeepLab)
- [ ] Add entropy model (hyperprior) for true neural compression
- [ ] Introduce perceptual losses (LPIPS, VGG-based)
- [ ] Train on multi-video dataset (Kinetics, UCF-101)
- [ ] Replace noise modulation with learned rate-control module
- [ ] Add transformer-based temporal modeling for long sequences
- [ ] Implement arithmetic coding for bitstream generation
- [ ] Add support for variable frame rates
- [ ] Create web demo for real-time compression visualization

---

## ⚠️ Limitations

- Latent bitrate is **proxy-based**, not a true entropy coder
- Semantic segmentation is **weak** for real-world frames (edge + saturation heuristic)
- Background degradation may become too aggressive under some videos
- No explicit temporal consistency enforcement across frames
- Requires GPU for reasonable training speed
- Limited to single-frame processing (no temporal modeling yet)

---

## 🔬 Research Context

This project explores the intersection of:

- **Neural compression:** Learning-based codecs vs. traditional methods (H.264, HEVC)
- **Semantic-aware processing:** Region-of-interest coding with deep learning
- **Rate-distortion optimization:** Balancing quality and bandwidth constraints
- **Perceptual quality metrics:** Beyond PSNR to human visual perception

### Related Work

- [**Ballé et al. (2018)**](https://arxiv.org/abs/1802.01436) - Variational image compression
- [**Cheng et al. (2020)**](https://arxiv.org/abs/2001.01568) - Learned image compression with discretized Gaussian mixture
- [**Lu et al. (2019)**](https://arxiv.org/abs/1904.05677) - DVC: Deep video compression framework

---

## 📖 Technical Documentation

### Model Architecture Details
```python
# Encoder
- Conv layers: [3→64→128→256]
- Downsampling: 3 stages (8× reduction)
- Bottleneck: 256 channels

# Semantic Head
- 1×1 Conv → Sigmoid
- Outputs: Per-pixel semantic probability

# Decoder
- Upsampling: 3 stages with skip connections
- Conv layers: [256→128→64→3]
- Final activation: Sigmoid
```

### Loss Function Components
```python
total_loss = (
    λ₁ * semantic_aware_mse +
    λ₂ * edge_loss +
    λ₃ * segmentation_loss +
    λ₄ * rate_penalty
)
```

---

## 🧑‍💻 Authors

**Aarush, Keerthana, Amulya**  
AI/ML Engineers (CSE — AI & ML)  
Specializing in model engineering, neural compression, and efficient deep learning systems

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Neural compression research community
- Open-source video compression benchmarks

---

## 📜 License

MIT Licence

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Improving semantic segmentation accuracy
- Implementing true entropy coding
- Adding perceptual loss functions
- Optimizing inference speed
- Creating interactive demos

Please feel free to submit a Pull Request or open an issue for discussion.

---

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Contact: [aarushinc1@gmail.com]
- Check the [Wiki](link-to-wiki) for detailed documentation

---

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@misc{abvcodec2024,
  author = {Aarush},
  title = {ABVCodec: Semantic-Aware Neural Video Codec with PSNR-Targeted Graceful Degradation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ABVCodec}
}
```

---

**Built with 🎥 for intelligent video compression research**
