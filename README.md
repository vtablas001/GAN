# Butterfly GAN Generator

A Streamlit demo that generates synthetic butterfly images using a pretrained Lightweight GAN. Each click produces a unique butterfly that does not exist in nature, showcasing the power of generative adversarial networks trained on limited data.

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/vtablas001/GAN1)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://huggingface.co/spaces/vtablas001/GAN1)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [What Are GANs?](#what-are-gans)
- [About the Model](#about-the-model)
- [Demo Preview](#demo-preview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Technologies](#technologies)
- [References](#references)
- [Author](#author)

---

## Overview

This project deploys a **Lightweight GAN** model as an interactive web application using **Streamlit** on **Hugging Face Spaces**. Users can generate photorealistic butterfly images at 512x512 resolution with a single click, exploring how generative models learn visual patterns from a curated dataset of only ~1,000 images.

---

## What Are GANs?

**Generative Adversarial Networks (GANs)** are a class of deep learning models introduced by Goodfellow et al. (2014). A GAN consists of two neural networks trained simultaneously in a competitive setup:

| Component       | Role                                                     |
|-----------------|----------------------------------------------------------|
| **Generator**   | Creates synthetic images from random noise vectors       |
| **Discriminator** | Evaluates whether an image is real or generated        |

During training, the generator improves at producing realistic images while the discriminator improves at detecting fakes. This adversarial dynamic drives both networks toward equilibrium, resulting in a generator capable of producing highly convincing outputs.

```
Random Noise (z) ──► [ Generator ] ──► Fake Image ──┐
                                                     ├──► [ Discriminator ] ──► Real or Fake?
                          Real Image ────────────────┘
```

---

## About the Model

This demo uses the pretrained model [`ceyda/butterfly_cropped_uniq1K_512`](https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512), a **Lightweight GAN** trained by [Ceyda Cinarel](https://huggingface.co/ceyda).

| Detail              | Description                                                |
|---------------------|------------------------------------------------------------|
| **Architecture**    | Lightweight GAN (Liu & Abbeel, 2021)                       |
| **Dataset**         | ~1,000 unique butterfly species from the Smithsonian Collection |
| **Resolution**      | 512 x 512 pixels                                           |
| **Training**        | 2x NVIDIA A4000 GPUs, ~24 hours                           |
| **Key Parameters**  | Batch size 64, gradient accumulation every 4 steps, FP16 mixed precision |
| **Purpose**         | Educational and experimental use                           |

The Lightweight GAN architecture is specifically designed to achieve high-quality image generation with limited training data, making it ideal for niche domains where large datasets are unavailable.

---

## Demo Preview

The application presents a simple interface where users can:

1. Click the **Generate** button to create a new butterfly image.
2. View the generated 512x512 image rendered in real time.
3. Repeat to explore the diversity of the model's learned representations.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/butterfly-gan-generator.git
cd butterfly-gan-generator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Requirements File

```text
streamlit
torch
huggan
numpy
Pillow
```

---

## Project Structure

```
butterfly-gan-generator/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

---

## How It Works

The generation pipeline follows three steps:

1. **Sampling**: A random noise vector `z` is sampled from a standard normal distribution with dimensionality matching the model's latent space.

2. **Forward Pass**: The noise vector is passed through the generator network, which transforms it into a 512x512x3 RGB image tensor.

3. **Post-processing**: The output tensor is clamped to [0, 1], rescaled to [0, 255], and converted to a NumPy array for display.

```python
import torch
from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN

gan = LightweightGAN.from_pretrained("ceyda/butterfly_cropped_uniq1K_512")
gan.eval()

with torch.no_grad():
    images = gan.G(torch.randn(1, gan.latent_dim)).clamp_(0., 1.) * 255
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
```

---

## Technologies

- **[PyTorch](https://pytorch.org/)**: Deep learning framework for model inference.
- **[Streamlit](https://streamlit.io/)**: Web application framework for ML demos.
- **[Hugging Face Hub](https://huggingface.co/)**: Model hosting and Spaces deployment.
- **[huggan](https://github.com/huggingface/community-events)**: Community library providing the Lightweight GAN implementation.

---

## References

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems, 27*, 2672-2680.

- Liu, B., & Abbeel, P. (2021). Towards faster and stabilized GAN training for high-fidelity few-shot image synthesis. *International Conference on Learning Representations (ICLR)*.

- Cinarel, C. (2022). *butterfly_cropped_uniq1K_512* [Pretrained model]. Hugging Face. https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512

---

## Author

**Victor Tablas**
- Hugging Face: [@vtablas001](https://huggingface.co/vtablas001)

---

> *"Every butterfly generated is a unique creation that exists nowhere in nature, born entirely from patterns learned by a neural network."*
