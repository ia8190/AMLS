# Adversarial Machine Learning System (AMLS)

## Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can be vulnerable to adversarial attacks and how adversarial training can improve robustness.

Using the CIFAR-10 dataset, the system allows users to:
- Train CNN models
- Generate FGSM and PGD attacks
- Compare baseline and defended models
- Test custom images
- Visualise adversarial examples

The project is designed as a lightweight and educational framework.

---

## Features

- Interactive CLI system
- Pre-trained models included
- Custom model training
- FGSM attack visualisations
- PGD attack visualisations
- Defence comparisons
- Adjustable epsilon values
- Multiple image predictions

---

## Project Structure

```text
ADV_ML_SYS/
│
├── checkpoints/          # Pre-trained models
├── custom_train/         # User-trained models (gitignored)
├── scripts/
│   ├── train/
│   ├── eval/
│   └── visual/
│
├── images/               # User test images
├── outputs/
│
├── main.py
├── predict_image.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ia8190/AMLS.git
```

Move into the project:

```bash
cd AMLS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the System

Start the interactive menu:

```bash
python main.py
```

From the menu users can:
- Use pre-trained models
- Train custom models
- Run attack visualisations
- Test custom images

---

## Images Folder

Create an `images/` folder and place your test images inside it.

Example:

```text
images/
├── ship.jpg
├── dog.jpg
└── bird.png
```

Supported formats:
- JPG
- PNG
- JPEG
- WEBP

---

## Training Models

The framework supports:
- Baseline CNN training
- FGSM adversarial training
- PGD adversarial training

Custom-trained models are automatically saved to:

```text
custom_train/
```

This folder is excluded from Git.

---

## Example Training Commands

Baseline CNN:

```bash
python scripts/train/train_cifar10_cnn.py
```

FGSM adversarial training:

```bash
python scripts/train/train_fgsm_adv.py
```

PGD adversarial training:

```bash
python scripts/train/train_pgd_adv.py
```

---

## Results

The project demonstrates that:
- CNNs are vulnerable to adversarial attacks
- PGD attacks are generally stronger than FGSM
- Adversarial training improves robustness
- There is a trade-off between robustness and clean accuracy

---

## References

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, 2014
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, 2018

---

## Author

Isihaq Abass

*Per Perseverantiam...*