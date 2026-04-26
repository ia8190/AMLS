# Adversarial Machine Learning System (AMLS)

## Overview

This project demonstrates the vulnerability of Convolutional Neural Networks (CNNs) to adversarial attacks and evaluates adversarial training as a defence mechanism.

Using the CIFAR-10 dataset, the system shows how small, imperceptible perturbations can significantly degrade model performance, while also exploring methods to improve robustness.

---

## Objectives

- Train a baseline CNN on CIFAR-10  
- Evaluate performance on clean data  
- Generate adversarial examples (FGSM, PGD)  
- Assess robustness under attack  
- Implement adversarial training  
- Compare standard vs defended models  

---

## Project Structure

```
ADV_ML_SYS/
│── checkpoints/ # Pre-trained model weights
│
│── scripts/
│ ├── train/ # Training scripts (baseline & adversarial)
│ ├── eval/ # Evaluation scripts (clean, FGSM, PGD)
│ ├── visual/ # Visualisation & attack demos
│
│── models/ # CNN architecture
│── images/ # Custom test images
│── outputs/ # Generated results (optional)
│
│── predict_image.py # Run prediction + attack demo
│── README.md
│── .gitignore
```



## Pre-trained Models (Recommended)

Pre-trained model checkpoints are included in the `checkpoints/` folder.

This allows users to:
- Run predictions immediately  
- Evaluate adversarial attacks without retraining  
- Compare baseline vs defended models  

### Run demo directly


python predict_image.py


---

## Installation


git clone https://github.com/ia8190/AMLS.git

cd AMLS <br>
pip install -r requirements.txt


---

## Usage

### Test your own image (recommended)

- Create `images/` folder  
- Place your image inside the `images/` folder  
- Update the image path in script files if needed  

---

### (Optional) Train models


python scripts/train/train_cifar10_cnn.py


Adversarial training:


python scripts/train/train_fgsm_adv.py<br>  
python scripts/train/train_pgd_adv.py


---

### Evaluate models


python scripts/eval/eval_clean.py


---

### Run adversarial attacks


python scripts/eval/fgsm_eval.py  
python scripts/eval/pgd_eval.py


---

## Custom Images

- Place images inside the `images/` folder  
- Recommended size: 32×32  
- Supported formats: PNG, JPG  

---

## Results

- Strong performance on clean data  
- Significant degradation under FGSM and PGD attacks  
- PGD is more effective than FGSM  
- Adversarial training improves robustness  

---

## Key Insight

Neural networks can be highly confident yet fragile. Robustness must be explicitly addressed in real-world machine learning systems.

---

## References

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, 2014  
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, 2018  

---

## Author

Isihaq Abass<br>
*Per Perseverantiam...*<br>

