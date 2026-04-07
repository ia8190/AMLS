# Adversarial Machine Learning System (AMLS)

##  Overview

This project demonstrates the vulnerability of Convolutional Neural Networks (CNNs) to adversarial attacks and evaluates adversarial training as a defence mechanism.

Using the CIFAR-10 dataset, the system shows how small, imperceptible perturbations can significantly degrade model performance, while also exploring methods to improve robustness.

---

##  Objectives

* Train a baseline CNN on CIFAR-10
* Evaluate performance on clean data
* Generate adversarial examples (FGSM, PGD)
* Assess robustness under attack
* Implement adversarial training
* Compare standard vs defended models

---

##  Project Structure

```
ADV_ML_SYS/
│── scripts/
│   ├── eval/
│   ├── train/
│   ├── visual/
│
│── models/
│── images/
│── predict_image.py
```

---

##  Usage

### Train model

```
python scripts/train/train_cifar10_cnn.py
```

### Evaluate

```
python scripts/eval/eval_clean.py
```

### Run attacks

```
python scripts/eval/fgsm_eval.py
python scripts/eval/pgd_eval.py
```

### Test your own image

```
python predict_image.py --image images/test.png
```

---

##  Custom Images

* Craete folder `images/` 
* Place images inside `images/`
* Recommended size: 32×32
* Formats: PNG, JPG

---

##  Results

* Strong performance on clean data
* Significant degradation under FGSM & PGD
* PGD is more effective
* Adversarial training improves robustness

---

##  Key Insight

Neural networks can be highly confident yet fragile. Robustness must be explicitly addressed in real-world ML systems.

---

##  References

* Goodfellow et al., 2014
* Madry et al., 2018

---

##  Author

Isihaq Abass
TII
