from pathlib import Path
import json
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# cifar-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# cnn model used for cifar-10
class CIFAR_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # feature layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )

        # classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # forward pass
    def forward(self, x):
        return self.classifier(self.features(x))


# find the project root folder
def find_root(start: Path) -> Path:
    p = start
    while p != p.parent:
        if (p / "checkpoints").exists():
            return p
        p = p.parent
    raise RuntimeError("Checkpoint folder not found.")


# load a saved model checkpoint
def load_model(device, ckpt_path: Path) -> nn.Module:
    model = CIFAR_CNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # support both checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# get normalised image bounds
def _norm_bounds(mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    x_min = (0.0 - mean_t) / std_t
    x_max = (1.0 - mean_t) / std_t
    return mean_t, std_t, x_min, x_max


# make batch predictions
@torch.no_grad()
def batch_predict(model, x):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


# create an fgsm adversarial batch
def fgsm_attack_norm(model, x_norm, y, eps_px, mean, std):
    model.eval()
    device = x_norm.device
    _, std_t, x_min, x_max = _norm_bounds(mean, std, device)

    # convert pixel epsilon to normalised space
    eps = eps_px / std_t
    x_adv = x_norm.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()

        # keep image inside valid range
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)

    return x_adv.detach()


# create a pgd adversarial batch
def pgd_attack_norm(model, x_norm, y, eps_px, alpha_px, steps, mean, std, random_start=True):
    model.eval()

    device = x_norm.device
    _, std_t, x_min, x_max = _norm_bounds(mean, std, device)

    # convert pixel values to normalised space
    eps = eps_px / std_t
    alpha = alpha_px / std_t
    x_orig = x_norm.detach()

    # start from random noise near the image
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)
    else:
        x_adv = x_orig.clone()

    loss_fn = nn.CrossEntropyLoss()

    # apply pgd steps
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # keep image inside epsilon range
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            # keep image inside valid range
            x_adv = torch.max(torch.min(x_adv, x_max), x_min)
        x_adv = x_adv.detach()
    return x_adv


# create a confusion matrix
def confusion_matrix_from_preds(y_true, y_pred, num_classes=10):
    idx = (y_true * num_classes + y_pred).to(torch.int64)

    cm = torch.bincount(
        idx,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    return cm


# calculate accuracy for each class
def per_class_accuracy(cm):
    correct = cm.diag()
    total = cm.sum(dim=1).clamp(min=1)
    return (correct / total).tolist()


# plot a confusion matrix
def plot_confusion_matrix(cm, title, out_path):
    cm = cm.cpu().numpy()
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.colorbar()

    # add numbers to cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]

            if val != 0:
                plt.text(j, i, str(val), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# plot accuracy or attack success curve
def plot_curve(x_vals, y1, y2, label1, label2, title, out_path, ylabel="Accuracy (%)"):
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y1, marker="o", label=label1)
    plt.plot(x_vals, y2, marker="o", label=label2)
    plt.xlabel("Epsilon (out of 255)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# plot bar chart comparison
def plot_bars(values1, values2, label1, label2, title, out_path, ylabel="Accuracy"):
    x = list(range(len(classes)))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.2 for i in x], values1, width=0.4, label=label1)
    plt.bar([i + 0.2 for i in x], values2, width=0.4, label=label2)
    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# plot confidence histogram
def plot_hist(values_a, values_b, label_a, label_b, title, out_path, bins=20):
    plt.figure(figsize=(8, 5))
    plt.hist(values_a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(values_b, bins=bins, alpha=0.6, label=label_b)
    plt.xlabel("Confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# evaluate clean or attacked model performance
def eval_mode(
    model,
    loader,
    device,
    mode_name,
    attack=None,
):
    y_all = []
    pred_all = []
    conf_all = []
    clean_correct_total = 0
    clean_correct_flipped = 0
    model.eval()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # clean evaluation
        if attack is None:
            pred, conf = batch_predict(model, x)

        # adversarial evaluation
        else:
            with torch.no_grad():
                pred_clean = model(x).argmax(1)
                clean_correct = (pred_clean == y)
                clean_correct_total += clean_correct.sum().item()

            x_adv = attack(x, y)
            pred, conf = batch_predict(model, x_adv)

            with torch.no_grad():
                adv_wrong = (pred != y)
                flipped = clean_correct & adv_wrong
                clean_correct_flipped += flipped.sum().item()

        y_all.append(y.detach().cpu())
        pred_all.append(pred.detach().cpu())
        conf_all.append(conf.detach().cpu())

    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    conf_all = torch.cat(conf_all, dim=0)
    cm = confusion_matrix_from_preds(y_all, pred_all, num_classes=10)
    acc = (pred_all == y_all).float().mean().item()
    confidences = conf_all.tolist()

    # attack success rate
    if attack is None:
        asr = None
    else:
        asr = clean_correct_flipped / max(clean_correct_total, 1)

    return acc, cm, confidences, asr


def main():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running full evaluation report...")
    print("May take several minutes depending on hardware.")
    print("Device:", device)

    ROOT = find_root(Path(__file__).resolve())

    # model paths
    baseline_ckpt = ROOT / "checkpoints" / "cnn_cifar10_best.pt"
    defence_ckpt = ROOT / "checkpoints" / "cnn_cifar10_pgdadv_best.pt"

    # output folder
    out_dir = ROOT / "outputs" / "eval_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cifar-10 normalisation values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # test transform
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load cifar-10 test set
    testset = torchvision.datasets.CIFAR10(
        root=str(ROOT / "data"),
        train=False,
        download=False,
        transform=tf
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    baseline = load_model(device, baseline_ckpt)
    defence = load_model(device, defence_ckpt)

    # evaluation settings
    eps_list = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255]
    eps_x = [int(round(e * 255)) for e in eps_list]

    # pgd settings
    pgd_alpha = 2 / 255
    pgd_steps = 7
    pgd_random_start = True

    # summary data
    summary = {
        "models": {
            "baseline": {"ckpt": str(baseline_ckpt)},
            "defence": {"ckpt": str(defence_ckpt)},
        },
        "settings": {
            "eps_list_px": [float(e) for e in eps_list],
            "pgd": {
                "alpha": float(pgd_alpha),
                "steps": int(pgd_steps),
                "random_start": bool(pgd_random_start)
            },
        },
        "results": {}
    }

    # clean evaluation
    print("\n=== CLEAN EVALUATION ===")

    b_clean_acc, b_clean_cm, b_clean_confs, _ = eval_mode(
        baseline,
        test_loader,
        device,
        "clean",
        attack=None
    )

    d_clean_acc, d_clean_cm, d_clean_confs, _ = eval_mode(
        defence,
        test_loader,
        device,
        "clean",
        attack=None
    )

    print(f"Baseline clean acc: {b_clean_acc*100:.2f}%")
    print(f"Defence  clean acc: {d_clean_acc*100:.2f}%")

    plot_confusion_matrix(
        b_clean_cm,
        "Baseline - Confusion Matrix (Clean)",
        out_dir / "cm_baseline_clean.png"
    )

    plot_confusion_matrix(
        d_clean_cm,
        "Defence - Confusion Matrix (Clean)",
        out_dir / "cm_defence_clean.png"
    )

    b_clean_pc = per_class_accuracy(b_clean_cm)
    d_clean_pc = per_class_accuracy(d_clean_cm)

    plot_bars(
        [v * 100 for v in b_clean_pc],
        [v * 100 for v in d_clean_pc],
        "Baseline",
        "Defence",
        "Per-class Accuracy (Clean)",
        out_dir / "per_class_clean.png",
        ylabel="Accuracy (%)"
    )

    # robustness curves
    print("\n=== ROBUSTNESS CURVES (FGSM + PGD) ===")

    b_fgsm_accs, d_fgsm_accs = [], []
    b_pgd_accs, d_pgd_accs = [], []
    b_fgsm_asr, d_fgsm_asr = [], []
    b_pgd_asr, d_pgd_asr = [], []
    b_pgd_confs_eps8 = None
    d_pgd_confs_eps8 = None
    b_pgd_cm_eps8 = None
    d_pgd_cm_eps8 = None

    for eps in eps_list:
        # use clean results when epsilon is zero
        if eps == 0:
            b_fgsm_accs.append(b_clean_acc)
            d_fgsm_accs.append(d_clean_acc)
            b_pgd_accs.append(b_clean_acc)
            d_pgd_accs.append(d_clean_acc)
            b_fgsm_asr.append(0.0)
            d_fgsm_asr.append(0.0)
            b_pgd_asr.append(0.0)
            d_pgd_asr.append(0.0)

            print(
                f"eps=0/255 | "
                f"FGSM (B/D) {b_clean_acc*100:.1f}% / {d_clean_acc*100:.1f}% | "
                f"PGD (B/D) {b_clean_acc*100:.1f}% / {d_clean_acc*100:.1f}%"
            )

            continue

        # fgsm attack wrapper
        def fgsm_attack_for(model):
            return lambda x, y: fgsm_attack_norm(
                model,
                x,
                y,
                eps_px=eps,
                mean=mean,
                std=std
            )

        b_fgsm_acc, _, _, b_asr = eval_mode(
            baseline,
            test_loader,
            device,
            "fgsm",
            attack=fgsm_attack_for(baseline)
        )

        d_fgsm_acc, _, _, d_asr = eval_mode(
            defence,
            test_loader,
            device,
            "fgsm",
            attack=fgsm_attack_for(defence)
        )

        b_fgsm_accs.append(b_fgsm_acc)
        d_fgsm_accs.append(d_fgsm_acc)

        b_fgsm_asr.append(b_asr)
        d_fgsm_asr.append(d_asr)

        # pgd attack wrapper
        def pgd_attack_for(model):
            return lambda x, y: pgd_attack_norm(
                model,
                x,
                y,
                eps_px=eps,
                alpha_px=pgd_alpha,
                steps=pgd_steps,
                mean=mean,
                std=std,
                random_start=pgd_random_start
            )

        b_pgd_acc, b_cm, b_confs, b_asr_pgd = eval_mode(
            baseline,
            test_loader,
            device,
            "pgd",
            attack=pgd_attack_for(baseline)
        )

        d_pgd_acc, d_cm, d_confs, d_asr_pgd = eval_mode(
            defence,
            test_loader,
            device,
            "pgd",
            attack=pgd_attack_for(defence)
        )

        b_pgd_accs.append(b_pgd_acc)
        d_pgd_accs.append(d_pgd_acc)
        b_pgd_asr.append(b_asr_pgd)
        d_pgd_asr.append(d_asr_pgd)

        # store eps 8 results for extra plots
        if abs(eps - (8 / 255)) < 1e-12:
            b_pgd_confs_eps8 = b_confs
            d_pgd_confs_eps8 = d_confs
            b_pgd_cm_eps8 = b_cm
            d_pgd_cm_eps8 = d_cm

        print(
            f"eps={int(round(eps*255))}/255 | "
            f"FGSM acc (B/D): {b_fgsm_acc*100:5.1f}% / {d_fgsm_acc*100:5.1f}% | "
            f"PGD acc (B/D): {b_pgd_acc*100:5.1f}% / {d_pgd_acc*100:5.1f}%"
        )

    # fgsm accuracy curve
    plot_curve(
        eps_x,
        [v * 100 for v in b_fgsm_accs],
        [v * 100 for v in d_fgsm_accs],
        "Baseline",
        "Defence",
        "FGSM Robustness Curve (Accuracy vs Epsilon)",
        out_dir / "curve_fgsm_accuracy.png",
        ylabel="Accuracy (%)"
    )

    # pgd accuracy curve
    plot_curve(
        eps_x,
        [v * 100 for v in b_pgd_accs],
        [v * 100 for v in d_pgd_accs],
        "Baseline",
        "Defence",
        "PGD Robustness Curve (Accuracy vs Epsilon)",
        out_dir / "curve_pgd_accuracy.png",
        ylabel="Accuracy (%)"
    )

    # fgsm attack success curve
    plot_curve(
        eps_x,
        [v * 100 for v in b_fgsm_asr],
        [v * 100 for v in d_fgsm_asr],
        "Baseline",
        "Defence",
        "FGSM Attack Success Rate (clean-correct → adv-wrong)",
        out_dir / "curve_fgsm_asr.png",
        ylabel="ASR (%)"
    )

    # pgd attack success curve
    plot_curve(
        eps_x,
        [v * 100 for v in b_pgd_asr],
        [v * 100 for v in d_pgd_asr],
        "Baseline",
        "Defence",
        "PGD Attack Success Rate (clean-correct > adv-wrong)",
        out_dir / "curve_pgd_asr.png",
        ylabel="ASR (%)"
    )

    # pgd eps 8 confusion matrices and per-class accuracy
    if b_pgd_cm_eps8 is not None and d_pgd_cm_eps8 is not None:
        plot_confusion_matrix(
            b_pgd_cm_eps8,
            "Baseline - Confusion Matrix (PGD eps=8/255)",
            out_dir / "cm_baseline_pgd_eps8.png"
        )

        plot_confusion_matrix(
            d_pgd_cm_eps8,
            "Defence - Confusion Matrix (PGD eps=8/255)",
            out_dir / "cm_defence_pgd_eps8.png"
        )

        b_pgd_pc = per_class_accuracy(b_pgd_cm_eps8)
        d_pgd_pc = per_class_accuracy(d_pgd_cm_eps8)

        plot_bars(
            [v * 100 for v in b_pgd_pc],
            [v * 100 for v in d_pgd_pc],
            "Baseline",
            "Defence",
            "Per-class Accuracy (PGD eps=8/255)",
            out_dir / "per_class_pgd_eps8.png",
            ylabel="Accuracy (%)"
        )

    # confidence histograms
    if b_pgd_confs_eps8 is not None and d_pgd_confs_eps8 is not None:
        plot_hist(
            b_clean_confs,
            b_pgd_confs_eps8,
            "Baseline Clean",
            "Baseline PGD eps=8/255",
            "Baseline Confidence Distribution (Clean vs PGD eps=8/255)",
            out_dir / "hist_conf_baseline_clean_vs_pgd8.png"
        )

        plot_hist(
            d_clean_confs,
            d_pgd_confs_eps8,
            "Defence Clean",
            "Defence PGD eps=8/255",
            "Defence Confidence Distribution (Clean vs PGD eps=8/255)",
            out_dir / "hist_conf_defence_clean_vs_pgd8.png"
        )

    # save summary results
    summary["results"] = {
        "clean_accuracy": {
            "baseline": b_clean_acc,
            "defence": d_clean_acc
        },
        "fgsm_accuracy_curve": {
            "eps_255": eps_x,
            "baseline": b_fgsm_accs,
            "defence": d_fgsm_accs
        },
        "pgd_accuracy_curve": {
            "eps_255": eps_x,
            "baseline": b_pgd_accs,
            "defence": d_pgd_accs
        },
        "fgsm_asr_curve": {
            "eps_255": eps_x,
            "baseline": b_fgsm_asr,
            "defence": d_fgsm_asr
        },
        "pgd_asr_curve": {
            "eps_255": eps_x,
            "baseline": b_pgd_asr,
            "defence": d_pgd_asr
        },
        "per_class_clean": {
            "baseline": b_clean_pc,
            "defence": d_clean_pc
        },
    }

    if b_pgd_cm_eps8 is not None and d_pgd_cm_eps8 is not None:
        summary["results"]["per_class_pgd_eps8"] = {
            "baseline": b_pgd_pc,
            "defence": d_pgd_pc
        }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # save headline csv
    csv_path = out_dir / "headline_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow([
            "Model",
            "CleanAcc(%)",
            "PGD@8/255 Acc(%)",
            "FGSM@8/255 Acc(%)"
        ])

        idx8 = eps_x.index(8) if 8 in eps_x else -1
        b_pgd8 = b_pgd_accs[idx8] if idx8 >= 0 else None
        d_pgd8 = d_pgd_accs[idx8] if idx8 >= 0 else None
        b_fgsm8 = b_fgsm_accs[idx8] if idx8 >= 0 else None
        d_fgsm8 = d_fgsm_accs[idx8] if idx8 >= 0 else None

        w.writerow([
            "Baseline",
            f"{b_clean_acc*100:.2f}",
            f"{(b_pgd8*100 if b_pgd8 is not None else ''):.2f}",
            f"{(b_fgsm8*100 if b_fgsm8 is not None else ''):.2f}"
        ])

        w.writerow([
            "Defence",
            f"{d_clean_acc*100:.2f}",
            f"{(d_pgd8*100 if d_pgd8 is not None else ''):.2f}",
            f"{(d_fgsm8*100 if d_fgsm8 is not None else ''):.2f}"
        ])

    # print final summary
    print("\n= FINAL SUMMARY =")
    print(f"Clean Acc:  Baseline={b_clean_acc*100:.2f}% | Defence={d_clean_acc*100:.2f}%")

    if 8 in eps_x:
        idx8 = eps_x.index(8)
        print(f"FGSM@8/255: Baseline={b_fgsm_accs[idx8]*100:.2f}% | Defence={d_fgsm_accs[idx8]*100:.2f}%")
        print(f"PGD @8/255: Baseline={b_pgd_accs[idx8]*100:.2f}% | Defence={d_pgd_accs[idx8]*100:.2f}%")
        print(f"PGD ASR@8/255 (clean-correct→adv-wrong): Baseline={b_pgd_asr[idx8]*100:.2f}% | Defence={d_pgd_asr[idx8]*100:.2f}%")

    print("\nSaved outputs to:", out_dir)
    print("Key files:")
    print(" - summary.json")
    print(" - headline_results.csv")
    print(" - curve_fgsm_accuracy.png, curve_pgd_accuracy.png")
    print(" - cm_baseline_clean.png, cm_defence_clean.png")
    print(" - cm_baseline_pgd_eps8.png, cm_defence_pgd_eps8.png")
    print(" - per_class_clean.png, per_class_pgd_eps8.png")
    print(" - hist_conf_baseline_clean_vs_pgd8.png, hist_conf_defence_clean_vs_pgd8.png")
    print("===============================================\n")


# start the script
if __name__ == "__main__":
    main()