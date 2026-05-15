import os
import subprocess
import sys


# run a python script without extra arguments
def run_script(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        return

    print(f"\n[RUNNING] {path}\n")
    subprocess.run([sys.executable, path])
    input("\nPress Enter to return to menu...")


# run a python script with extra arguments
def run_script_with_args(path, *args):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        return

    cmd = [sys.executable, path] + list(args)

    print(f"\n[RUNNING] {' '.join(cmd)}\n")

    subprocess.run(cmd)

    input("\nPress Enter to return to menu...")


# let the user choose an image from the images folder
def choose_image():
    folder = "images"

    if not os.path.exists(folder):
        print("\n[ERROR] 'images/' folder not found.")
        print("Create a folder called 'images' and place your test images inside it.")
        print("Note: the images folder is ignored by Git, so users must create it locally.")
        return None

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not files:
        print("\n[ERROR] No images found in 'images/' folder.")
        print("Add images such as .jpg, .jpeg, .png, or .webp files.")
        return None

    print("\nAvailable images:")
    print("-" * 30)

    for i, file in enumerate(files, start=1):
        print(f"{i}. {file}")

    choice = input("\nSelect image number: ").strip()

    try:
        index = int(choice) - 1

        if index < 0 or index >= len(files):
            print("\nInvalid image selection.")
            return None

        return os.path.join(folder, files[index])

    except ValueError:
        print("\nInvalid input. Please enter a number.")
        return None


# let the user choose a saved custom model
def choose_custom_model():
    folder = "custom_train"

    if not os.path.exists(folder):
        print("\n[ERROR] 'custom_train/' folder not found.")
        print("Train a custom model first.")
        return None

    models = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".pt")
    ]

    if not models:
        print("\n[ERROR] No custom models found in 'custom_train/'.")
        print("Train a custom model first.")
        return None

    print("\nAvailable custom models:")
    print("-" * 30)

    for i, model in enumerate(models, start=1):
        print(f"{i}. {model}")

    choice = input("\nSelect model number: ").strip()

    try:
        index = int(choice) - 1

        if index < 0 or index >= len(models):
            print("\nInvalid model selection.")
            return None

        return os.path.join(folder, models[index])

    except ValueError:
        print("\nInvalid input. Please enter a number.")
        return None


# get epsilon value from the user
def choose_epsilon():
    while True:
        value = input("\nEnter epsilon value (0-255): ").strip()

        try:
            eps = float(value)

            if eps < 0 or eps > 255:
                print("\n[ERROR] Enter a value between 0 and 255.")
                continue

            return str(eps)

        except ValueError:
            print("\n[ERROR] Invalid epsilon value.")


# get a number from the user with optional limits
def choose_number(prompt, default=None, min_value=None, max_value=None, number_type=float):
    while True:
        default_text = f" [default: {default}]" if default is not None else ""
        value = input(f"{prompt}{default_text}: ").strip()

        if value == "" and default is not None:
            return default

        try:
            num = number_type(value)

            if min_value is not None and num < min_value:
                print(f"\n[ERROR] Minimum value is {min_value}.")
                continue

            if max_value is not None and num > max_value:
                print(f"\n[ERROR] Maximum value is {max_value}.")
                continue

            return num

        except ValueError:
            print("\n[ERROR] Invalid value.")


# get the name for a new model
def choose_model_name():
    while True:
        name = input("\nEnter model name: ").strip()

        if name:
            return name

        print("\n[ERROR] Model name cannot be empty.")


# confirm before starting training
def confirm_training():
    print("\nTraining may take a long time and may overwrite existing custom models with the same name.")
    answer = input("Continue? (y/n): ").strip().lower()
    return answer == "y"


# run a script after selecting an image
def run_with_selected_image(script_path):
    img_path = choose_image()

    if img_path:
        run_script_with_args(script_path, img_path)


# run a script after selecting an image and epsilon
def run_with_image_and_eps(script_path):
    img_path = choose_image()

    if not img_path:
        return

    eps = choose_epsilon()

    run_script_with_args(script_path, img_path, eps)


# run clean prediction using a custom model
def run_custom_clean_prediction():
    model_path = choose_custom_model()

    if not model_path:
        return

    img_path = choose_image()

    if not img_path:
        return

    run_script_with_args("predict_image.py", img_path, model_path)


# run attack or defence visualisation using a custom model
def run_custom_visual_with_eps(script_path):
    model_path = choose_custom_model()

    if not model_path:
        return

    img_path = choose_image()

    if not img_path:
        return

    eps = choose_epsilon()

    run_script_with_args(script_path, img_path, eps, model_path)


# display a menu and return the user choice
def menu(title, options):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    if title == "ADVERSARIAL MACHINE LEARNING SYSTEM":
        print(
            "Reminder: create an 'images' folder and add the images "
            "you want to test. This folder is ignored by Git."
        )
        print("-" * 50)

    for key, value in options.items():
        print(f"{key}. {value}")

    return input("\nSelect an option: ").strip()


# pre-trained model menu
def pretrained_menu():
    while True:
        choice = menu("USE PRE-TRAINED MODEL", {
            "1": "Clean prediction",
            "2": "Visualisations",
            "3": "Full evaluation report",
            "0": "Back"
        })

        if choice == "1":
            run_with_selected_image("predict_image.py")

        elif choice == "2":
            visual_menu()

        elif choice == "3":
            run_script("scripts/eval/eval_full_report.py")

        elif choice == "0":
            break

        else:
            print("Invalid choice.")


# custom model menu
def custom_model_menu():
    while True:
        choice = menu("USE CUSTOM-TRAINED MODEL", {
            "1": "Clean prediction",
            "2": "FGSM attack visualisation",
            "3": "PGD attack visualisation",
            "4": "Multiple image predictions",
            "0": "Back"
        })

        if choice == "1":
            run_custom_clean_prediction()

        elif choice == "2":
            run_custom_visual_with_eps("scripts/visual/fgsm_visualise.py")

        elif choice == "3":
            run_custom_visual_with_eps("scripts/visual/pgd_visual.py")

        elif choice == "4":
            model_path = choose_custom_model()

            if not model_path:
                continue

            img_path = choose_image()

            if not img_path:
                continue

            eps = choose_epsilon()

            run_script_with_args(
                "scripts/visual/multiple_image_pred.py",
                img_path,
                eps,
                model_path
            )

        elif choice == "0":
            break

        else:
            print("Invalid choice.")


# visualisation menu
def visual_menu():
    while True:
        choice = menu("VISUALISATIONS", {
            "1": "FGSM attack visualisation",
            "2": "PGD attack visualisation",
            "3": "FGSM defence visualisation",
            "4": "PGD defence visualisation",
            "5": "Multiple image predictions",
            "0": "Back"
        })

        if choice == "1":
            run_with_image_and_eps("scripts/visual/fgsm_visualise.py")

        elif choice == "2":
            run_with_image_and_eps("scripts/visual/pgd_visual.py")

        elif choice == "3":
            run_with_image_and_eps("scripts/visual/defences/fgsmadv_visual.py")

        elif choice == "4":
            run_with_image_and_eps("scripts/visual/defences/pgdadv_visual.py")

        elif choice == "5":
            run_with_image_and_eps("scripts/visual/multiple_image_pred.py")

        elif choice == "0":
            break

        else:
            print("Invalid choice.")


# training menu
def train_menu():
    while True:
        choice = menu("TRAIN NEW MODEL", {
            "1": "Train baseline CNN",
            "2": "Train FGSM adversarial model",
            "3": "Train PGD adversarial model",
            "0": "Back"
        })

        if choice == "0":
            break

        if choice not in ["1", "2", "3"]:
            print("Invalid choice.")
            continue

        # get common training settings
        model_name = choose_model_name()

        epochs = choose_number(
            "Epochs",
            default=20,
            min_value=1,
            number_type=int
        )

        batch_size = choose_number(
            "Batch size",
            default=128,
            min_value=1,
            number_type=int
        )

        lr = choose_number(
            "Learning rate",
            default=0.001,
            min_value=0.000001,
            number_type=float
        )

        # train normal baseline model
        if choice == "1":
            if confirm_training():
                run_script_with_args(
                    "scripts/train/train_cifar10_cnn.py",
                    "--name", model_name,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr)
                )

        # train fgsm adversarial model
        elif choice == "2":
            eps = choose_epsilon()

            adv_ratio = choose_number(
                "Adversarial ratio (0-1)",
                default=0.5,
                min_value=0,
                max_value=1,
                number_type=float
            )

            if confirm_training():
                run_script_with_args(
                    "scripts/train/train_fgsm_adv.py",
                    "--name", model_name,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr),
                    "--eps", str(eps),
                    "--adv-ratio", str(adv_ratio)
                )

        # train pgd adversarial model
        elif choice == "3":
            eps = choose_epsilon()

            alpha = choose_number(
                "Alpha value (0-255)",
                default=2,
                min_value=0,
                max_value=255,
                number_type=float
            )

            steps = choose_number(
                "PGD steps",
                default=7,
                min_value=1,
                number_type=int
            )

            adv_weight = choose_number(
                "Adversarial weight (0-1)",
                default=0.5,
                min_value=0,
                max_value=1,
                number_type=float
            )

            if confirm_training():
                run_script_with_args(
                    "scripts/train/train_pgd_adv.py",
                    "--name", model_name,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr),
                    "--eps", str(eps),
                    "--alpha", str(alpha),
                    "--steps", str(steps),
                    "--adv-weight", str(adv_weight)
                )


# main program menu
def main():
    while True:
        choice = menu("ADVERSARIAL MACHINE LEARNING SYSTEM", {
            "1": "Use pre-trained model",
            "2": "Use custom-trained model",
            "3": "Train a new model",
            "0": "Exit"
        })

        if choice == "1":
            pretrained_menu()

        elif choice == "2":
            custom_model_menu()

        elif choice == "3":
            train_menu()

        elif choice == "0":
            print("\nExiting...")
            break

        else:
            print("Invalid choice.")


# start the program
if __name__ == "__main__":
    main()