# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.8682799992482861

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.55797) has done: 'The script was not producing a competitive score because it blended a randomly‑initialized EfficientNet model (the checkpoint is missing) with a simple intensity heuristic.  Relying mainly on the intensity‑based probability gives a more consistent AUC, so the blend weight is changed to favour the heuristic (model weight = 0.0).  The rest of the pipeline is untouched, preserving all original logic while ensuring a valid `submission.csv` is created.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "efficientnet_pytorch"]
    )
    from efficientnet_pytorch import EfficientNet




## === cell 1
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "./input/histopathologic-cancer-detection"

TEST_DIR = os.path.join(DATA_DIR, "test")
if not os.path.isdir(TEST_DIR):
    for root, dirs, _ in os.walk(DATA_DIR):
        if "test" in dirs:
            TEST_DIR = os.path.join(root, "test")
            break

MODEL_PATH = "/kaggle/input/as-week-4-baseline-cnn-training/model_best.pth"
SUBMISSION_FILE = "submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_SIZE = (96, 96)  # same as original script
BATCH_SIZE = 64
NUM_CLASSES = 2  # binary classification




## === cell 2
test_transform = T.Compose(
    [
        T.Resize(TARGET_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class HistologyTestDataset(Dataset):
    def __init__(self, img_ids, img_dir, transform):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_id




## === cell 3
class CancerClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        try:
            self.model = EfficientNet.from_pretrained("efficientnet-b0")
        except Exception as e:
            print(f"EfficientNet pretrained load failed ({e}); using random init.")
            self.model = EfficientNet.from_name("efficientnet-b0")
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


model = CancerClassifier().to(device)

use_model = os.path.exists(MODEL_PATH)
if use_model:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("Model weights loaded.")
else:
    print(
        f"Warning: model checkpoint not found at {MODEL_PATH}. Using ImageNet‑pretrained weights only."
    )

model.eval()
print("Model ready.")




## === cell 4
test_img_ids = [
    fname.split(".")[0]
    for fname in os.listdir(TEST_DIR)
    if fname.lower().endswith(".tif")
]
test_img_ids.sort()  # deterministic order matching typical submission files
print(f"Total test images: {len(test_img_ids)}")

test_dataset = HistologyTestDataset(test_img_ids, TEST_DIR, test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)




## === cell 5
def central_intensity_prob(img_path):
    """
    Simple heuristic: tumor patches tend to be darker.
    Returns a probability in [0,1] based on the mean intensity of the
    32×32 central region after resizing to TARGET_SIZE.
    """
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize(TARGET_SIZE)
    w, h = img.size
    left = (w - 32) // 2
    top = (h - 32) // 2
    central = img.crop((left, top, left + 32, top + 32))
    mean_intensity = np.array(central).mean() / 255.0
    return 1.0 - mean_intensity  # lower intensity → higher tumor probability


def generate_predictions(model, loader, img_dir, model_weight=0.9):
    """
    Blend the EfficientNet model probability with the intensity heuristic.
    A moderate model_weight (e.g., 0.4) lets the pretrained EfficientNet
    contribute useful visual features while still respecting the robust
    intensity heuristic, moving the AUC toward the target.
    """
    model.eval()
    all_ids = []
    all_probs = []

    intensity_weight = 1.0 - model_weight

    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc="Generating predictions"):
            images = images.to(device)

            if model_weight > 0:
                outputs = model(images)  # logits
                model_probs = torch.softmax(outputs, dim=1)[:, 1]  # positive class
            else:
                model_probs = torch.zeros(len(img_ids), device=device)

            batch_intensity = [
                central_intensity_prob(os.path.join(img_dir, f"{img_id}.tif"))
                for img_id in img_ids
            ]
            intensity_probs = torch.tensor(
                batch_intensity, device=device, dtype=torch.float32
            )

            final_probs = (
                model_weight * model_probs + intensity_weight * intensity_probs
            )

            all_probs.extend(final_probs.cpu().numpy())
            all_ids.extend(img_ids)

    return all_ids, all_probs


img_ids, predictions = generate_predictions(
    model, test_loader, TEST_DIR, model_weight=0.4
)




## === cell 6
submission_df = pd.DataFrame({"id": img_ids, "label": predictions})
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file '{SUBMISSION_FILE}' created with {len(submission_df)} rows.")
