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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8203384708371109

# 6. Current score

0.18498

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.213) has done: 'We replace the deprecated `DataFrame.append` with a list‑accumulation approach, building the submission DataFrame after inference. This fixes the runtime error and ensures a valid `submission.csv` is written while keeping the original model logic unchanged.'
- What this solution (achieved 0.10463) has done: 'I add a robust weight‑loading step that automatically finds a fine‑tuned *.pt file under the input directory when the hard‑coded path is missing, and load it with `strict=False` to avoid key‑mismatch errors. This should switch the model from plain ImageNet weights (which gave ~0.21 accuracy) to the competition‑specific fine‑tuned weights, moving the score much closer to the target while keeping the original architecture and inference logic unchanged.'
- What this solution (achieved 0.34865) has done: 'I make two minimal but effective tweaks: (1) improve the weight‑loading logic so the fine‑tuned ViT checkpoint is reliably found and loaded (searching for files that contain “CassavaViT” and confirming the head is restored), and (2) add a simple test‑time augmentation – horizontal flip – and average the logits before taking the arg‑max. These changes keep the original model and pipeline unchanged while expectedly raising the validation accuracy toward the target score.'
- What this solution (achieved 0.36809) has done: 'I make two small, safe adjustments: (1) improve the weight‑loading routine by stripping a possible “module.” prefix from checkpoint keys so the fine‑tuned weights are fully applied, and (2) add a vertical‑flip test‑time augmentation and average its logits with the original and horizontal‑flip predictions, which modestly boosts validation accuracy without altering the core model or training logic. These changes keep the overall pipeline identical while moving the score nearer to the target.'
- What this solution (achieved 0.28774) has done: 'I tighten the checkpoint loading so the fine‑tuned weights are correctly extracted even when they are stored under a nested “model” or “state_dict” key, and I strip any “module.” prefix. This ensures the classifier head is populated, which should raise accuracy substantially toward the target. Additionally, I add a simple 90° rotation test‑time augmentation and average its logits with the existing three predictions, giving a modest boost without changing the core model or training logic. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.13416) has done: 'We make the weight‑loading logic more robust so the fine‑tuned checkpoint is actually applied (fallback to any “CassavaViT” .pt file and strip possible prefixes).  
Then we average *probabilities* from several test‑time augmentations rather than raw logits, which typically yields a modest but reliable gain in classification accuracy while keeping the model architecture unchanged.'
- What this solution (achieved 0.0938) has done: 'I make the weight‑loading step more robust so that the fine‑tuned checkpoint’s classifier head is correctly applied (otherwise the model predicts almost randomly, giving a very low score). After extracting the state dict I check for missing head keys and, if the original checkpoint contains a separate “head” sub‑dictionary, I merge those weights before loading. This small change keeps the overall architecture and inference unchanged while allowing the model to use the proper fine‑tuned head, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.18498) has done: 'The fix makes the weight‑loading step robust: it searches all possible input directories (including the Kaggle default “/kaggle/input”), picks the largest *.pt file that contains “CassavaViT”, and correctly extracts the model‑state dict (handling possible prefixes and nested “head” entries). This ensures the fine‑tuned ViT weights—especially the classifier head—are actually applied, which should raise the validation accuracy toward the target score. No other logic or model architecture is changed.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



## === cell 1
import timm



## === cell 2
print("Available ViT Models (short list):")
print(timm.list_models("vit*")[:5])



## === cell 3
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
Cassava_model_path = (
    "../input/cassavaaugmtp98epochs4lr175/CassavaViT_Augm_TP98_Epochs4_LR1-75e05.pt"
)




## === cell 4
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cassava_model = ViTBase16(n_classes=5, pretrained=True)

load_path = None
if os.path.isfile(Cassava_model_path):
    load_path = Cassava_model_path
else:
    import glob

    search_patterns = [
        "/kaggle/input/**/*.pt",  # Kaggle default mount point
        "../input/**/*.pt",  # relative path from notebook root
        "./**/*.pt",  # fallback to current directory
    ]
    candidates = []
    for pat in search_patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    candidates = [p for p in candidates if "CassavaViT" in os.path.basename(p)]
    if candidates:
        load_path = max(candidates, key=os.path.getsize)
        print(
            f"Fine‑tuned weights not found at the default location. Using fallback: {load_path}"
        )
    else:
        print(
            "No fine‑tuned .pt weights found; proceeding with ImageNet‑pretrained model."
        )

if load_path is not None:
    raw_state = torch.load(load_path, map_location=device)

    if isinstance(raw_state, dict):
        if "model" in raw_state and isinstance(raw_state["model"], dict):
            raw_state = raw_state["model"]
        elif "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
            raw_state = raw_state["state_dict"]

    state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}

    if not any(k.startswith("head.") for k in state_dict.keys()):
        head_subdict = raw_state.get("head")
        if isinstance(head_subdict, dict):
            for hk, hv in head_subdict.items():
                state_dict[f"head.{hk}"] = hv

    missing, unexpected = cassava_model.load_state_dict(state_dict, strict=False)

    if any("head." in k for k in missing):
        print(
            "Warning: classifier head weights still missing after loading; using random head."
        )
    else:
        print("Fine‑tuned weights loaded successfully.")
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

cassava_model = cassava_model.to(device)
cassava_model.eval()
torch.backends.cudnn.benchmark = True  # minor speed boost without affecting accuracy




## === cell 5
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )
        print(f"Cassava Disease Test Dataset Length = {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name




## === cell 6
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 7
testset = TestSet2(test_dir=test_path, transform=test_transform)
test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=False)



## === cell 8
for images, names in test_loader:
    im = make_grid(images, nrow=1)
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.title(names[0])
    plt.axis("off")
    plt.show()
    break



## === cell 9
import torch.nn.functional as F

col_names = ["image_id", "label"]
rows = []
tic = time.time()
with torch.no_grad():
    for X_test, img_name in test_loader:
        X_test = X_test.to(device)

        prob_orig = F.softmax(cassava_model(X_test), dim=1)

        prob_h = F.softmax(cassava_model(torch.flip(X_test, dims=[-1])), dim=1)

        prob_v = F.softmax(cassava_model(torch.flip(X_test, dims=[-2])), dim=1)

        prob_r90 = F.softmax(
            cassava_model(torch.rot90(X_test, k=1, dims=[-2, -1])), dim=1
        )

        prob_avg = (prob_orig + prob_h + prob_v + prob_r90) / 4.0

        pred = torch.argmax(prob_avg, dim=1).item()
        rows.append({"image_id": img_name[0], "label": int(pred)})

submission_df = pd.DataFrame(rows, columns=col_names)

print(f"Inference time: {time.time() - tic:.2f}s")
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
print(submission_df.head())
