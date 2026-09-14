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

0.7234814143245694

# 6. Current score

0.17825

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1491) has done: 'I make the script robust by handling missing model weight files, falling back to ImageNet‑pretrained models, and ensuring the prediction loop always produces a label for every test image listed in the sample submission. This eliminates the earlier `FileNotFoundError` and `NameError`, guarantees the submission length matches the expected number of rows, and writes a proper `submission.csv` file.'
- What this solution (achieved 0.14761) has done: 'I add proper image normalization (scale to 0‑1 and apply ImageNet mean/std) in the preprocessing step, because the models were trained on normalized inputs; this small change can markedly raise the classification accuracy without altering the core architecture or training logic. The rest of the pipeline stays the same, so the script still produces a valid `submission.csv` while moving the score closer to the target.'
- What this solution (achieved 0.11398) has done: 'I add lightweight test‑time augmentation (horizontal flip and 90° rotation) and average the raw logits from both models over these augmentations before applying soft‑max. This keeps the original architectures and inference flow unchanged, while providing a modest boost in predictive accuracy that moves the current score toward the target. The rest of the pipeline (loading, preprocessing, submission) remains identical.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight fallback that predicts the most frequent class from the training labels when the ensemble weights are missing (the original models otherwise use random heads and give very low accuracy). This change keeps the existing model‑based pipeline unchanged when valid weight files are present, but when they are absent it replaces the essentially random predictions with a sensible majority‑class guess, which substantially raises the expected accuracy and moves the score toward the target. The rest of the script (pre‑processing, TTA, CSV writing) remains the same.'
- What this solution (achieved 0.19096) has done: 'The fix corrects the ImageNet normalization tensors to have shape (3, 1, 1) so they broadcast properly over the image dimensions, eliminating the RuntimeError during preprocessing. With this change the preprocessing runs, the inference loop populates `names` and `labels`, and the final cells successfully write a valid `submission.csv` file.'
- What this solution (achieved 0.17825) has done: 'I lower the confidence threshold to 0 so the model’s predicted label is always used (no fallback to the majority class). This tiny change keeps the core pipeline unchanged while allowing more model‑derived predictions, which should raise the accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import glob
import cv2
import torch
import timm
import pandas as pd
import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_model(arch):
    """Load an ImageNet‑pretrained model for the given architecture."""
    model = timm.create_model(arch, pretrained=True, num_classes=5)
    model.to(device).eval()
    return model


efficient = load_model("tf_efficientnet_b5")
hrnet = load_model("hrnet_w48")

print("Pretrained models loaded (fallback weights ignored).")
use_ensemble = True  # force ensemble inference




## === cell 1
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_sub_path)
test_ids = sample_df["image_id"].tolist()

test_dir = "../input/cassava-leaf-disease-classification/test_images"

imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)


def preprocess_image(image):
    """Resize, apply CLAHE, convert to torch tensor, scale to [0,1] and normalize.
    Returns a (3, 384, 384) tensor on the target device (no batch dim)."""
    img = cv2.resize(image, (384, 384))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float, device=device)
    tensor = tensor / 255.0
    tensor = (tensor - imagenet_mean) / imagenet_std
    return tensor


def tta_augmentations(batch):
    """Generate the five TTA versions for a batch tensor (B,3,384,384)."""
    aug = [batch]  # original
    aug.append(torch.flip(batch, dims=[3]))  # horizontal flip
    aug.append(torch.flip(batch, dims=[2]))  # vertical flip
    aug.append(torch.rot90(batch, k=1, dims=[2, 3]))  # 90° rotation
    aug.append(torch.rot90(batch, k=2, dims=[2, 3]))  # 180° rotation
    return aug


train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    majority_label = int(train_df["label"].mode()[0])
else:
    majority_label = 0
print(f"Majority class (fallback) label: {majority_label}")

confidence_threshold = 0.0  # previously 0.4

base_tensors = []
valid_ids = []
for img_id in tqdm.tqdm(test_ids, desc="Loading & preprocessing"):
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    if img is None:
        tensor = torch.zeros((3, 384, 384), dtype=torch.float, device=device)
    else:
        tensor = preprocess_image(img)
    base_tensors.append(tensor)
    valid_ids.append(img_id)

base_batch = torch.stack(base_tensors)  # shape (N,3,384,384)

batch_size = 32
names, labels = [], []

with torch.no_grad():
    for start in range(0, len(valid_ids), batch_size):
        end = min(start + batch_size, len(valid_ids))
        batch_ids = valid_ids[start:end]
        batch_tensor = base_batch[start:end]  # (B,3,384,384)

        total_logits = torch.zeros((batch_tensor.size(0), 5), device=device)
        for aug_tensor in tta_augmentations(batch_tensor):
            hr_out = hrnet(aug_tensor)
            eff_out = efficient(aug_tensor)
            total_logits += (hr_out + eff_out) / 2.0

        avg_logits = total_logits / 5.0
        prob = F.softmax(avg_logits, dim=1)
        max_conf, pred = prob.max(dim=1)

        for i, img_id in enumerate(batch_ids):
            if max_conf[i].item() < confidence_threshold:
                pred_label = majority_label
            else:
                pred_label = int(pred[i].item())
            names.append(img_id)
            labels.append(pred_label)




## === cell 2
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, header=True)
print(f"Submission saved to {submission_path} (rows: {len(submission)})")
