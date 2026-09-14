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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8912058023572076

# 6. Current score

0.10314

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I filter the test image list to only JPEG files (using the provided sample submission order) and adjust the transform builder so that `RandomResizedCrop` receives the correct height‑width arguments instead of a single integer. These fixes remove the Albumentations validation error and ensure the generated CSV has the exact number of rows required for a valid submission.'
- What this solution (achieved 0.61099) has done: 'I fix the Albumentations `RandomResizedCrop` error and remove stochastic augmentations during inference, replacing it with a deterministic `Resize` + `Normalize`. This resolves the validation exception, ensures the model receives correctly‑sized inputs, and yields a stable prediction that can reach the target accuracy.'
- What this solution (achieved 0.14686) has done: 'I enable ImageNet pretrained weights for EfficientNet‑B7 (the model was previously loaded with random weights) and add a simple deterministic test‑time augmentation: for each batch we also evaluate the horizontally‑flipped images and average the logits with the original predictions. These tiny changes keep the original architecture and training logic intact while providing a more informative model and a modest boost in validation accuracy, moving the score closer to the target.'
- What this solution (achieved 0.14686) has done: 'I adjust the model‑loading logic so that the fine‑tuned EfficientNet‑B7 checkpoint is correctly found in the Kaggle `/kaggle/input` directory (the original relative path often points to a non‑existent location). By loading the proper checkpoint the model’s learned weights for the five disease classes are restored, which should raise the validation accuracy substantially and move the score toward the target. No other parts of the pipeline are changed, preserving the original architecture, training‑free inference flow, and deterministic preprocessing.'
- What this solution (achieved 0.18124) has done: 'We eliminate the unnecessary ten‑fold inference loop: the model does not change between epochs, so averaging ten identical prediction sets is redundant. By running inference only once and scaling the result, we keep exactly the same final predictions while cutting the runtime dramatically. The rest of the pipeline, model architecture, and augmentation logic remain unchanged.'
- What this solution (achieved 0.18124) has done: 'I fix the checkpoint‑loading logic so the fine‑tuned EfficientNet‑B7 weights are actually found (adding the common Kaggle input path under the competition folder). This lets the model use its learned parameters instead of only ImageNet weights, which should raise the validation accuracy substantially toward the target. I also keep the deterministic inference pipeline unchanged.'
- What this solution (achieved 0.07623) has done: 'The failure occurs because the dataset returns `torch.DoubleTensor` (float64) while the EfficientNet‑B7 weights are `float32`. We fix this by defining `mean` and `std` as `float32` and explicitly casting the normalized image back to `float32` before converting to a tensor. This restores type compatibility, allowing inference to run and producing a correct `submission.csv` file.'
- What this solution (achieved 0.1719) has done: 'I simplify the inference step by removing the deterministic test‑time augmentations (horizontal flip, vertical flip, rotation) that were hurting accuracy when only ImageNet‑pretrained weights are available, and keep only the original forward pass. This minimal change preserves the overall pipeline and model loading logic while expected to move the validation accuracy closer to the target score.'
- What this solution (achieved 0.12033) has done: 'I make two minimal adjustments: (1) improve checkpoint discovery so a fine‑tuned EfficientNet‑B7 model is loaded if it exists, and (2) add a cheap deterministic test‑time augmentation (horizontal flip) and average its logits with the original prediction. Loading the proper checkpoint and a small TTA should raise validation accuracy toward the target without altering the core architecture or training pipeline.'
- What this solution (achieved 0.10314) has done: 'I make the model‑loading logic robust so that the fine‑tuned EfficientNet‑B7 checkpoint is actually found and loaded (instead of falling back to ImageNet weights only). The patch adds a prioritized absolute path, a more flexible state‑dict extraction (handling keys like “model”, “state_dict”, or a raw dict), and loads with `strict=False` to avoid mismatch errors. This change preserves the original architecture and inference pipeline while substantially improving the validation accuracy, moving the score much closer to the target.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b7
import cv2
from tqdm import tqdm

image_size = 600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    seed=22,
    experiment_name="enb7",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/cassava-leaf-disease-classification/savedmodels2",
    checkpoint="enb7best.pt",
    model="efficientnet-b7",
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(name="Resize", params=dict(height=image_size, width=image_size)),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
)




## === cell 1
def load_model():
    model = efficientnet_b7(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

    possible_paths = [
        os.path.join(
            "/kaggle/input",
            "cassava-leaf-disease-classification",
            "savedmodels2",
            config["checkpoint"],
        ),
        os.path.join(config["checkpoint_path"], config["checkpoint"]),
        os.path.join("/kaggle/input", "savedmodels2", config["checkpoint"]),
        os.path.join(
            "/kaggle/input",
            "cassava-leaf-disease-classification",
            "savedmodels2",
            config["checkpoint"],
        ),
    ]

    ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if ckpt_path is None:
        for root, _, files in os.walk("/kaggle/input"):
            if config["checkpoint"] in files:
                ckpt_path = os.path.join(root, config["checkpoint"])
                break
            for f in files:
                if f.lower().endswith((".pt", ".pth")):
                    ckpt_path = os.path.join(root, f)
                    break
            if ckpt_path:
                break

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict")
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print("Warning: strict loading failed, attempting non‑strict load:", e)
            model.load_state_dict(state_dict, strict=False)

        epoch = (
            checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"
        )
        train_loss = (
            checkpoint.get("train_loss", "N/A")
            if isinstance(checkpoint, dict)
            else "N/A"
        )
        val_loss = (
            checkpoint.get("val_loss", "N/A") if isinstance(checkpoint, dict) else "N/A"
        )
        metrics = (
            checkpoint.get("metrics", "N/A") if isinstance(checkpoint, dict) else "N/A"
        )
        lr = checkpoint.get("lr", "N/A") if isinstance(checkpoint, dict) else "N/A"
        print("Epoch:", epoch)
        print("Train loss:", train_loss)
        print("Validation loss:", val_loss)
        print("Accuracy (saved):", metrics)
        print("Learning rate:", lr)
    else:
        print("Checkpoint not found, using pretrained ImageNet weights only.")

    model.eval()
    return model.to(device)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, img_size):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float()


def get_dataloader():
    sample_path = os.path.join(
        "/kaggle/input",
        "cassava-leaf-disease-classification",
        "sample_submission.csv",
    )
    if not os.path.exists(sample_path):
        sample_path = os.path.join(
            "../input/cassava-leaf-disease-classification",
            "sample_submission.csv",
        )
    sample_df = pd.read_csv(sample_path)
    test_ids = sample_df["image_id"].tolist()

    global test
    test = sample_df.copy()

    img_dir = os.path.abspath(config["test_location"])
    if not os.path.isdir(img_dir):
        img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

    dataset = TestDataset(
        image_ids=test_ids,
        img_dir=img_dir,
        img_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=True,
    )
    return loader




## === cell 3
def infer(model, dataloader):
    """
    Perform inference with a cheap deterministic test‑time augmentation:
    average predictions from the original image and its horizontal flip.
    This simple TTA often improves accuracy without changing the core model.
    """
    print("Running inference with horizontal‑flip TTA...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            logits = model(batch)

            flipped = torch.flip(batch, dims=[-1])  # flip width dimension
            logits_flip = model(flipped)

            avg_logits = (logits + logits_flip) / 2.0
            predictions.append(avg_logits.cpu())

    return torch.cat(predictions, dim=0)




## === cell 4
if __name__ == "__main__":
    torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    print("Inferring experiment", config["experiment_name"])

    start_time = time.time()
    predictions = infer(model, dataloader)
    print("Inference time:", time.time() - start_time)

    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)

    submission_path = "submission.csv"
    test.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
