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

geopandas==0.14.4
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

0.8655182834693261

# 6. Current score

0.31726

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'We stop moving tensors to CUDA inside the Dataset (which caused a fork‑process CUDA re‑initialisation error) and instead move each batch to the device right before inference. To avoid multiprocessing issues we also set `num_workers=0`. These minimal fixes let the DataLoader run, produce predictions for every test image, and write a correctly‑sized `submission.csv` that matches the expected format.'
- What this solution (achieved 0.10613) has done: 'I make two minimal adjustments: (1) locate the provided baseline weight file more robustly by checking several common Kaggle input paths, and (2) fall back to an ImageNet‑pretrained ResNet‑50 when the weight file is missing, so the model has useful features instead of random initialization. These changes keep the original architecture and inference flow unchanged while ensuring the model loads meaningful parameters, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.13565) has done: 'I expand the search for the pretrained fine‑tuned weight file by adding the most common Kaggle input directories (including the competition folder itself and the working directory). This lets the script load the real checkpoint when it exists, which should raise accuracy dramatically toward the target while keeping all other logic unchanged.'
- What this solution (achieved 0.09567) has done: 'I added a more robust weight‑file discovery step: after checking the original list of expected locations, the script now also scans common Kaggle input directories recursively for any “.pth” files (preferring one that contains “epoch_16” in its name). This ensures the fine‑tuned checkpoint is found when it exists, so the model loads meaningful weights instead of falling back to a raw ImageNet‑pretrained net, which should raise the validation accuracy toward the target score while keeping the original architecture and inference flow unchanged.'
- What this solution (achieved 0.31726) has done: 'I broaden the weight‑file search to include “.pt” files and handle checkpoints that store the model under a “state_dict” key. Then, during inference I apply a simple test‑time augmentation (horizontal flip) and average the logits before taking the arg‑max, which typically raises classification accuracy without altering the core model architecture. These small, targeted tweaks keep the original workflow intact while moving the score markedly closer to the target.'

# 9. Code solution

## === cell 0
import os
import glob
import copy
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms




## === cell 1
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

possible_paths = [
    "/kaggle/input/baseline-weights/epoch_16.pth",
    "/kaggle/input/cassava-leaf-disease-classification/baseline-weights/epoch_16.pth",
    "/kaggle/input/cassava-leaf-disease-classification/epoch_16.pth",
    "/kaggle/input/epoch_16.pth",
    "../input/baseline-weights/epoch_16.pth",
    "../input/cassava-leaf-disease-classification/baseline-weights/epoch_16.pth",
    "../input/cassava-leaf-disease-classification/epoch_16.pth",
    "baseline-weights/epoch_16.pth",
    "input/baseline-weights/epoch_16.pth",
    "epoch_16.pth",
    "weights/epoch_16.pth",
    "/kaggle/working/epoch_16.pth",
]
WEIGHTS_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        WEIGHTS_PATH = p
        break

if WEIGHTS_PATH is None:
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
        "../input",
        "./",
    ]
    candidate_files = []
    for root in search_roots:
        if os.path.isdir(root):
            candidate_files.extend(
                glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
            )
    candidate_files = [
        c for c in candidate_files if c.lower().endswith((".pth", ".pt"))
    ]
    for cand in candidate_files:
        if "epoch_16" in os.path.basename(cand):
            WEIGHTS_PATH = cand
            break
    if WEIGHTS_PATH is None and candidate_files:
        WEIGHTS_PATH = candidate_files[0]

if WEIGHTS_PATH is None:
    print("Warning: baseline weight file not found in any expected location.")
else:
    print(f"Using weight file: {WEIGHTS_PATH}")

TEST_IMG_DIR = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def build_model(num_classes=NUM_CLASSES, pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


model = build_model(pretrained=True).to(DEVICE)

if WEIGHTS_PATH is not None:
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    model_state = model.state_dict()
    filtered_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    if filtered_dict:
        model_state.update(filtered_dict)
        model.load_state_dict(model_state)
    else:
        print("Warning: No matching keys found in checkpoint; using ImageNet weights.")
else:
    print(
        "Proceeding with ImageNet‑pretrained ResNet‑50 (no fine‑tuned weights loaded)."
    )
model.eval()




## === cell 3
val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 4
class InferDataset(Dataset):
    def __init__(self, img_dir, transform=None, device=DEVICE):
        self.img_dir = img_dir
        self.transform = transform
        self.device = device
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, img_name




## === cell 5
infer_dataset = InferDataset(TEST_IMG_DIR, transform=val_transform, device=DEVICE)
infer_loader = DataLoader(infer_dataset, batch_size=32, shuffle=False, num_workers=0)




## === cell 6
submission_path = "./submission.csv"
with open(submission_path, "w") as f:
    f.write("image_id,label\n")
    with torch.no_grad():
        for imgs, fnames in infer_loader:
            imgs = imgs.to(DEVICE)
            imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
            logits = model(imgs)
            logits_flip = model(imgs_flipped)
            avg_logits = (logits + logits_flip) / 2.0
            preds = avg_logits.argmax(dim=1).cpu().numpy()
            for pred, fn in zip(preds, fnames):
                f.write(f"{fn},{int(pred)}\n")
print(f"Submission written to {submission_path}")




## === cell 7
df = pd.read_csv(submission_path)
print(
    f"Rows (including header): {len(df)}  | Expected test images: {len(infer_dataset)}"
)
