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

0.8689936536718041

# 6. Current score

0.13752

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18161) has done: 'The fix removes the missing‑module imports, adds a self‑contained data loader, builds a ResNet‑50 model (loading the provided checkpoint if it exists), runs inference on all test images, and writes a correctly‑formatted `submission.csv` whose length matches the test set. This resolves the import and name errors and guarantees a valid Kaggle submission file.'
- What this solution (achieved 0.60277) has done: 'The fix guarantees the submission file contains a row for every JPEG test image by computing the expected length from the dataset rather than the raw directory listing, and it makes the model load any available checkpoint automatically (searching the input folder for a *.pth file if the original path is missing). This resolves the AssertionError and improves model performance, moving the score toward the target while preserving the original architecture and inference flow.'
- What this solution (achieved 0.17152) has done: 'I resize the images to the native 224 × 224 size that the ResNet‑50 expects (instead of 640 × 640) and add a simple test‑time augmentation: for each batch we also run a horizontally‑flipped copy, average the two logits, and take the arg‑max. This modest change usually improves validation accuracy without altering the core model or training logic, moving the score closer to the target.'
- What this solution (achieved 0.10912) has done: 'The updates add all missing imports, define the device and a fallback path for the weights, provide a proper test‑dataset class, create image transforms, build a DataLoader, run inference with a simple horizontal‑flip test‑time augmentation, and finally write a correctly‑formatted `submission.csv`. These fixes resolve the NameError issues, ensure every test image gets a prediction, and produce a valid Kaggle submission, moving the solution from “not yielded” toward the target score.'
- What this solution (achieved 0.13939) has done: 'The patch broadens the checkpoint search to both *.pth and *.pt files, adds a fallback warning, and logs the chosen weight path. This ensures that an available pretrained model is actually loaded during inference, which should raise the validation accuracy substantially and move the score toward the target while leaving the original architecture and inference flow unchanged.'
- What this solution (achieved 0.13528) has done: 'I broaden the checkpoint‑search logic so that a fine‑tuned model (if present) is found in the usual Kaggle input directories *and* in the working directory or current folder. This allow the model to load proper weights instead of falling back to randomly‑initialized ImageNet‑only logits, which should raise the validation accuracy substantially and move the score toward the target while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved 0.34268) has done: 'I tweak the weight‑search utility so it reliably picks the largest *.pth/​*.pt file (most likely the fine‑tuned checkpoint) instead of the first file found. This modest change keeps the original model and inference pipeline intact while dramatically improving the chance of loading a good checkpoint, moving the validation accuracy toward the target score.'
- What this solution (achieved 0.13752) has done: 'I add a lightweight test‑time augmentation that also uses a vertical flip in addition to the existing horizontal flip, averaging the three logits before taking the arg‑max. This small change keeps the original model and loading logic untouched while giving the predictions a modest boost, moving the validation accuracy closer to the target score.'

# 9. Code solution

## === cell 0
import os
import glob
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm.auto import tqdm
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", None)




## === cell 1
def resolve_weights_path(provided_path: str | None) -> str | None:
    """Return a valid checkpoint path.
    1. Use the provided path if it exists.
    2. Otherwise search common Kaggle folders plus the working / current directory
       for the first *.pth or *.pt file – now choosing the *largest* file,
       which is most likely the fine‑tuned checkpoint.
    """
    if provided_path and os.path.exists(provided_path):
        print(f"Using provided weights path: {provided_path}")
        return provided_path

    search_dirs = [
        "/kaggle/input",  # standard Kaggle input root
        "/kaggle/working",  # notebook working directory
        os.getcwd(),  # current working directory
    ]

    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            candidates.extend(glob.glob(os.path.join(d, "**/*.pth"), recursive=True))
            candidates.extend(glob.glob(os.path.join(d, "**/*.pt"), recursive=True))

    if not candidates:
        print(
            "WARNING: No checkpoint file (*.pth or *.pt) found in searched locations."
        )
        return None

    selected = max(candidates, key=lambda p: os.path.getsize(p))
    print(f"Found checkpoint automatically (largest file): {selected}")
    return selected


def load_state_dict(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load checkpoint handling common key variations and possible DataParallel prefixes."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        elif "weights" in state:  # some checkpoints store under 'weights'
            state = state["weights"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def build_model(weights_path: str | None):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5 disease classes
    model = model.to(device)
    resolved_path = resolve_weights_path(weights_path)
    if resolved_path and os.path.exists(resolved_path):
        load_state_dict(model, resolved_path)
    else:
        print(
            "Proceeding with ImageNet‑pretrained weights only (no fine‑tuned checkpoint)."
        )
    model.eval()
    return model


model = build_model(WEIGHTS_PATH)




## === cell 2
class CassavaTestDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        pattern = os.path.join(img_dir, "**", "*.jpg")
        self.filepaths = sorted(glob.glob(pattern, recursive=True))
        self.filenames = [os.path.basename(p) for p in self.filepaths]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        fn = self.filenames[idx]
        img = cv2.imread(img_path)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, fn




## === cell 3
test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 4
possible_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/test_images",
    "./test_images",
]
TEST_IMG_DIR = next((d for d in possible_dirs if os.path.isdir(d)), None)
if TEST_IMG_DIR is None:
    raise RuntimeError("Test image directory not found.")

test_dataset = CassavaTestDataset(TEST_IMG_DIR, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 5
predictions = []
filenames = []

model.to(device)  # ensure model is on the correct device

with torch.no_grad():
    for imgs, fnames in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device)

        logits = model(imgs)

        imgs_hflip = torch.flip(imgs, dims=[3])  # flip width dimension
        logits_hflip = model(imgs_hflip)

        imgs_vflip = torch.flip(imgs, dims=[2])  # flip height dimension
        logits_vflip = model(imgs_vflip)

        avg_logits = (logits + logits_hflip + logits_vflip) / 3.0

        preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
        predictions.extend(preds.tolist())
        filenames.extend(fnames)




## === cell 6
submission = pd.DataFrame({"image_id": filenames, "label": predictions})
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    submission = (
        submission.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()
    )
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
