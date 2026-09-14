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

3.13

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

0.8301601692354186

# 6. Current score

0.60912

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43572) has done: 'The fixes address the import error (`torchvision.v2` does not exist), replace the unavailable transforms with standard torchvision ones, and ensure the submission CSV has exactly the same number of rows as the official test set by iterating over the `sample_submission.csv` image list. These changes keep the model architecture unchanged while making the pipeline runnable and producing a valid submission file.'
- What this solution (achieved 0.26121) has done: 'The fix updates the image size so both EfficientNet and ViT receive the correct dimensions, uses a valid fallback weight enum for ViT, and ensures the inference loop runs to completion, producing a submission CSV with matching lengths.'
- What this solution (achieved 0.2571) has done: 'I enable loading of the provided fine‑tuned checkpoints without strict key matching and use model‑specific image sizes (EfficientNet 480 px, ViT 518 px) so each model receives inputs at the resolution it was trained on. These minimal changes keep the architecture unchanged while likely improving predictions, moving the validation accuracy from ~0.26 toward the target 0.83.'
- What this solution (achieved 0.12967) has done: 'The update speeds up inference by loading more images per batch, keeping DataLoader workers alive, and using a small GPU‑autocast block (which does not change the arg‑max predictions). No model architecture or training logic is altered, and all file paths and transforms remain the same.'
- What this solution (achieved 0.12967) has done: 'I added a safe fallback that searches for the EfficientNet and ViT checkpoint files when the hard‑coded paths are missing, ensuring the fine‑tuned weights are actually loaded instead of falling back to ImageNet weights. This small change keeps the model architecture untouched while allowing the ensemble to use the intended trained parameters, which should raise the validation accuracy toward the target score. The rest of the pipeline remains unchanged, and the script still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.60912) has done: 'I add a class‑frequency prior calculated from the training labels and bias the averaged logits with the log‑prior before taking the argmax. This small, non‑intrusive adjustment keeps the model architecture unchanged while nudging predictions toward the more common classes, which should raise the accuracy toward the target score. The rest of the pipeline remains the same, and a valid `submission.csv` is still written.'

# 9. Code solution

## === cell 0
import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_data_directory):
    test_data_directory = os.path.join(os.getcwd(), "test_images")
    if not os.path.isdir(test_data_directory):
        raise FileNotFoundError(
            f"Test images directory not found at '{test_data_directory}'"
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/3/vit_h_14_518_8927.pth"

if not os.path.isfile(en_model_path):
    en_candidates = glob.glob("/kaggle/input/**/efficientnet*/*.pth", recursive=True)
    if en_candidates:
        en_model_path = en_candidates[0]
        print(f"[Info] EfficientNet checkpoint found at: {en_model_path}")
    else:
        print(
            "[Warning] EfficientNet checkpoint not found; will fall back to ImageNet weights."
        )

if not os.path.isfile(vit_model_path):
    vit_candidates = glob.glob("/kaggle/input/**/vit*/*.pth", recursive=True)
    if vit_candidates:
        vit_model_path = vit_candidates[0]
        print(f"[Info] ViT checkpoint found at: {vit_model_path}")
    else:
        print("[Warning] ViT checkpoint not found; will fall back to ImageNet weights.")

en_image_size = 480
vit_image_size = 518




## === cell 2
en_transforms = transforms.Compose(
    [
        transforms.Resize((en_image_size, en_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

vit_transforms = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)




## === cell 3
try:
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    en_state = torch.load(en_model_path, map_location=device, weights_only=True)
    en_model.load_state_dict(en_state, strict=False)  # allow missing classifier keys
except Exception as e:
    print(f"EfficientNet custom checkpoint not loaded ({e}); using ImageNet weights.")
    en_model = models.efficientnet_v2_l(
        weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    )
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
en_model.to(device)
en_model.eval()

try:
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
    vit_state = torch.load(vit_model_path, map_location=device, weights_only=True)
    vit_model.load_state_dict(vit_state, strict=False)  # allow missing classifier keys
except Exception as e:
    print(f"ViT custom checkpoint not loaded ({e}); using ImageNet weights.")
    try:
        vit_weights = models.ViT_H_14_Weights.DEFAULT
    except AttributeError:
        vit_weights = None
    vit_model = models.vit_h_14(weights=vit_weights, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
vit_model.to(device)
vit_model.eval()




## === cell 4
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    sample_sub_path = os.path.join(os.getcwd(), "sample_submission.csv")
    if not os.path.isfile(sample_sub_path):
        raise FileNotFoundError(
            f"sample_submission.csv not found at '{sample_sub_path}'"
        )
sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if not os.path.isfile(train_csv_path):
    train_csv_path = os.path.join(os.getcwd(), "train.csv")
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at '{train_csv_path}'")
train_df = pd.read_csv(train_csv_path)
class_counts = train_df["label"].value_counts().sort_index()
for c in range(num_classes):
    if c not in class_counts:
        class_counts.loc[c] = 0
class_counts = class_counts.sort_index()
prior = class_counts.values.astype("float32")
prior = prior / prior.sum()
prior_tensor = torch.from_numpy(prior).to(device)




## === cell 5
valid_paths = []
valid_idxs = []
for idx, image_name in enumerate(image_ids):
    image_path = os.path.join(test_data_directory, image_name)
    if os.path.isfile(image_path):
        valid_paths.append(image_path)
        valid_idxs.append(idx)

from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, paths, indices, en_tf, vit_tf):
        self.paths = paths
        self.indices = indices
        self.en_tf = en_tf
        self.vit_tf = vit_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        en_tensor = self.en_tf(img)
        vit_tensor = self.vit_tf(img)
        return en_tensor, vit_tensor, self.indices[i]


num_workers = min(8, os.cpu_count() or 1)  # more workers for I/O
batch_size = 128  # larger batch reduces loop overhead

dataset = TestDataset(valid_paths, valid_idxs, en_transforms, vit_transforms)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,  # keep workers alive
    prefetch_factor=2,
    collate_fn=lambda x: (
        torch.stack([item[0] for item in x]),
        torch.stack([item[1] for item in x]),
        torch.tensor([item[2] for item in x], dtype=torch.long),
    ),
)

predictions = [0] * len(image_ids)  # default label 0 for missing files

for en_batch_cpu, vit_batch_cpu, batch_idxs in tqdm(loader, desc="Test"):
    en_batch = en_batch_cpu.to(device, non_blocking=True)
    vit_batch = vit_batch_cpu.to(device, non_blocking=True)

    en_batch_f = torch.flip(en_batch, dims=[3])
    vit_batch_f = torch.flip(vit_batch, dims=[3])

    en_combined = torch.cat([en_batch, en_batch_f], dim=0)
    vit_combined = torch.cat([vit_batch, vit_batch_f], dim=0)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            en_out_comb = en_model(en_combined)
            vit_out_comb = vit_model(vit_combined)

    B = en_batch.size(0)
    en_out, en_out_f = en_out_comb[:B], en_out_comb[B:]
    vit_out, vit_out_f = vit_out_comb[:B], vit_out_comb[B:]

    avg_output = (en_out + en_out_f + vit_out + vit_out_f) / 4.0

    adjusted_output = avg_output + torch.log(prior_tensor + 1e-8)

    batch_preds = torch.argmax(adjusted_output, dim=1).cpu()

    for idx, pred in zip(batch_idxs.tolist(), batch_preds.tolist()):
        predictions[idx] = int(pred)




## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
output_path = os.path.join(os.getcwd(), "submission.csv")
submission_df.to_csv(output_path, index=False)
print(f"Submission file created: {output_path}")
