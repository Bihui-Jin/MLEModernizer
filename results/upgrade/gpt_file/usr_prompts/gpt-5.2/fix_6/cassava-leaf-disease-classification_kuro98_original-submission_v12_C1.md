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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8937745542459957

# 6. Current score

0.2201

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing model checkpoint path by loading the ViT weights from a local Kaggle dataset if present, and otherwise fall back to a torchvision ViT with a compatible 5-class head so inference can run end-to-end. I also fix inference-time issues that break submission validity: test DataLoader must not shuffle (to avoid duplicates/misalignment), and TTA currently uses random transforms (non-deterministic + inconsistent); I replace them with deterministic flip-based TTA while keeping the same “average TTA probabilities then argmax” core logic. Finally, I enforce submission alignment exactly to `sample_submission.csv` order and length, guaranteeing a valid `submission.csv` with correct columns.'
- What this solution (achieved 0.19283) has done: 'The crash comes from a mismatch between your preprocessing size (384) and the torchvision ViT fallback’s fixed `image_size=224`, which asserts on input height/width. I make the image size derive from the loaded model (224 for torchvision ViT; keep 384 only when the custom checkpoint supports it), so the existing transforms and inference loop stay the same but won’t error. I also make the TTA transforms deterministic by using always-on flips (instead of random flip modules) so results are stable run-to-run while preserving the same “average probabilities then argmax” logic. Finally, I keep the submission aligned to `sample_submission.csv` order and ensure `submission.csv` is written.'
- What this solution (achieved 0.19357) has done: 'The crash is due to `os.listdir(test_dir)` picking up a nested `test_images/` directory inside the provided path, so PIL tries to open a directory as an image; I filter the dataset file list to include only actual image files and (as a safety) also fall back to the `sample_submission.csv` image list to guarantee exact coverage/order. I also make the model loading more robust by ensuring we use the correct callable model (some checkpoints store dicts) without changing the core ViT inference logic. These changes are execution-unblocking and should also improve the score versus predicting many missing rows as class 0. Finally, the script still write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.19096) has done: 'Your score is far below target, so the smallest safe way to move upward is to fix likely inference/preprocessing mismatch rather than changing the model. The current `CenterCrop((600,600))` can cut off the leaf for many images and is also inconsistent with common ViT/ImageNet preprocessing; I replace it with `Resize` + `CenterCrop` to the model input size, preserving the same normalization and TTA averaging logic. I also add an automatic unwrapping step for common checkpoint formats (e.g., `state_dict`) to ensure your intended trained weights actually load into the ViT architecture instead of silently falling back or misloading. Finally, I keep submission alignment to `sample_submission.csv` identical and still write `submission.csv`.'
- What this solution (achieved 0.2201) has done: 'Your score is far below target, so the smallest safe improvement is to ensure the inference preprocessing exactly matches the ViT backbone’s expected ImageNet pipeline instead of a custom resize/crop that can shift accuracy. I switch the test transforms to use the official `ViT_B_16_Weights.IMAGENET1K_V1.transforms()` when the fallback torchvision ViT is used (and keep your existing logic otherwise), which preserves the same model/loop/TTA averaging while improving calibration and input scaling. I also make the checkpoint loading stricter when it’s a state_dict (avoid silently missing most keys) so you don’t accidentally run with near-random weights. The submission writing and sample_submission alignment remain identical.'

# 9. Code solution

## === cell 0
import os
import glob

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

vit_model = None
using_torchvision_fallback = False

ckpt_candidates = [
    "/kaggle/input/vit-v1-update/vit_v1_1.pt",
]
for p in ckpt_candidates:
    if os.path.exists(p):
        vit_model = torch.load(p, map_location=device)
        print(f"Loaded checkpoint: {p}")
        break

if vit_model is None:
    matches = glob.glob("/kaggle/input/**/vit*.pt", recursive=True) + glob.glob(
        "/kaggle/input/**/*vit*.pt", recursive=True
    )
    matches = [m for m in matches if os.path.isfile(m)]
    if len(matches) > 0:
        vit_model = torch.load(matches[0], map_location=device)
        print(f"Loaded checkpoint: {matches[0]}")
    else:
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        using_torchvision_fallback = True
        print(
            "WARNING: No provided .pt checkpoint found under /kaggle/input; using torchvision ViT_B_16 "
            "ImageNet weights with a new 5-class head."
        )

if isinstance(vit_model, dict):
    if "model" in vit_model and hasattr(vit_model["model"], "forward"):
        vit_model = vit_model["model"]
        print("Unwrapped checkpoint dict -> vit_model['model']")
    else:
        state_key = None
        for k in ["state_dict", "model_state_dict", "net", "weights"]:
            if k in vit_model and isinstance(vit_model[k], dict):
                state_key = k
                break

        if state_key is not None:
            from torchvision.models import vit_b_16

            _m = vit_b_16(weights=None)
            _m.heads.head = torch.nn.Linear(_m.heads.head.in_features, num_classes)

            sd = vit_model[state_key]
            if any(kk.startswith("module.") for kk in sd.keys()):
                sd = {kk.replace("module.", "", 1): vv for kk, vv in sd.items()}

            missing, unexpected = _m.load_state_dict(sd, strict=False)

            total_params = len(_m.state_dict())
            miss_ratio = len(missing) / max(1, total_params)
            print(
                f"Loaded state_dict from dict['{state_key}'] into torchvision ViT. "
                f"missing={len(missing)}/{total_params} ({miss_ratio:.2%}), unexpected={len(unexpected)}"
            )
            if miss_ratio > 0.30:
                from torchvision.models import ViT_B_16_Weights

                _m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                _m.heads.head = torch.nn.Linear(_m.heads.head.in_features, num_classes)
                using_torchvision_fallback = True
                print(
                    "WARNING: state_dict mismatch too large; falling back to torchvision ImageNet ViT_B_16 weights."
                )
            vit_model = _m
        else:
            raise TypeError(
                "Loaded checkpoint is a dict but does not contain a usable 'model' or known state_dict key."
            )

if hasattr(vit_model, "image_size"):
    try:
        img_size = int(vit_model.image_size)
        print(f"Using model-enforced image_size={img_size}")
    except Exception:
        pass

vit_model.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (returns image tensor(s), filename)."""

    def __init__(self, data_dir, transform=None, ttas=None, image_ids=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = []
            for fn in os.listdir(data_dir):
                full = os.path.join(data_dir, fn)
                if os.path.isfile(full) and os.path.splitext(fn.lower())[1] in exts:
                    files.append(fn)
            self.images = sorted(files)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform is not None:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
if "using_torchvision_fallback" in globals() and using_torchvision_fallback:
    from torchvision.models import ViT_B_16_Weights

    test_transforms = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
else:
    test_transforms = v2.Compose(
        [
            v2.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

if tta:
    ttas = [
        v2.Identity(),
        v2.Lambda(lambda x: v2.functional.horizontal_flip(x)),
        v2.Lambda(lambda x: v2.functional.vertical_flip(x)),
    ]
else:
    ttas = None

sample_sub_for_ids = pd.read_csv(sample_sub_path)
test_ids = sample_sub_for_ids["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, transform=test_transforms, ttas=ttas, image_ids=test_ids
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)

            preds = normalizer(vit_model(inputs))  # [n_tta*B, num_classes]
            n_tta = len(ttas)
            bsz = len(filenames)

            preds = preds.view(n_tta, bsz, -1)  # [n_tta, B, num_classes]
            mean_preds = preds.mean(dim=0)  # [B, num_classes]
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(vit_model(inputs))
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print(f"Predicted {len(all_preds)} labels for {len(all_names)} images")



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)
pred_df.head()



## === cell 5
pred_df.shape, pred_df["label"].value_counts().sort_index()
