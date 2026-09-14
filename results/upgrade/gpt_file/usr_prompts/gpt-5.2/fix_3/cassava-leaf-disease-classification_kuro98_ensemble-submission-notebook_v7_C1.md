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

0.8981565427621638

# 6. Current score

0.10613

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54709) has done: 'I fix the pipeline so it always runs end-to-end and writes a valid `submission.csv`. The main blocker is missing external model files under `/kaggle/input/...`, so I add a safe fallback that uses a pretrained torchvision model when those files aren’t present, without changing the overall “image → model → argmax label” semantics. I also fix inference bugs that cause invalid submission length: the test loader must not be shuffled, and test filenames must be returned in the exact sample submission order. Finally, I make TTA deterministic at inference by using non-random augmentations (flips/rotations), keeping the intended “average across TTAs” logic while ensuring stability.'
- What this solution (achieved 0.10613) has done: 'Your current score is far below the target, and the biggest (minimal) win without changing your “image → model → argmax label” core logic is to make the fallback path actually use a meaningful classifier instead of a randomly-initialized 5-class head. I keep the exact same preprocessing/inference flow and simply (1) keep EfficientNet-B0’s pretrained 1000-class head, then (2) map its ImageNet predictions to your 5 cassava classes using a tiny “nearest-label” rule built from the provided disease names in `label_num_to_disease_map.json`. This preserves evaluation semantics (still argmax over 5 classes) while giving you a much stronger baseline than random weights, moving accuracy toward the target. I also replace `Random*` TTA ops with deterministic equivalents (same transforms, no RNG) to avoid any residual nondeterminism at inference.'

# 9. Code solution

## === cell 0
import os
import json
import warnings

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images/"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
label_map_path = f"{DATA_ROOT}/label_num_to_disease_map.json"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _try_load_model(path: str):
    if path and os.path.exists(path):
        return torch.load(path, map_location=device).to(device)
    return None


vit_model = _try_load_model("/kaggle/input/vit-v1-update/vit_v1_1.pt")
eff_model = _try_load_model("/kaggle/input/efficient-net/vit_cont_3.pt")
linear_head = _try_load_model("/kaggle/input/linear-head/linear_cls.pt")

fallback_mode = (vit_model is None) or (eff_model is None) or (linear_head is None)
if fallback_mode:
    warnings.warn(
        "One or more external model files were not found under /kaggle/input/. "
        "Falling back to torchvision EfficientNet_B0 pretrained ImageNet classifier and "
        "a lightweight label-name-based mapping to 5 cassava classes (still argmax over 5)."
    )
    import torchvision

    fallback_model = torchvision.models.efficientnet_b0(weights="DEFAULT").to(device)

    with open(label_map_path, "r") as f:
        label_name_map = json.load(f)
    cassava_names = [label_name_map[str(i)].lower() for i in range(num_classes)]

    KEYWORDS = {
        0: {"mosaic", "cmd", "cassava mosaic"},
        1: {"brown", "cbsd", "streak"},
        2: {"green", "mite", "cgm"},
        3: {"blight", "cbb", "bacterial"},
        4: {"healthy"},
    }

    def _map_imagenet_to_cassava(imagenet_logits: torch.Tensor) -> torch.Tensor:
        """
        Deterministic mapping from ImageNet logits (B,1000) to cassava logits (B,5).
        Uses top-k ImageNet class names when available; otherwise falls back to simple priors.
        """
        B = imagenet_logits.shape[0]
        cassava_logits = torch.zeros(
            (B, num_classes), device=imagenet_logits.device, dtype=imagenet_logits.dtype
        )

        try:
            categories = torchvision.models.EfficientNet_B0_Weights.DEFAULT.meta[
                "categories"
            ]
        except Exception:
            categories = None

        if categories is None:
            cassava_logits[:, 4] = 1.0
            return cassava_logits

        topk = 5
        top_idx = torch.topk(imagenet_logits, k=topk, dim=1).indices  # (B, topk)

        for i in range(B):
            scores = torch.zeros(
                num_classes, device=imagenet_logits.device, dtype=imagenet_logits.dtype
            )
            for j in range(topk):
                idx = int(top_idx[i, j].item())
                name = categories[idx].lower()

                for c in range(num_classes):
                    if any(k in name for k in KEYWORDS[c]):
                        scores[c] += 1.0

                if ("leaf" in name) or ("plant" in name) or ("tree" in name):
                    scores[4] += 0.2

            if float(scores.sum().item()) == 0.0:
                scores[4] = 1.0

            cassava_logits[i] = scores

        return cassava_logits




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images.

    Bugfix: to ensure a valid submission, we must preserve the exact test image order
    given by sample_submission.csv. Using os.listdir + shuffled DataLoader breaks that.
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        image_ids,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = list(image_ids)  # ordered list from sample_submission
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None and self.transform is not None:
            vit_img = [self.transform(t(vit_img)) for t in self.ttas]
            eff_img = [self.transform(t(eff_img)) for t in self.ttas]
        elif self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),  # deterministic since p=1.0
        v2.RandomVerticalFlip(p=1.0),  # deterministic since p=1.0
        v2.RandomRotation(degrees=(90, 90)),  # deterministic since fixed degrees
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    image_ids=test_image_ids,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
all_names = []
all_preds = []

if fallback_mode:
    fallback_model.eval()
else:
    vit_model.eval()
    eff_model.eval()
    linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        bsz = len(filenames)

        if fallback_mode:
            if tta:
                eff_inputs_cat = torch.cat(eff_inputs, dim=0).to(
                    device, non_blocking=True
                )
                imagenet_logits = fallback_model(eff_inputs_cat)  # (n_tta*B, 1000)

                imagenet_logits_tta = torch.stack(
                    torch.split(imagenet_logits, bsz), dim=0
                )  # (n_tta, B, 1000)
                mean_imagenet_logits = torch.mean(
                    imagenet_logits_tta, dim=0
                )  # (B, 1000)

                cassava_logits = _map_imagenet_to_cassava(
                    mean_imagenet_logits
                )  # (B, 5)
                probs = normalizer(cassava_logits)
                pred_labels = torch.argmax(probs, 1).tolist()
            else:
                eff_inputs = eff_inputs.to(device, non_blocking=True)
                imagenet_logits = fallback_model(eff_inputs)
                cassava_logits = _map_imagenet_to_cassava(imagenet_logits)
                probs = normalizer(cassava_logits)
                pred_labels = torch.argmax(probs, 1).tolist()
        else:
            if tta:
                vit_inputs_cat = torch.cat(vit_inputs, dim=0).to(
                    device, non_blocking=True
                )
                eff_inputs_cat = torch.cat(eff_inputs, dim=0).to(
                    device, non_blocking=True
                )

                vit_outputs = vit_model(vit_inputs_cat)
                eff_outputs = eff_model(eff_inputs_cat)

                vit_batch_logits = torch.stack(torch.split(vit_outputs, bsz), dim=0)
                vit_mean_logits = torch.mean(vit_batch_logits, dim=0)

                eff_batch_logits = torch.stack(torch.split(eff_outputs, bsz), dim=0)
                eff_mean_logits = torch.mean(eff_batch_logits, dim=0)

                logit_inputs = torch.cat([vit_mean_logits, eff_mean_logits], dim=1)
                outputs = linear_head(logit_inputs)

                probs = normalizer(outputs)
                pred_labels = torch.argmax(probs, 1).tolist()
            else:
                vit_inputs = vit_inputs.to(device, non_blocking=True)
                eff_inputs = eff_inputs.to(device, non_blocking=True)

                vit_outputs = vit_model(vit_inputs)
                eff_outputs = eff_model(eff_inputs)

                logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
                outputs = linear_head(logit_inputs)

                probs = normalizer(outputs)
                pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## === cell 5
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.set_index("image_id").reindex(sample_sub["image_id"])
if pred_df["label"].isna().any():
    pred_df["label"] = pred_df["label"].fillna(0).astype(int)
else:
    pred_df["label"] = pred_df["label"].astype(int)

submission = pred_df.reset_index()
submission.to_csv("submission.csv", index=False)

print("submission.csv written:", submission.shape)
submission.head()
