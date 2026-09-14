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

0.8919613176186159

# 6. Current score

0.74103

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10426) has done: 'You’re hitting a ViT input-size assertion because your fallback `vit_b_16` expects 224×224 images but the pipeline resizes to 384×384; we align the resize sizes to the actual loaded models’ expected `image_size` to fix this robustly for both external and fallback models. The “missing predictions” error is a downstream consequence of the crash in the prediction loop; once inference completes, the merge be fully populated. I also make the ensemble math consistent (currently you halve the logits after already using 0.8/0.2 weights), which is a minimal logic fix that should slightly improve accuracy without changing the core approach. Finally, I ensure `linear_head` (if present) is moved to the right device to avoid a future device-mismatch crash.'
- What this solution (achieved 0.23206) has done: 'Your current score (0.10426) is far below the target (0.89196), and the most likely cause is that you’re running untrained fallback ViTs (weights=None) because the external model files are not being found/loaded in this environment. I make the model loading robust by (1) searching common Kaggle input locations for the provided `.pt` files, and (2) if still missing, switching the fallback to a pretrained torchvision ViT (same architecture) and only replacing the classifier head to 5 classes—this preserves the core ViT approach while avoiding near-random predictions. I also ensure the inference output handling works whether the loaded `.pt` models return logits directly or return a dict/tuple, which prevents silently broken ensembling. These are minimal, execution-safe changes aimed directly at improving accuracy toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.15022) has done: 'Your score gap is large (0.23206 vs target 0.89196), and with this inference-only script the most likely remaining cause is that the “fallback” models are still effectively random because the classifier head is newly initialized. I keep the same ViT architecture and ensemble logic, but change the fallback construction to use an ImageNet-pretrained ViT and keep its pretrained head (1000 classes), then add a small linear adapter to 5 classes using the already-declared `linear_head` concept (so we don’t change the inference loop style). This makes predictions non-random without changing the overall approach (two ViTs + weighted logits + softmax + argmax) and keeps execution under the time limit. I also ensure the adapter is always on the right device and that logits extraction works consistently.'
- What this solution (achieved 0.08146) has done: 'I fix the adapter shape bug causing the matmul crash by applying the adapter to the correct tensor (the ViT feature embedding) instead of the 1000-class logits, while keeping the same “backbone + optional linear_head + weighted ensemble” inference semantics. Concretely, for the torchvision fallback we replace the ViT classification head with an Identity so the backbone outputs 768-d features, matching the adapter’s expected input, and we ensure logits extraction is robust. This unblocks inference so every test image gets a prediction, which also resolves the “missing predictions” submission error. The change is execution-critical and also score-improving versus random/untrained behavior because it properly uses the pretrained backbone features.'
- What this solution (achieved 0.06876) has done: 'Your current score is far below the target, which strongly suggests the inference pipeline is still misaligned with what the loaded models (or fallback torchvision ViTs) actually output. I make two minimal, score-relevant fixes: (1) use the correct preprocessing for torchvision’s pretrained ViT weights (their specific normalize/resize/crop recipe) when we are in fallback mode, and (2) fix the adapter application so it is fed the 768-d backbone features (not 5-class logits), by applying the adapter directly to each backbone’s feature output and then ensembling in 5-class space. These changes preserve your core approach (two ViTs + weighted logits + optional linear head + argmax), but remove a major source of near-random predictions due to preprocessing mismatch and incorrect head usage. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.30194) has done: 'Your score (0.06876) is far below the target (0.89196), and with a correct submission pipeline this usually indicates the fallback path is producing near-random predictions. The smallest change that should materially improve accuracy without changing the core “two ViTs + weighted ensemble + argmax” logic is to fix the fallback adapter initialization: the current “grouped ImageNet classes into 5 chunks” mapping is arbitrary and tends to be uninformative for cassava labels. Instead, we keep the exact same architecture/inference flow but initialize the 5-way adapter from the pretrained ViT head using SVD/PCA-style compression (top-5 right-singular directions), which preserves maximal variance from the pretrained classifier and typically yields much stronger zero-shot features than random/chunked grouping. Everything else (data order, ensembling weights 0.8/0.2, softmax+argmax, submission writing) stays the same, and it still run within the time limit.'
- What this solution (achieved 0.29858) has done: 'Your current score is far below the target, so the safest path is to improve calibration/feature quality without changing the core “two ViTs + weighted ensemble + argmax” logic. The biggest score drag in the fallback path is likely a distribution mismatch caused by doing an extra CenterCrop/resize before applying `weights.transforms()` (which itself already resizes/crops); I remove that extra crop/resize only for fallback to match the pretrained ViT’s expected preprocessing. I also make the fallback deterministic across the two ensemble members by reusing the same pretrained backbone weights and initializing both adapters from the same SVD basis (still two models + 0.8/0.2 weighting, just less noise), which should move accuracy upward toward the target. Finally, I make the filename ordering robust by stripping any accidental path components so every prediction merges cleanly into `sample_submission.csv`.'
- What this solution (achieved 0.74103) has done: 'Your current score (0.29858) is far below the target (0.89196), so we should improve accuracy with minimal, inference-only changes that preserve your “two ViTs + (optional) linear adapter + 0.8/0.2 weighted logits + softmax + argmax” core logic. The biggest safe gain here is to stop using an essentially untrained SVD-compressed adapter in fallback mode and instead fit the 5-class adapter weights on the provided `train.csv` using the frozen pretrained ViT backbone features (no backbone training, just a linear classifier). This keeps the same architecture (pretrained ViT-B/16 backbone + linear head to 5 classes) and the same inference loop semantics, but replaces weak zero-shot initialization with a supervised head learned from the training labels. To keep changes minimal and runtime under 600s, we train the adapter on a small, deterministic subset of training images for a few epochs using the same CrossEntropy objective implied by accuracy, then run the existing test inference and write `submission.csv` unchanged.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images/"
train_dir = f"{DATA_ROOT}/train_images/"
train_csv_path = f"{DATA_ROOT}/train.csv"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"

model_b_img_size = 384
model_a_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _build_fallback_vit_with_adapter(num_classes: int, shared_W5: torch.Tensor | None):
    """
    Fallback: pretrained torchvision ViT-B/16 backbone, head removed -> returns 768-d features.

    Note: we keep this adapter definition intact, but in fallback mode we will *fit* the adapter
    on cassava train labels using frozen backbone features (score-relevant, preserves core logic).
    """
    import torchvision

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    backbone = torchvision.models.vit_b_16(weights=weights)

    head = backbone.heads.head  # Linear(768 -> 1000)
    W = head.weight.detach().clone()  # [1000, 768]

    backbone.heads.head = torch.nn.Identity()

    if shared_W5 is None:
        with torch.no_grad():
            W32 = W.to(dtype=torch.float32)
            try:
                _, _, Vh = torch.linalg.svd(W32, full_matrices=False)
                W5 = Vh[:num_classes, :].contiguous()  # [5, 768]
            except Exception:
                Q, _ = torch.linalg.qr(W32.T, mode="reduced")  # [768, 768]
                W5 = Q[:, :num_classes].T.contiguous()  # [5, 768]
    else:
        W5 = shared_W5.detach().clone()

    adapter = torch.nn.Linear(W5.shape[1], num_classes)
    with torch.no_grad():
        adapter.weight.copy_(W5.to(adapter.weight.dtype))
        adapter.bias.zero_()

    return backbone, adapter, weights, W5


def _infer_vit_image_size(model) -> int | None:
    if hasattr(model, "image_size"):
        try:
            return int(model.image_size)
        except Exception:
            pass
    for attr in ("img_size", "input_size", "image_resolution"):
        if hasattr(model, attr):
            try:
                v = getattr(model, attr)
                if isinstance(v, (tuple, list)):
                    return int(v[0])
                return int(v)
            except Exception:
                pass
    return None


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _candidate_paths(rel_name: str):
    roots = [
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]
    cands = []
    if rel_name.startswith("/"):
        cands.append(rel_name)
    for r in roots:
        cands.append(os.path.join(r, rel_name.lstrip("/")))
    dataset_names = ["vit-v1", "vit-boosted", "linear-head"]
    for dn in dataset_names:
        cands.append(f"/kaggle/input/{dn}/{os.path.basename(rel_name)}")
    return cands


vit_a_path = _find_first_existing(_candidate_paths("/kaggle/input/vit-v1/vit_v1.pt"))
vit_b_path = _find_first_existing(
    _candidate_paths("/kaggle/input/vit-boosted/vit_boosted.pt")
)
linear_head_path = _find_first_existing(
    _candidate_paths("/kaggle/input/linear-head/linear_cls.pt")
)

have_external_models = False
model_a = None
model_b = None
linear_head = None

using_fallback = False
fallback_weights = None

try:
    if vit_a_path is not None and vit_b_path is not None:
        model_a = torch.load(vit_a_path, map_location=device)
        model_b = torch.load(vit_b_path, map_location=device)
        have_external_models = True
    if linear_head_path is not None:
        linear_head = torch.load(linear_head_path, map_location=device)
except Exception as e:
    print(
        "Warning: failed to load external checkpoints, falling back to pretrained torchvision ViT. Error:",
        repr(e),
    )
    have_external_models = False
    model_a = None
    model_b = None
    linear_head = None

if not have_external_models:
    using_fallback = True

    model_a, linear_head_a, w_a, shared_W5 = _build_fallback_vit_with_adapter(
        num_classes, shared_W5=None
    )
    model_b, linear_head_b, w_b, _ = _build_fallback_vit_with_adapter(
        num_classes, shared_W5=shared_W5
    )
    fallback_weights = w_a

    linear_head = torch.nn.ModuleList([linear_head_a, linear_head_b])

    model_a = model_a.to(device)
    model_b = model_b.to(device)
    linear_head = linear_head.to(device)

model_a.to(device)
model_b.to(device)
if linear_head is not None:
    linear_head.to(device)

a_size = _infer_vit_image_size(model_a)
b_size = _infer_vit_image_size(model_b)
if a_size is not None:
    model_a_img_size = a_size
if b_size is not None:
    model_b_img_size = b_size

print(f"Using image sizes -> model_a: {model_a_img_size}, model_b: {model_b_img_size}")
print(f"External models loaded: {have_external_models}")
print(f"Using torchvision fallback: {using_fallback}")
if have_external_models:
    print("Loaded:", vit_a_path, vit_b_path, linear_head_path)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test images.

    If `image_list` is provided, images are returned in that exact order to match sample_submission.csv.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_list=None,
        center_crop_size=600,
        skip_preresize=False,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.ttas = ttas
        self.skip_preresize = skip_preresize

        if image_list is None:
            self.images = sorted(os.listdir(data_dir))
        else:
            self.images = list(image_list)

        self.cc = (
            None
            if skip_preresize
            else v2.CenterCrop((center_crop_size, center_crop_size))
        )
        self.resize_model_a = (
            None
            if skip_preresize
            else v2.Resize(
                (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
            )
        )
        self.resize_model_b = (
            None
            if skip_preresize
            else v2.Resize(
                (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
            )
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        filename_key = os.path.basename(filename)

        img = Image.open(os.path.join(self.root, filename_key)).convert("RGB")

        if self.skip_preresize:
            model_a_img = img
            model_b_img = img
        else:
            img = self.cc(img)
            model_a_img = self.resize_model_a(img)
            model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename_key

    def __len__(self):
        return len(self.images)




## === cell 2
if using_fallback and fallback_weights is not None:
    weights_tfm = fallback_weights.transforms()
    test_transforms = weights_tfm
    skip_preresize = True
    fallback_crop_size = 224
    model_a_img_size = 224
    model_b_img_size = 224
else:
    test_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    skip_preresize = False
    fallback_crop_size = 600

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_list = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_list=test_image_list,
    center_crop_size=fallback_crop_size,
    skip_preresize=skip_preresize,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
class CassavaTrainDataset(VisionDataset):
    def __init__(self, data_dir, df, transform):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, os.path.basename(image_id))).convert(
            "RGB"
        )
        x = self.transform(img) if self.transform is not None else img
        return x, y


def _fit_fallback_adapters_on_train(
    model_a,
    model_b,
    linear_head: torch.nn.ModuleList,
    weights_tfm,
    train_csv_path: str,
    train_dir: str,
    device: torch.device,
):
    train_df = pd.read_csv(train_csv_path)

    max_train = 4096
    if len(train_df) > max_train:
        train_df = train_df.iloc[:max_train].copy()

    train_ds = CassavaTrainDataset(train_dir, train_df, transform=weights_tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model_a.eval()
    model_b.eval()
    for p in model_a.parameters():
        p.requires_grad = False
    for p in model_b.parameters():
        p.requires_grad = False

    linear_head.train()
    for p in linear_head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(linear_head.parameters(), lr=3e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 3
    for ep in range(epochs):
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device, dtype=torch.long)

            with torch.no_grad():
                fa = model_a(x)
                fb = model_b(x)

            logits = 0.8 * linear_head[0](fa) + 0.2 * linear_head[1](fb)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)

        print(f"[fallback head fit] epoch {ep+1}/{epochs} loss={running/max(n,1):.4f}")

    linear_head.eval()
    return linear_head


if (
    using_fallback
    and isinstance(linear_head, torch.nn.ModuleList)
    and len(linear_head) == 2
    and fallback_weights is not None
):
    linear_head = _fit_fallback_adapters_on_train(
        model_a=model_a,
        model_b=model_b,
        linear_head=linear_head,
        weights_tfm=fallback_weights.transforms(),
        train_csv_path=train_csv_path,
        train_dir=train_dir,
        device=device,
    )




## === cell 4
def _to_tensor(model_out):
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("logits", "out", "output", "features", "embeddings"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
        for v in model_out.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


all_names = []
all_preds = []

model_a.eval()
model_b.eval()
if linear_head is not None:
    linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            a_out = _to_tensor(model_a(model_a_inputs))
            b_out = _to_tensor(model_b(model_b_inputs))

            a_batch = torch.stack(torch.split(a_out, bs), dim=0)
            b_batch = torch.stack(torch.split(b_out, bs), dim=0)
            a_mean = torch.mean(a_batch, dim=0)
            b_mean = torch.mean(b_batch, dim=0)
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            a_mean = _to_tensor(model_a(model_a_inputs))
            b_mean = _to_tensor(model_b(model_b_inputs))

        if (
            using_fallback
            and isinstance(linear_head, torch.nn.ModuleList)
            and len(linear_head) == 2
        ):
            cassava_logits = 0.8 * linear_head[0](a_mean) + 0.2 * linear_head[1](b_mean)
            probs = normalizer(cassava_logits)
        else:
            outputs = 0.8 * a_mean + 0.2 * b_mean
            if (
                linear_head is not None
                and outputs.ndim == 2
                and outputs.shape[1] != num_classes
            ):
                if (
                    isinstance(linear_head, torch.nn.ModuleList)
                    and len(linear_head) == 2
                ):
                    cassava_logits = 0.8 * linear_head[0](a_mean) + 0.2 * linear_head[
                        1
                    ](b_mean)
                else:
                    cassava_logits = linear_head(outputs)
                probs = normalizer(cassava_logits)
            else:
                probs = normalizer(outputs)

        pred_labels = torch.argmax(probs, 1).tolist()
        all_names.extend(filenames)
        all_preds.extend(pred_labels)

if len(all_names) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(all_names)} predictions, expected {len(sample_sub)}."
    )



## === cell 5
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_sub[["image_id"]].merge(my_submission, on="image_id", how="left")
if my_submission["label"].isna().any():
    missing = my_submission[my_submission["label"].isna()]["image_id"].head(10).tolist()
    raise RuntimeError(f"Missing predictions for some images, e.g.: {missing}")

my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
