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

0.8694469628286491

# 6. Current score

0.07399

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with the exact rows/order expected by `sample_submission.csv`. The immediate blocker is missing model files under `/kaggle/input/vit-v1/...`, so I add a safe fallback that loads a standard torchvision ViT if those assets aren’t present, keeping the same inference flow and TTA logic. I also fix DataLoader shuffling (must be `False`) and align predictions to `sample_submission.csv` to prevent length/order mismatches. Finally, I make the dataset/TTA path deterministic and robust (RGB conversion, sorted filenames, correct TTA stacking) so it doesn’t crash or produce invalid outputs.'
- What this solution (achieved 0.58744) has done: 'I fix the ViT input-size assertion by making the fallback ViT models use an `image_size` that matches your configured 384px preprocessing, so inference runs without crashing. This is a bug fix only (no architecture/training loop changes) and preserves your current ensemble + TTA flow. I also make the TTA transforms deterministic at inference by switching to functional flips/rotations/perspective with fixed parameters; this avoids randomness that can destabilize accuracy and typically improves it for a fixed model. Finally, I keep the submission alignment merge against `sample_submission.csv` exactly as you already do, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the safest way to move it upward (without changing your ensemble/TTA core logic) is to ensure you are not accidentally running with random, untrained ViT weights. I keep your exact inference flow, but change the fallback model creation to use ImageNet-pretrained ViT weights (same architecture, same input size), and ensure the linear head is actually applied (it’s currently loaded but unused). These two fixes typically yield a large accuracy jump while preserving your model+TTA approach and submission semantics. I also keep the sample_submission alignment exactly as you already do so the output CSV remains valid.'
- What this solution (achieved 0.20703) has done: 'I fix the runtime error caused by trying to instantiate an ImageNet-pretrained `vit_b_16` with `image_size=384`, which torchvision disallows for those weights, by building the fallback ViT at 224 when using pretrained weights and resizing inputs to match that expected size. To preserve your existing ensemble/TTA inference flow, I keep the same model averaging + optional `linear_head` application, and only adjust the dataset resize sizes to match each model’s actual required image size. This unblock execution (so `model_a`/`model_b` exist) and should substantially increase accuracy versus the currently-broken run that never reaches inference. The submission writing/alignment against `sample_submission.csv` remains unchanged and always produce a valid `submission.csv`.'
- What this solution (achieved 0.22347) has done: 'Your current score is far below the target, so we should make the smallest changes that reliably improve accuracy without changing your overall inference core (2-model ensemble + TTA + softmax/argmax + sample_submission alignment). The biggest likely issue is preprocessing mismatch for torchvision ImageNet-pretrained ViT: it expects a specific resize/crop pipeline (resize shorter side to 256 then center-crop 224) rather than always center-cropping 600 and resizing, which can severely hurt accuracy. I keep your architecture and TTA logic intact, but adjust the dataset preprocessing to use the official `ViT_B_16_Weights` transforms when a model is running with ImageNet weights, while preserving your existing flow for custom checkpoints. I also make the `linear_head` loading robust to state_dict checkpoints (so it actually loads when provided), which can further move the score upward.'
- What this solution (achieved 0.05531) has done: 'We keep your exact ensemble + TTA + softmax/argmax flow, but fix a likely major accuracy issue: when ImageNet-pretrained ViT is used, you currently replace the model’s classifier head with 5 classes (randomly initialized), which destroys the benefit of pretraining. Instead, we keep the pretrained 1000-class head and add a small 1000→5 adapter head (initialized deterministically) so the model’s pretrained logits remain informative; this is a minimal change that usually lifts accuracy substantially from the current ~0.22 toward your ~0.87 target. We also apply the linear head consistently as a mapper on the ensemble logits only when its input dimension matches; otherwise we fall back safely to Identity to avoid silent dimension misuse. All paths, I/O, and submission alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly recover real model signal without changing your ensemble/TTA inference core. The biggest issue is that the fallback “adapter” for ImageNet-pretrained ViT is initialized to all zeros, which forces every prediction to class 0 and yields near-random/majority-class accuracy; we instead use a deterministic, non-degenerate mapping from the pretrained 1000-class logits to 5 classes (still no training, same inference flow). We also fix a subtle but important TTA batching bug: your DataLoader collates TTA lists into a nested structure, and `torch.cat(list(model_a_inputs))` can produce wrong shapes/order; we explicitly stack to `[bs, num_tta, C, H, W]` then reshape to `[bs*num_tta, ...]` deterministically. These two changes preserve your architecture and pipeline semantics (2-model ensemble + TTA + softmax/argmax + sample_submission alignment) while moving accuracy upward toward the target band.'
- What this solution (achieved 0.07399) has done: 'The crash comes from how the DataLoader collates your TTA lists: by default it returns a list of tensors shaped `[bs, C, H, W]` per TTA (not a list-per-sample), so your current `torch.stack([torch.stack(x) for x in model_a_inputs])` is stacking incorrectly. I fix cell 3 to robustly handle both possible collate layouts by converting the batch into a single tensor of shape `[bs, num_tta, C, H, W]` and then flattening to `[bs*num_tta, C, H, W]` for inference. This is a pure bug fix (same ensemble + TTA + averaging + linear head + softmax/argmax logic), and it let the pipeline run end-to-end and write a valid `submission.csv`. With the code now actually running TTA inference correctly, your score should move upward toward the target.'

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _safe_load_or_build_vit(checkpoint_path: str, img_size: int, num_classes: int):
    state_dict = None
    if os.path.exists(checkpoint_path):
        obj = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            obj.expected_image_size = int(getattr(obj, "expected_image_size", img_size))
            obj.using_imagenet_pretrained = False
            obj.logits_adapter = getattr(obj, "logits_adapter", torch.nn.Identity())
            return obj
        state_dict = obj

    from torchvision.models import vit_b_16, ViT_B_16_Weights

    using_pretrained = state_dict is None
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if using_pretrained else None

    if using_pretrained:
        enforced_img_size = int(weights.meta["min_size"][0])  # typically 224
    else:
        enforced_img_size = int(img_size)

    model = vit_b_16(weights=weights, image_size=enforced_img_size)

    if state_dict is not None:
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.logits_adapter = torch.nn.Identity()
    else:
        adapter = torch.nn.Linear(1000, num_classes, bias=True)
        with torch.no_grad():
            adapter.weight.zero_()
            adapter.bias.zero_()
            splits = torch.linspace(0, 1000, steps=num_classes + 1, dtype=torch.int64)
            for c in range(num_classes):
                lo = int(splits[c].item())
                hi = int(splits[c + 1].item())
                if hi <= lo:
                    hi = min(lo + 1, 1000)
                adapter.weight[c, lo:hi] = 1.0 / float(hi - lo)
        model.logits_adapter = adapter

    model.expected_image_size = enforced_img_size
    model.using_imagenet_pretrained = bool(using_pretrained)
    return model


def _safe_load_linear_head(path: str, device: torch.device):
    if not os.path.exists(path):
        return torch.nn.Identity().to(device)

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        w_key = None
        b_key = None
        for k in ["weight", "fc.weight", "head.weight", "linear.weight"]:
            if k in obj:
                w_key = k
                break
        for k in ["bias", "fc.bias", "head.bias", "linear.bias"]:
            if k in obj:
                b_key = k
                break

        if w_key is not None:
            w = obj[w_key]
            out_features, in_features = int(w.shape[0]), int(w.shape[1])
            layer = torch.nn.Linear(in_features, out_features)
            sd = {}
            sd["weight"] = obj[w_key]
            if b_key is not None:
                sd["bias"] = obj[b_key]
            layer.load_state_dict(sd, strict=False)
            return layer.to(device)

    return torch.nn.Identity().to(device)


model_a_path = "/kaggle/input/vit-v1/vit_v1.pt"
model_b_path = "/kaggle/input/vit-boosted/vit_boosted.pt"
linear_head_path = "/kaggle/input/linear-head/linear_cls.pt"

model_a = _safe_load_or_build_vit(model_a_path, model_a_img_size, num_classes).to(
    device
)
model_b = _safe_load_or_build_vit(model_b_path, model_b_img_size, num_classes).to(
    device
)

model_a_img_size = int(getattr(model_a, "expected_image_size", model_a_img_size))
model_b_img_size = int(getattr(model_b, "expected_image_size", model_b_img_size))

linear_head = _safe_load_linear_head(linear_head_path, device)


def _maybe_wrap_linear_head(
    lh: torch.nn.Module, expected_in: int, expected_out: int, device
):
    if isinstance(lh, torch.nn.Identity):
        return lh
    if hasattr(lh, "in_features") and hasattr(lh, "out_features"):
        if int(lh.in_features) == int(expected_in) and int(lh.out_features) == int(
            expected_out
        ):
            return lh.to(device)
    return torch.nn.Identity().to(device)


linear_head = _maybe_wrap_linear_head(
    linear_head, expected_in=num_classes, expected_out=num_classes, device=device
)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data."""

    def __init__(
        self,
        data_dir,
        model_a,
        model_b,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.ttas = ttas

        self.model_a = model_a
        self.model_b = model_b

        from torchvision.models import ViT_B_16_Weights

        self._a_use_imagenet = bool(
            getattr(model_a, "using_imagenet_pretrained", False)
        )
        self._b_use_imagenet = bool(
            getattr(model_b, "using_imagenet_pretrained", False)
        )

        self._a_pre = (
            ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            if self._a_use_imagenet
            else None
        )
        self._b_pre = (
            ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            if self._b_use_imagenet
            else None
        )

        a_size = int(getattr(model_a, "expected_image_size", 384))
        b_size = int(getattr(model_b, "expected_image_size", 384))
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (a_size, a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (b_size, b_size), interpolation=InterpolationMode.BICUBIC
        )

    def _preprocess_a(self, img):
        if self._a_pre is not None:
            return self._a_pre(img)
        img = self.cc(img)
        img = self.resize_model_a(img)
        return self.transform(img) if self.transform is not None else img

    def _preprocess_b(self, img):
        if self._b_pre is not None:
            return self._b_pre(img)
        img = self.cc(img)
        img = self.resize_model_b(img)
        return self.transform(img) if self.transform is not None else img

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None:
            model_a_img = [self._preprocess_a(t(img)) for t in self.ttas]
            model_b_img = [self._preprocess_b(t(img)) for t in self.ttas]
        else:
            model_a_img = self._preprocess_a(img)
            model_b_img = self._preprocess_b(img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        lambda x: x,
        lambda x: v2.functional.horizontal_flip(x),
        lambda x: v2.functional.vertical_flip(x),
        lambda x: v2.functional.rotate(
            x, angle=90, interpolation=InterpolationMode.BILINEAR
        ),
        lambda x: v2.functional.perspective(
            x,
            startpoints=[
                (0, 0),
                (x.size[0] - 1, 0),
                (x.size[0] - 1, x.size[1] - 1),
                (0, x.size[1] - 1),
            ],
            endpoints=[
                (10, 10),
                (x.size[0] - 11, 5),
                (x.size[0] - 5, x.size[1] - 11),
                (5, x.size[1] - 5),
            ],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a=model_a,
    model_b=model_b,
    transform=test_transforms,
    ttas=ttas,
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
def _as_batched_tta_tensor(batch_tta):
    if isinstance(batch_tta, (list, tuple)):
        if len(batch_tta) == 0:
            raise ValueError("Empty TTA batch.")
        if torch.is_tensor(batch_tta[0]):
            return (
                torch.stack(list(batch_tta), dim=0).permute(1, 0, 2, 3, 4).contiguous()
            )
        if (
            isinstance(batch_tta[0], (list, tuple))
            and len(batch_tta[0]) > 0
            and torch.is_tensor(batch_tta[0][0])
        ):
            return torch.stack(
                [torch.stack(x, dim=0) for x in batch_tta], dim=0
            ).contiguous()
    if torch.is_tensor(batch_tta):
        return batch_tta
    raise TypeError(f"Unsupported batch type for TTA inputs: {type(batch_tta)}")


all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()
if hasattr(model_a, "logits_adapter"):
    model_a.logits_adapter.eval()
if hasattr(model_b, "logits_adapter"):
    model_b.logits_adapter.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)
        filenames = list(filenames)

        if tta:
            model_a_inputs = _as_batched_tta_tensor(model_a_inputs)  # [bs, T, C, H, W]
            model_b_inputs = _as_batched_tta_tensor(model_b_inputs)  # [bs, T, C, H, W]
            num_tta = int(model_a_inputs.shape[1])

            model_a_inputs = model_a_inputs.view(
                bs * num_tta, *model_a_inputs.shape[2:]
            ).to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.view(
                bs * num_tta, *model_b_inputs.shape[2:]
            ).to(device, non_blocking=True)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            if hasattr(model_a, "logits_adapter") and not isinstance(
                model_a.logits_adapter, torch.nn.Identity
            ):
                model_a_outputs = model_a.logits_adapter(model_a_outputs)
            if hasattr(model_b, "logits_adapter") and not isinstance(
                model_b.logits_adapter, torch.nn.Identity
            ):
                model_b_outputs = model_b.logits_adapter(model_b_outputs)

            model_a_batch_logits = model_a_outputs.view(bs, num_tta, -1).mean(dim=1)
            model_b_batch_logits = model_b_outputs.view(bs, num_tta, -1).mean(dim=1)

            outputs = (model_a_batch_logits + model_b_batch_logits) / 2.0
            outputs = linear_head(outputs)

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            if hasattr(model_a, "logits_adapter") and not isinstance(
                model_a.logits_adapter, torch.nn.Identity
            ):
                model_a_outputs = model_a.logits_adapter(model_a_outputs)
            if hasattr(model_b, "logits_adapter") and not isinstance(
                model_b.logits_adapter, torch.nn.Identity
            ):
                model_b_outputs = model_b.logits_adapter(model_b_outputs)

            outputs = (model_a_outputs + model_b_outputs) / 2.0
            outputs = linear_head(outputs)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 4
sample_sub = pd.read_csv(sample_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if merged["label"].isna().any():
    merged["label"] = merged["label"].fillna(0).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
merged
