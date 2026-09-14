# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8921124206708976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.22085) has done: 'I fix the runtime crash by ensuring the ViT model’s expected input size matches the preprocessing size: if the loaded checkpoint model expects 224, we resize to 224 (score-neutral bug fix that makes inference run). I also fix the missing/KeyError submission issue by writing predictions directly in dataloader order (since `shuffle=False`), avoiding fragile name-to-pred mapping when a crash/partial loop occurs. Additionally, I make checkpoint loading robust to both “full model” and “state_dict” formats without changing the architecture. Finally, I ensure a valid `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.2216) has done: 'Your current score suggests the checkpoint is either not being applied correctly (so you’re effectively using an untrained/random 5-class head) or inference preprocessing is mismatched to what the checkpoint expects. I make the smallest changes that keep your ViT architecture and inference flow identical, but (1) load the checkpoint in a more robust way by handling common key prefixes like `model.`/`module.` and verifying that head weights are actually loaded, and (2) remove the hard-coded CenterCrop(600,600) which can badly cut leaves in this dataset and often tanks accuracy; we rely on the resize to `img_size` only (still deterministic). These changes are directly targeted at moving accuracy up toward your target without changing the core model/training approach, and still produce a valid `submission.csv` with correct ordering.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly fix a major correctness issue rather than “tuning.” The biggest likely issue is that the checkpoint isn’t actually being applied to the model due to key mismatches (different head naming, extra prefixes, etc.), leaving you effectively with an ImageNet backbone plus random 5-class head. I add a minimal but robust checkpoint loader that (a) strips common prefixes, (b) remaps common ViT head keys to torchvision’s `heads.head.*`, and (c) asserts that head weights loaded (otherwise it falls back to a safe bias initialization using train label priors to avoid catastrophic random predictions). These changes keep your model architecture and inference logic the same and only address weight loading / calibration so the accuracy moves upward toward the target.'
- What this solution (achieved 0.61099) has done: 'Your score gap is large (0.611 vs 0.892), which strongly suggests a major “semantic mismatch” at inference time rather than a need for tuning. I keep your ViT-B/16 architecture and inference loop intact, but make two minimal, high-impact fixes: (1) load the checkpoint in a way that guarantees the correct `image_size` used by the model is respected (so we don’t resize to the wrong resolution when the checkpoint expects 384), and (2) use the *exact* ViT preprocessing normalization/crop pipeline from the corresponding torchvision `ViT_B_16_Weights` (instead of generic ImageNet normalize + direct resize), which commonly accounts for large accuracy jumps. These changes don’t alter your model/training approach; they only align preprocessing and checkpoint application to what the model expects. The script still run end-to-end and write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8921), so the most likely issue is a preprocessing mismatch rather than “model weakness.” I make a minimal change to use the torchvision ViT weights’ preprocessing *as-is* (it already includes resize/crop/normalize), and only override its resize size to match your `img_size`—right now you’re effectively applying resize twice in a conflicting order. I also remove the explicit `Softmax` during inference (argmax is invariant to softmax) to avoid any tiny numerical differences and speed slightly, without changing prediction semantics. Everything else (model, checkpoint loading, dataset, dataloader ordering, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images/"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
train_csv_path = f"{DATA_ROOT}/train.csv"

img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False

ckpt_path = "/kaggle/input/vit-v3/vit_v3.pt"


def _build_default_model(image_size: int):
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1, image_size=image_size)
    in_features = m.heads.head.in_features
    m.heads.head = torch.nn.Linear(in_features, num_classes)
    return m


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return None


def _strip_prefixes(state_dict):
    prefixes = ("module.", "model.", "net.")
    out = state_dict
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if any(k.startswith(p) for k in out.keys()):
                out = {k[len(p) :]: v for k, v in out.items()}
                changed = True
    return out


def _remap_head_keys_for_torchvision_vit(state_dict):
    remaps = {
        "head.weight": "heads.head.weight",
        "head.bias": "heads.head.bias",
        "fc.weight": "heads.head.weight",
        "fc.bias": "heads.head.bias",
        "classifier.weight": "heads.head.weight",
        "classifier.bias": "heads.head.bias",
        "heads.weight": "heads.head.weight",
        "heads.bias": "heads.head.bias",
        "head1.weight": "heads.head.weight",
        "head1.bias": "heads.head.bias",
    }
    out = dict(state_dict)
    for src, dst in remaps.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    return out


def _init_head_bias_with_train_priors(model, train_csv, num_classes):
    df = pd.read_csv(train_csv)
    counts = (
        df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=1)
        .astype(float)
    )
    priors = (counts / counts.sum()).values
    bias = torch.tensor(priors, dtype=torch.float32).clamp_min(1e-6).log()
    with torch.no_grad():
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            model.heads.head.bias.copy_(bias.to(model.heads.head.bias.device))
            model.heads.head.weight.zero_()
        else:
            raise AttributeError("Expected torchvision ViT model with model.heads.head")


def _infer_image_size_from_pos_embed(state_dict):
    k = None
    for cand in ("encoder.pos_embedding", "pos_embed", "pos_embedding"):
        if cand in state_dict:
            k = cand
            break
    if k is None:
        return None, None

    pe = state_dict[k]
    if not torch.is_tensor(pe) or pe.ndim != 3:
        return None, k

    n_tokens = int(pe.shape[1])  # includes class token
    n_patches = n_tokens - 1
    if n_patches <= 0:
        return None, k

    grid = int(round(n_patches**0.5))
    if grid * grid != n_patches:
        return None, k

    patch = 16  # vit_b_16
    inferred = grid * patch
    return inferred, k


model = None
loaded_sd = None
ckpt_obj = None

if os.path.exists(ckpt_path):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, torch.nn.Module):
        model = ckpt_obj
        if hasattr(model, "image_size"):
            try:
                img_size = int(model.image_size)
            except Exception:
                pass
    else:
        loaded_sd = _extract_state_dict(ckpt_obj)
        if loaded_sd is not None:
            loaded_sd = _strip_prefixes(loaded_sd)
            loaded_sd = _remap_head_keys_for_torchvision_vit(loaded_sd)

            inferred_size, pe_key = _infer_image_size_from_pos_embed(loaded_sd)
            if inferred_size is not None and inferred_size > 0:
                if img_size != inferred_size:
                    print(
                        f"Inferred image_size={inferred_size} from ckpt key '{pe_key}'; overriding img_size {img_size} -> {inferred_size}"
                    )
                img_size = inferred_size

        model = _build_default_model(img_size)

        if loaded_sd is not None:
            missing, unexpected = model.load_state_dict(loaded_sd, strict=False)
            print(
                f"Loaded ckpt with strict=False; missing={len(missing)} unexpected={len(unexpected)}"
            )
            pe_missing = any("encoder.pos_embedding" in k for k in missing)
            if pe_missing:
                print(
                    "WARNING: encoder.pos_embedding missing after load. This usually indicates an image_size/variant mismatch and hurts accuracy."
                )
else:
    model = _build_default_model(img_size)

if isinstance(ckpt_obj, dict):
    for k in ("image_size", "img_size", "input_size"):
        if k in ckpt_obj:
            try:
                v = ckpt_obj[k]
                if isinstance(v, (list, tuple)) and len(v) >= 1:
                    v = v[0]
                v = int(v)
                if v > 0 and v != img_size:
                    print(
                        f"Detected {k}={v} in checkpoint; using it for inference resize."
                    )
                    img_size = v
                    break
            except Exception:
                pass

if hasattr(model, "image_size"):
    required_img_size = int(model.image_size)
    if img_size != required_img_size:
        print(
            f"Adjusting img_size from {img_size} to {required_img_size} to match model.image_size"
        )
        img_size = required_img_size

model = model.to(device)

head_loaded = False
if loaded_sd is not None and isinstance(loaded_sd, dict):
    has_w = "heads.head.weight" in loaded_sd
    has_b = "heads.head.bias" in loaded_sd
    head_loaded = bool(has_w or has_b)

    head_keys = [
        k for k in loaded_sd.keys() if ("head" in k and ("weight" in k or "bias" in k))
    ]
    print(
        f"Checkpoint head-related keys (sample): {head_keys[:8]} (count={len(head_keys)})"
    )
    print(
        f"Detected mapped torchvision head present in ckpt: weight={has_w}, bias={has_b}"
    )

if not head_loaded:
    print(
        "WARNING: checkpoint head weights not detected as loaded; initializing head bias from train label priors."
    )
    _init_head_bias_with_train_priors(model, train_csv_path, num_classes)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4227331948.py in <cell line: 0>()
    180                 )
    181 else:
--> 182     model = _build_default_model(img_size)
    183 
    184 # Keep existing behavior: allow metadata-provided size to override, but only if sane.

/tmp/ipykernel_55/4227331948.py in _build_default_model(image_size)
     39     from torchvision.models import vit_b_16, ViT_B_16_Weights
     40 
---> 41     m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1, image_size=image_size)
     42     in_features = m.heads.head.in_features
     43     m.heads.head = torch.nn.Linear(in_features, num_classes)

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    226                 kwargs[weights_param] = default_weights_arg
    227 
--> 228             return builder(*args, **kwargs)
    229 
    230         return inner_wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in vit_b_16(weights, progress, **kwargs)
    639     weights = ViT_B_16_Weights.verify(weights)
    640 
--> 641     return _vision_transformer(
    642         patch_size=16,
    643         num_layers=12,

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _vision_transformer(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, weights, progress, **kwargs)
    319         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    320         assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
--> 321         _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    322     image_size = kwargs.pop("image_size", 224)
    323 

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in _ovewrite_named_param(kwargs, param, new_value)
    236     if param in kwargs:
    237         if kwargs[param] != new_value:
--> 238             raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    239     else:
    240         kwargs[param] = new_value

ValueError: The parameter 'image_size' expected value 224 but got 384 instead.

## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (image_id list provided)."""

    def __init__(self, data_dir, image_ids, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.image_ids = list(image_ids)
        self.ttas = ttas
        self.cc = None

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        path = os.path.join(self.root, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing test image: {path}")

        img = Image.open(path).convert("RGB")

        if self.cc is not None:
            img = self.cc(img)

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform is not None:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.image_ids)




## === cell 2
from torchvision.models import ViT_B_16_Weights

_vit_weights = ViT_B_16_Weights.IMAGENET1K_V1

test_transforms = _vit_weights.transforms(crop_size=img_size, resize_size=img_size)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, image_ids=test_image_ids, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # critical for alignment
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 3
all_names = []
all_preds = []

model.eval()
with torch.no_grad():
    for inputs, filenames in test_loader:
        filenames = list(filenames)

        if tta:
            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            logits = model(inputs)
            batch_logits = torch.stack(torch.split(logits, len(filenames)), dim=0)
            mean_logits = torch.mean(batch_logits, dim=0)
            pred_labels = torch.argmax(mean_logits, 1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            pred_labels = torch.argmax(logits, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

if len(all_names) != len(test_image_ids):
    raise RuntimeError(
        f"Pred count mismatch: got {len(all_names)} predictions but expected {len(test_image_ids)}. "
        "Inference likely crashed mid-epoch or dataset ordering changed."
    )

if all_names != test_image_ids:
    pred_map = dict(zip(all_names, all_preds))
    missing = [iid for iid in test_image_ids if iid not in pred_map]
    if missing:
        raise KeyError(
            f"Missing predictions for {len(missing)} images, e.g. {missing[:3]}"
        )
    ordered_preds = [int(pred_map[iid]) for iid in test_image_ids]
else:
    ordered_preds = [int(x) for x in all_preds]

my_submission = pd.DataFrame({"image_id": test_image_ids, "label": ordered_preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(my_submission))
print(my_submission.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3982080486.py in <cell line: 0>()
      2 all_preds = []
      3 
----> 4 model.eval()
      5 with torch.no_grad():
      6     for inputs, filenames in test_loader:

AttributeError: 'NoneType' object has no attribute 'eval'

## === cell 4
my_submission

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3983424916.py in <cell line: 0>()
----> 1 my_submission

NameError: name 'my_submission' is not defined
