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

0.8872771229978845

# 6. Current score

0.4503

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08744) has done: 'I remove the unnecessary TensorFlow imports that are triggering the protobuf `MessageFactory.GetPrototype` crash, since the pipeline is purely PyTorch/torchvision. I also make the pretrained weight paths robust by auto-falling back to the built-in torchvision ImageNet weights when the Kaggle input checkpoints are missing, so inference can run end-to-end and produce a valid `submission.csv`. Finally, I fix the CUDA dtype/device mismatch by ensuring the model parameters are moved to the same device as the input (and handle older/newer torch versions’ `torch.load(weights_only=...)` compatibility). These changes keep the core inference logic intact (same model families, same preprocessing intent, same argmax prediction) while unblocking execution and yielding a valid submission file.'
- What this solution (achieved 0.47795) has done: 'Your current score (0.08744) is far below the target (0.8873), and the most likely cause is that you’re not actually using the intended cassava-finetuned checkpoints (so the model is effectively random due to a freshly reinitialized 5-class head). I make a minimal change to load checkpoints robustly by (1) searching the entire `/kaggle/input` tree for a `.pth/.pt` whose filename matches your intended checkpoint name, and (2) if still not found, falling back to “any `.pth/.pt` under that dataset folder”. This keeps the same model, transforms, and argmax inference, but greatly increases the chance we load the real finetuned weights and move accuracy toward your target. I also print which checkpoint got loaded so you can verify it in the output logs.'
- What this solution (achieved 0.08707) has done: 'Your score is far below the target, so the most likely issue is a mismatch between the checkpoint you load and the model definition (especially for ViT: image_size and head/classifier key names), causing either partial loading or effectively random predictions. I keep the same inference core (single model, same transforms intent, argmax) but make checkpoint loading robust to common Kaggle-trained formats by (1) unwrapping nested keys like `state_dict`, `model`, and `module.` prefixes, and (2) falling back to `strict=False` with a clear log of missing/unexpected keys instead of silently failing quality. I also ensure the ViT is instantiated with `image_size=model_image_size` (currently hardcoded to 518) so the positional embedding shape matches the checkpoint, which is a frequent accuracy killer if wrong. These are minimal, execution-safe changes that should move accuracy substantially toward your target if the finetuned checkpoint is present.'
- What this solution (achieved 0.09492) has done: 'Your current score is far below the target, so we should focus on the most likely score-killers while keeping your inference core unchanged (same single-model argmax pipeline and same transforms intent). The biggest red flag is the non-standard `invert_square_pad` transform, which almost certainly does not match the preprocessing used when the finetuned checkpoint was trained, making even correct weights perform poorly; we switch to a standard center-crop/resize pipeline that matches torchvision ViT defaults and typical cassava training. We also ensure the ViT model is created with the *exact* expected `image_size` only when loading a finetuned checkpoint, and otherwise use torchvision’s default image_size for pretrained weights to avoid positional-embedding mismatches. Finally, we keep your robust checkpoint searching/unwrapping, and we keep submission ordering/alignment identical.'
- What this solution (achieved 0.13004) has done: 'Your score is far below the target, so the most likely remaining issue is that the inference normalization doesn’t match what torchvision ViT/EfficientNet checkpoints expect (ImageNet mean/std), which can destroy accuracy even with correct weights. I make the smallest change that preserves your core logic (same model selection, same single-model argmax inference, same transforms structure) by switching `Normalize([0.5]*3,[0.5]*3)` to the correct ImageNet normalization and ensuring RGB conversion stays identical. I also keep checkpoint loading exactly as-is, but add a tiny safeguard to ensure we always use the same preprocessing regardless of which pretrained/fine-tuned weights are loaded. This should move accuracy substantially upward toward your target without changing architecture/training/inference semantics.'
- What this solution (achieved 0.4503) has done: 'Your score is far below the target, so we should make a minimal change that most plausibly fixes a remaining “quiet” mismatch between your finetuned checkpoint and the instantiated model, without changing the model family or inference approach. The most common cause here is a head naming mismatch for ViT (`heads.head.*` vs `heads.0.*`) and/or loading a checkpoint saved from a wrapped module; this can result in the classifier head not being loaded (or the wrong keys being used) and severely hurt accuracy. I add a small key-remapping step for ViT heads plus a clearer “loaded keys sanity check” so we actually use the finetuned 5-class head weights when present. Everything else (single-model argmax inference, transforms, dataset order, submission writing) stays the same.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/4/vit_h_14_518_8907_ISP_CBP.pth"
)
vit_image_size = 518

model_select = "vit"  # keep original selection default

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size



## === cell 2
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose(
    [
        transforms.Resize(
            int(round(model_image_size * 256 / 224)),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop((model_image_size, model_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)




## === cell 3
def _safe_torch_load(path, map_location):
    """
    Fix: torch.load(weights_only=...) is version-dependent.
    Try weights_only=True first, then fall back.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_existing_checkpoint(primary_path):
    """
    Prefer loading the intended finetuned checkpoint. If the exact path doesn't exist,
    search more robustly.
    """
    if os.path.exists(primary_path):
        return primary_path

    fname = os.path.basename(primary_path)

    global_matches = glob.glob(
        os.path.join("/kaggle/input", "**", fname), recursive=True
    )
    global_matches = [p for p in global_matches if os.path.isfile(p)]
    if len(global_matches) > 0:
        global_matches = sorted(global_matches, key=lambda p: (len(p), p))
        return global_matches[0]

    parts = primary_path.split("/")
    base = "/".join(parts[:4])  # /kaggle/input/<dataset_name>
    if base.startswith("/kaggle/input") and os.path.exists(base):
        candidates = glob.glob(os.path.join(base, "**", "*.pth"), recursive=True)
        candidates += glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
        candidates = [p for p in candidates if os.path.isfile(p)]
        if len(candidates) > 0:
            candidates = sorted(candidates, key=lambda p: (len(p), p))
            return candidates[0]

    return None


def _extract_state_dict(obj):
    """
    Unwrap common checkpoint formats and strip DDP prefixes so we actually load the finetuned weights.
    """
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], (dict,)):
                obj = obj[k]
                break

    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()):
            if any(k.startswith("module.") for k in obj.keys()):
                obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def _remap_vit_head_keys_if_needed(state):
    """
    Score fix (minimal): many ViT checkpoints store classifier head as:
      - 'heads.head.weight' / 'heads.head.bias' (newer torchvision style), OR
      - 'heads.0.weight' / 'heads.0.bias' (Sequential head style)
    If we instantiate one style but checkpoint uses the other, strict loading will fail
    or (worse) strict=False will skip the head, hurting accuracy a lot.
    We remap between these two naming conventions when the shapes match.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    has_head_dot = ("heads.head.weight" in state) or ("heads.head.bias" in state)
    has_head0 = ("heads.0.weight" in state) or ("heads.0.bias" in state)

    if has_head0 and (not has_head_dot):
        w = state.get("heads.0.weight", None)
        b = state.get("heads.0.bias", None)
        if w is not None:
            state["heads.head.weight"] = w
        if b is not None:
            state["heads.head.bias"] = b

    if has_head_dot and (not has_head0):
        w = state.get("heads.head.weight", None)
        b = state.get("heads.head.bias", None)
        if w is not None:
            state["heads.0.weight"] = w
        if b is not None:
            state["heads.0.bias"] = b

    return state


def _load_model_weights(model, ckpt_path, device, model_kind=None):
    """
    Load strictly when possible; otherwise strict=False but log key issues.
    """
    raw = _safe_torch_load(ckpt_path, map_location=device)
    state = _extract_state_dict(raw)

    if model_kind == "vit":
        state = _remap_vit_head_keys_if_needed(state)

    try:
        model.load_state_dict(state, strict=True)
        print("[INFO] load_state_dict(strict=True): OK")
    except RuntimeError as e:
        print("[WARN] load_state_dict(strict=True) failed; retrying strict=False.")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[INFO] strict=False loaded. Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}"
        )
        if len(missing) > 0:
            print("[INFO] Missing sample:", missing[:10])
        if len(unexpected) > 0:
            print("[INFO] Unexpected sample:", unexpected[:10])
        print("[INFO] Original strict=True error (truncated):", str(e)[:300])

    if model_kind == "vit":
        sd = model.state_dict()
        hw = sd.get("heads.head.weight", None)
        if hw is not None:
            print(f"[INFO] ViT head weight shape in model: {tuple(hw.shape)}")


if model_select == "vit":
    vit_ckpt = _find_existing_checkpoint(vit_model_path)

    if vit_ckpt is not None:
        print(f"[INFO] Loading ViT checkpoint: {vit_ckpt}")

        vit_model = models.vit_h_14(weights=None, image_size=model_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        _load_model_weights(vit_model, vit_ckpt, device=device, model_kind="vit")
    else:
        print(
            "[WARN] No ViT finetuned checkpoint found; using ImageNet weights + new head (likely low score)."
        )

        try:
            vit_model = models.vit_h_14(
                weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            )
        except Exception:
            vit_model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )

    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    en_ckpt = _find_existing_checkpoint(en_model_path)

    if en_ckpt is not None:
        print(f"[INFO] Loading EfficientNetV2 checkpoint: {en_ckpt}")
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        _load_model_weights(en_model, en_ckpt, device=device, model_kind="en")
    else:
        print(
            "[WARN] No EfficientNet finetuned checkpoint found; using ImageNet/None weights + new head (likely low score)."
        )
        try:
            en_model = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.DEFAULT
            )
        except Exception:
            en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )

    en_model.to(device)
    en_model.eval()



## === cell 4
sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].astype(str).tolist()

missing = [
    img_id
    for img_id in image_ids
    if not os.path.exists(os.path.join(test_data_directory, img_id))
]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images in {test_data_directory}. Example: {missing[:5]}"
    )


class TestImageDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform=None):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_name


batch_size = 16 if torch.cuda.is_available() else 8
test_ds = TestImageDataset(test_data_directory, image_ids, transform=val_transforms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

predictions = []
out_image_ids = []



## === cell 5
with torch.no_grad():
    for batch_imgs, batch_names in tqdm(test_loader, desc="Test"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        if model_select == "vit":
            outputs = vit_model(batch_imgs)
        else:
            outputs = en_model(batch_imgs)

        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().astype(np.int64)
        predictions.extend(pred.tolist())
        out_image_ids.extend(list(batch_names))

if out_image_ids != image_ids:
    pred_map = {k: v for k, v in zip(out_image_ids, predictions)}
    predictions = [int(pred_map[k]) for k in image_ids]
    out_image_ids = image_ids

submission_df = pd.DataFrame(
    {"image_id": out_image_ids, "label": np.array(predictions, dtype=np.int64)}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_df)})")
print(
    "Label value counts:\n",
    submission_df["label"].value_counts(dropna=False).sort_index(),
)
