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

0.8381686310063463

# 6. Current score

0.12668

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13528) has done: 'I fix the missing checkpoint crash by falling back to a torchvision-pretrained model when the custom `.pth` files aren’t present, while keeping the same ViT/EfficientNet architecture choice. I also fix the CUDA/CPU mismatch by ensuring the loaded model weights and the model itself are moved to the same `device`. Finally, I make prediction collection robust so `predictions` always matches `sample_submission.csv` length (even if an image can’t be read), guaranteeing a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 0.05531) has done: 'Your score is low because the code is very likely running an ImageNet-pretrained model with an incompatible input pipeline for those weights (normalization) and also using an unsupported/incorrect ViT image size, so predictions are effectively close to random. To move accuracy toward your target with minimal changes and without changing the model family or inference approach, I (1) switch the normalization to the official ImageNet mean/std expected by torchvision pretrained weights, (2) set ViT’s `image_size` to a supported value (224) and make the resize match, and (3) run simple test-time augmentation (original + horizontal flip) averaged in logits to raise accuracy without changing training. These changes keep the same overall architecture choice and still produce the same valid `submission.csv` format.'
- What this solution (achieved 0.38976) has done: 'The timeout is dominated by per-image Python overhead: opening 2,676 JPEGs one-by-one, running transforms twice per image, and launching two separate forward passes. I preserve the exact same model and test-time augmentation (original + hflip averaged), but switch to a DataLoader-based pipeline with batched GPU inference, pinned-memory transfers, and multi-worker image decoding to cut wall time dramatically. I also fuse the “two passes” into a single forward over a 2× larger batch (equivalent math) to reduce kernel launch overhead and improve GPU utilization. Finally, I enable inference-only execution (`inference_mode`) and keep determinism/paths unchanged.'
- What this solution (achieved 0.21936) has done: 'Your score is still far below the target, so we should make a small change that legitimately increases accuracy without changing the model family or inference approach. The biggest remaining issue is that the transform always resizes to a square, but your code already defines an `invert_square_pad()` function that reflect-pads to square before resizing (often improves leaf classification because it preserves aspect ratio and avoids distortions). I wire that padding function into the existing inference dataset (keeping the same ViT/EfficientNet selection, same ImageNet normalization, and same 2-view TTA averaging) and keep everything else the same. This is a minimal, safe change that should move accuracy upward toward the target.'
- What this solution (achieved 0.17302) has done: 'Your accuracy is far below the target because `model_select="vit"` is using `vit_h_14` with `model_image_size=518`, which mismatches what torchvision’s ViT-H-14 backbone expects (224) and causes the pretrained model to perform very poorly. To move the score upward with minimal, core-logic-preserving changes, I keep the same model family (torchvision ViT) and the same inference/TTA loop, but (1) switch to `vit_l_16` (still ViT) which supports 224 cleanly and is much more stable under torchvision defaults, and (2) set `vit_image_size=224` so preprocessing matches the pretrained backbone. Everything else (ImageNet normalization, invert_square_pad, 2-view TTA averaging, DataLoader batching, submission writing) remains the same, so this is a small compatibility fix aimed specifically at improving accuracy toward your target.'
- What this solution (achieved 0.17302) has done: 'Your current score suggests the checkpoint isn’t actually being applied (so you’re effectively predicting with a 5-class head on ImageNet-pretrained features, which be close to random). I make a minimal, targeted fix to load your ViT checkpoint non-strictly and with common key fallbacks (e.g., `model.`, `backbone.`, `heads.` naming), and I add a lightweight sanity print of how many tensors matched so you can confirm weights are really loaded. This preserves your core model choice (torchvision ViT), transforms (including your invert-square-pad), and the same 2-view TTA averaging/inference loop—just makes the checkpoint loading robust so accuracy can move up toward your target. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.25747) has done: 'Your score is far below the target, so we should make a small, high-impact compatibility fix rather than tweaking speed or TTA. The biggest likely issue is that your ViT checkpoint path/name indicates it was trained from a `vit_h_14_518` setup, but the current code instantiates `vit_l_16` and thus can’t load most weights (so you’re effectively using an ImageNet backbone with a random 5-class head). I keep the same overall inference logic (torchvision ViT + 2-view TTA + same DataLoader pipeline), but switch the model to `vit_h_14` and set `model_image_size=518` so preprocessing matches the checkpoint and far more tensors load. I also set the ViT interpolation mode to bicubic (what ViTs typically expect) without changing the semantics of resizing.'
- What this solution (achieved 0.25747) has done: 'Your score is far below the target, so the most likely issue is still “weights aren’t really being applied correctly” even though the checkpoint file exists—especially for ViT-H/14 where many third‑party checkpoints store keys/shapes that need small remaps (e.g., `head` vs `heads.head`, pos-embedding tokens, or classifier shape). I keep the same model (vit_h_14), same image size (518), same transforms (including your invert_square_pad), and the same 2-view TTA inference loop, but make checkpoint loading more robust by (1) handling common nested checkpoint formats, (2) remapping `head`/`fc` keys to `heads.head`, and (3) safely skipping only the final classifier weights if their shape doesn’t match 5 classes (while still loading the rest strictly). This should increase accuracy legitimately by ensuring the backbone weights from your cassava checkpoint actually load, without changing your core inference semantics. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.12668) has done: 'Your score is far below the target, so we should focus on a minimal, high-impact compatibility fix rather than tuning. The biggest likely cause is a preprocessing mismatch for the ViT-H/14 checkpoint: most ViT checkpoints (and torchvision ViTs) expect normalization after mapping pixels to `[-1, 1]` (mean=std=0.5), but your current pipeline uses ImageNet mean/std, which can make predictions near-random even if weights load. I keep the same model (vit_h_14), same image size (518), same invert-square-pad, same 2-view TTA averaging, and the same DataLoader inference loop, but switch normalization to ViT-style `[0.5,0.5,0.5]/[0.5,0.5,0.5]` when `model_select=="vit"` (while keeping ImageNet normalization for EfficientNet). This is a small, targeted change expected to move accuracy materially upward toward your target without changing the core inference logic.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_data_directory = f"{DATA_DIR}/test_images"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8369_base.pth"
)
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 2
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()
    del top_combined, bottom_combined, flipped_img

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]

norm_mean, norm_std = (
    (VIT_MEAN, VIT_STD) if model_select == "vit" else (IMAGENET_MEAN, IMAGENET_STD)
)

val_transforms = transforms.Compose(
    [
        transforms.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(
            (model_image_size, model_image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.Normalize(norm_mean, norm_std),
    ]
)




## === cell 4
def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) == 0:
        return state_dict
    keys = list(state_dict.keys())

    if all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    return state_dict


def _extract_state_dict(maybe_ckpt):
    """
    Minimal robustness: many Kaggle checkpoints wrap the real tensors under common keys.
    This keeps core logic unchanged; it only ensures we read the intended tensors.
    """
    if not isinstance(maybe_ckpt, dict):
        return maybe_ckpt

    for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if (
            k in maybe_ckpt
            and isinstance(maybe_ckpt[k], dict)
            and len(maybe_ckpt[k]) > 0
        ):
            return maybe_ckpt[k]

    tensor_like = 0
    for v in maybe_ckpt.values():
        if torch.is_tensor(v):
            tensor_like += 1
        if tensor_like >= 3:
            return maybe_ckpt

    return maybe_ckpt


def _remap_keys_for_torchvision_vit(state_dict):
    """
    Try to adapt common checkpoint naming schemes to torchvision ViT naming.

    This does NOT change architecture or inference; it only maps parameter names so that
    more weights can be loaded (moving accuracy toward target).
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    keys = list(state_dict.keys())
    prefixes = ("model.", "net.", "backbone.", "encoder.")
    for p in prefixes:
        if all(k.startswith(p) for k in keys):
            state_dict = {k[len(p) :]: v for k, v in state_dict.items()}
            keys = list(state_dict.keys())
            break

    mapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("head."):
            nk = "heads." + nk  # head.weight -> heads.head.weight
        if nk.startswith("fc."):
            nk = "heads.head." + nk[len("fc.") :]  # fc.weight -> heads.head.weight
        if nk.startswith("classifier."):
            nk = "heads.head." + nk[len("classifier.") :]
        mapped[nk] = v
    state_dict = mapped

    return state_dict


def _drop_mismatched_classifier_weights(state_dict, model):
    """
    Minimal accuracy-improving fix: if checkpoint has a classifier head with wrong shape
    (e.g., 1000 classes), drop only those tensors so the backbone still loads.
    This preserves the same model head definition (5 classes) and inference semantics.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    model_sd = model.state_dict()

    to_drop = []
    for k in ("heads.head.weight", "heads.head.bias"):
        if k in state_dict and k in model_sd:
            if tuple(state_dict[k].shape) != tuple(model_sd[k].shape):
                to_drop.append(k)

    if to_drop:
        for k in to_drop:
            state_dict.pop(k, None)
        print(f"NOTE: Dropped mismatched classifier tensors from checkpoint: {to_drop}")

    return state_dict


def _load_checkpoint_if_exists(model, ckpt_path, device, *, strict=True):
    if ckpt_path is None:
        return False
    if not os.path.exists(ckpt_path):
        return False

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    state = _extract_state_dict(state)

    state = _clean_state_dict_keys(state)

    if isinstance(model, models.VisionTransformer):
        state = _remap_keys_for_torchvision_vit(state)
        state = _drop_mismatched_classifier_weights(state, model)

    incompatible = model.load_state_dict(state, strict=strict)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    total_params = len(list(model.state_dict().keys()))
    loaded_params = total_params - len(missing)

    print(
        f"Loaded checkpoint: {ckpt_path}\n"
        f"  strict={strict} | loaded_tensors≈{loaded_params}/{total_params} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    return True


if model_select == "vit":
    vit_model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    loaded = _load_checkpoint_if_exists(vit_model, vit_model_path, device, strict=False)
    if not loaded:
        print(
            "WARNING: ViT checkpoint not found; using ImageNet pretrained weights only."
        )

    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    en_model = models.efficientnet_v2_l(
        weights=models.EfficientNet_V2_L_Weights.DEFAULT
    )
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )

    loaded = _load_checkpoint_if_exists(en_model, en_model_path, device, strict=False)
    if not loaded:
        print(
            "WARNING: EfficientNet checkpoint not found; using ImageNet pretrained weights only."
        )

    en_model.to(device)
    en_model.eval()



## === cell 5
from torch.utils.data import Dataset, DataLoader

sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].astype(str).tolist()


class TestImageDataset(Dataset):
    def __init__(self, image_ids, root_dir, tfm):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, image_name)

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing test image: {image_path}")

            img = Image.open(image_path).convert("RGB")
            x1 = self.tfm(img)
            x2 = self.tfm(transforms.functional.hflip(img))
            return x1, x2, idx, True
        except Exception:
            dummy = torch.zeros(
                (3, model_image_size, model_image_size), dtype=torch.float32
            )
            return dummy, dummy, idx, False


def _collate(batch):
    x1 = torch.stack([b[0] for b in batch], dim=0)
    x2 = torch.stack([b[1] for b in batch], dim=0)
    idx = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    ok = torch.tensor([b[3] for b in batch], dtype=torch.bool)
    return x1, x2, idx, ok


ds = TestImageDataset(test_image_ids, test_data_directory, val_transforms)

num_workers = min(4, (os.cpu_count() or 2))
pin_memory = torch.cuda.is_available()

batch_size = 8 if model_select == "vit" else 16

dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=_collate,
)

predictions = np.zeros(len(test_image_ids), dtype=np.int64)

model = vit_model if model_select == "vit" else en_model

with torch.inference_mode():
    for x1, x2, idx, ok in tqdm(dl, desc="Test", total=len(dl)):
        if ok.any():
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            x = torch.cat([x1, x2], dim=0)
            out = model(x)
            bsz = x1.shape[0]
            out1, out2 = out[:bsz], out[bsz:]
            output = (out1 + out2) / 2.0
            pred = torch.argmax(output, dim=1).to("cpu").numpy().astype(np.int64)

            idx_np = idx.cpu().numpy()
            ok_np = ok.cpu().numpy()

            predictions[idx_np[ok_np]] = pred[ok_np]


predictions = predictions.tolist()

if len(predictions) != len(test_image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(predictions)} vs {len(test_image_ids)}"
    )



## === cell 6
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})

assert submission_df.shape[0] == sample_df.shape[0], "Submission row count mismatch."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch."

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
