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

0.8854638863705047

# 6. Current score

0.17078

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blockers by making checkpoint discovery work in this environment (your provided ViT/EN checkpoint paths don’t exist), and ensure `model_image_size` is always defined so transforms build reliably. To preserve the original inference-only core logic while still producing a valid submission, I fall back to torchvision’s pretrained weights when no custom `.pth` is found (same architectures, same forward pass; only the weight source changes). I also fix the glob to be recursive, and make the head/num_classes handling robust so the model can run even when pretrained weights have 1000 classes. Finally, I make sure we always generate `submission.csv` with the exact `sample_submission.csv` ordering.'
- What this solution (achieved 0.40209) has done: 'I fix the ViT input-size mismatch that causes the runtime `AssertionError` by resizing images to the actual `vit_model.image_size` (or EfficientNet’s expected size) rather than the hardcoded checkpoint image size. This is a correctness fix (not a model change) and should also materially improve accuracy because the model now receive inputs at the resolution it was designed for, instead of failing or being mismatched. I also make the classifier-head handling robust when falling back to torchvision pretrained weights so the model always outputs 5 classes without the current “mod num_classes” fallback that destroys accuracy. Finally, I keep the submission ordering aligned to `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.40209) is far below the target (0.88546), so we should improve accuracy with minimal, semantics-preserving fixes. The biggest issue is that when no custom cassava checkpoint is found, the code replaces the 1000-class ImageNet head with a fresh 5-class head, which makes predictions essentially random; instead, we should keep the pretrained head and map its logits to 5 cassava classes via a fixed ImageNet→cassava class mapping. We also switch normalization to use the model’s official pretrained weights transforms (when using torchvision pretrained weights), which is a correctness fix for inference-time preprocessing. These changes keep the same architectures and inference-only approach, and should move the score substantially toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we should fix the biggest correctness issue while keeping the same inference-only architecture/loop: right now you’re (often) using ImageNet-pretrained weights with an ad‑hoc ImageNet→cassava mapping that is effectively random for this task. The minimal, legitimate way to move accuracy toward the target without changing the core approach is to use the provided cassava TFRecords to build a small “prototype classifier” in the existing model’s embedding/logit space (no training loop changes: still pure inference, just a better post-processing head). Concretely, we compute per-class mean vectors from a capped number of TFRecord training images using the frozen model outputs, then classify each test image by nearest prototype (cosine similarity). This keeps the same backbone forward pass, preserves evaluation semantics (predict 0–4), stays within time by limiting the number of TFRecord examples, and writes a valid `submission.csv` in sample order.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the goal is to fix the biggest *correctness* bug in the current “prototype from TFRecords” inference head while keeping the same backbone models, forward pass, and inference loop. Right now your TFRecord prototype builder is effectively broken because it reads images with `torch.io.BytesIO` (nonexistent) and the TFRecord protobuf parsing is fragile, so your prototypes are likely `None` or garbage—leading to near-random predictions. I replace the TFRecord parsing with a small, robust TFRecord+protobuf extractor (no TensorFlow, no extra packages), keep the same prototype logic (mean normalized embeddings per class), and keep submission ordering identical to `sample_submission.csv`. This should materially raise accuracy toward the target without changing model architectures or adding any training.'
- What this solution (achieved 0.17078) has done: 'Your current score is far below the target, so the priority is to fix a correctness issue in the prototype head while keeping the same backbone/inference-only core logic. Right now prototypes are computed from the model’s *final logits* (5-way for custom ckpt, 1000-way for ImageNet), which is a poor embedding space; instead we extract a stable penultimate feature vector (ViT `pre_logits` / EfficientNet pooled features) and build prototypes in that feature space. This preserves architecture and training approach (still no training; same backbone forward), but makes the nearest-prototype classifier meaningful and should move accuracy substantially toward the target. I also ensure the TFRecord parsing uses `image`/`target` fallback keys (some cassava TFRecords use `target`) without changing the overall TFRecord-based prototype approach.'
- What this solution (achieved 0.17078) has done: 'Your current score (0.17078) is far below the target (0.88546), so we should improve accuracy with minimal, semantics-preserving fixes rather than changing the model/training approach. The biggest correctness issue is that your TFRecord protobuf parser is not actually parsing `tf.train.Example` correctly (it treats the `features` message as raw bytes and skips the required nested `Features`/`Feature` structure), so your prototypes are built from wrong/empty data and predictions become near-random. I replace only the TFRecord Example parsing with a minimal but correct protobuf decoder for the specific `Example -> Features -> feature{key: Feature}` structure, keeping the same prototype logic (mean L2-normalized penultimate features per class) and the same inference loop. This should materially move the score upward toward the target while staying within runtime and still writing a valid `submission.csv` in sample order.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torch.utils.data import (
    DataLoader,
)  # kept (though unused) to preserve original intent
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob
import warnings
import io
import struct

warnings.filterwarnings("ignore", category=UserWarning)



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
en_image_size = 480

vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/3/vit_h_14_518_8927.pth"
vit_image_size = 518

model_select = "vit"  # preserve original selection




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
def resolve_checkpoint_path(preferred_path: str, pattern: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    matches = sorted(glob.glob(pattern, recursive=True))
    if matches:
        return matches[-1]
    return None


def set_model_num_classes(model, num_classes: int):
    """
    Used ONLY when a cassava-trained custom checkpoint exists (so head weights match).
    """
    if (
        hasattr(model, "heads")
        and hasattr(model.heads, "head")
        and hasattr(model.heads.head, "in_features")
    ):
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model
    if hasattr(model, "head") and hasattr(model.head, "in_features"):
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        return model
    if hasattr(model, "classifier"):
        if (
            isinstance(model.classifier, torch.nn.Sequential)
            and len(model.classifier) > 0
        ):
            last = model.classifier[-1]
            if hasattr(last, "in_features"):
                model.classifier[-1] = torch.nn.Linear(last.in_features, num_classes)
                return model
        if hasattr(model.classifier, "in_features"):
            model.classifier = torch.nn.Linear(
                model.classifier.in_features, num_classes
            )
            return model
    raise AttributeError(
        "Unexpected model head structure; cannot set classifier head to num_classes."
    )


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def extract_features(model, x: torch.Tensor, model_select_local: str) -> torch.Tensor:
    if model_select_local == "vit":
        n = x.shape[0]
        x = model._process_input(x)
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        x = x[:, 0]
        if hasattr(model, "heads") and hasattr(model.heads, "pre_logits"):
            x = model.heads.pre_logits(x)
        return x
    else:
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


resolved_vit_path = resolve_checkpoint_path(vit_model_path, "/kaggle/input/**/vit*.pth")
resolved_en_path = resolve_checkpoint_path(
    en_model_path, "/kaggle/input/**/efficientnet*.pth"
)

vit_model = None
en_model = None

model_image_size = None
weights_for_transforms = None
using_custom_ckpt = False

if model_select == "vit":
    vit_ctor_errors = []
    for ctor_name in ["vit_h_14", "vit_l_16", "vit_b_16"]:
        if hasattr(models, ctor_name):
            try:
                if resolved_vit_path is None:
                    weights_enum_name = ctor_name.upper() + "_Weights"
                    if hasattr(models, weights_enum_name):
                        weights_enum = getattr(models, weights_enum_name)
                        weights_for_transforms = weights_enum.DEFAULT
                        vit_model = getattr(models, ctor_name)(
                            weights=weights_enum.DEFAULT
                        )
                    else:
                        vit_model = getattr(models, ctor_name)(weights=None)
                        weights_for_transforms = None
                    using_custom_ckpt = False
                else:
                    vit_model = getattr(models, ctor_name)(weights=None)
                    using_custom_ckpt = True
                break
            except Exception as e:
                vit_ctor_errors.append((ctor_name, repr(e)))
    if vit_model is None:
        raise AttributeError(
            "No compatible ViT constructor found in torchvision.models. "
            f"Tried: {vit_ctor_errors}"
        )

    if using_custom_ckpt:
        vit_model = set_model_num_classes(vit_model, num_classes)
        state = torch.load(resolved_vit_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[len("module.") :] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        vit_model.load_state_dict(state, strict=True)

    vit_model.to(device)
    vit_model.eval()

    if hasattr(vit_model, "image_size"):
        model_image_size = int(vit_model.image_size)
    else:
        model_image_size = int(vit_image_size)

if model_select == "en":
    using_custom_ckpt = resolved_en_path is not None
    if using_custom_ckpt:
        en_model = models.efficientnet_v2_l(weights=None)
        en_model = set_model_num_classes(en_model, num_classes)
        state = torch.load(resolved_en_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[len("module.") :] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        en_model.load_state_dict(state, strict=True)
        weights_for_transforms = None
    else:
        if hasattr(models, "EfficientNet_V2_L_Weights"):
            weights_for_transforms = models.EfficientNet_V2_L_Weights.DEFAULT
            en_model = models.efficientnet_v2_l(weights=weights_for_transforms)
        else:
            en_model = models.efficientnet_v2_l(weights=None)
            weights_for_transforms = None
    en_model.to(device)
    en_model.eval()
    model_image_size = int(en_image_size)

if model_image_size is None:
    raise RuntimeError("model_image_size was not set; cannot build transforms.")

print("Using device:", device)
print("Model:", model_select, "image_size:", model_image_size)
print("Resolved vit checkpoint:", resolved_vit_path)
print("Resolved en checkpoint:", resolved_en_path)
print("Using custom ckpt:", using_custom_ckpt)
print(
    "Weights for transforms:",
    (
        type(weights_for_transforms).__name__
        if weights_for_transforms is not None
        else None
    ),
)



## === cell 4
if weights_for_transforms is not None and not using_custom_ckpt:
    w_t = weights_for_transforms.transforms()
    val_transforms = transforms.Compose([v2.Lambda(invert_square_pad), w_t])
else:
    val_transforms = transforms.Compose(
        [
            v2.Lambda(invert_square_pad),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((model_image_size, model_image_size)),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_test_image_ids(test_dir, sample_path):
    candidate_dirs = [
        test_dir,
        os.path.join(test_dir, "test_images"),
        os.path.join(
            os.path.dirname(test_dir),
            "cassava-leaf-disease-classification",
            "test_images",
        ),
        os.path.join(
            os.path.dirname(test_dir),
            "cassava-leaf-disease-classification",
            "test_images",
            "test_images",
        ),
    ]
    for d in candidate_dirs:
        if os.path.isdir(d):
            ids = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            if ids:
                ids.sort()
                return ids, d
    sample_df_local = pd.read_csv(sample_path)
    return sample_df_local["image_id"].tolist(), test_dir


def iter_tfrecord_images_and_labels(tfrec_paths, max_examples: int = 2500):
    """
    Read TFRecord files and parse tf.train.Example for fields:
      - image: bytes_list (JPEG bytes)
      - label/target: int64_list

    Change is targeted for score: the previous parser did not correctly decode the
    nested Example->Features->Feature protobuf structure, producing wrong/empty
    prototypes and near-random predictions. This is a correctness fix; the prototype
    logic and inference semantics remain identical.
    """

    def read_records(path):
        with open(path, "rb") as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes:
                    return
                (length,) = struct.unpack("<Q", length_bytes)
                f.read(4)  # length CRC (ignored)
                data = f.read(length)
                f.read(4)  # data CRC (ignored)
                if len(data) != length:
                    return
                yield data

    def read_varint(buf, i):
        shift = 0
        result = 0
        while True:
            if i >= len(buf):
                return None, i
            b = buf[i]
            i += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, i
            shift += 7

    def read_len_delimited(buf, i):
        ln, i = read_varint(buf, i)
        if ln is None or i + ln > len(buf):
            return None, i
        out = buf[i : i + ln]
        return out, i + ln

    def skip_field(buf, i, wire):
        if wire == 0:
            _, i = read_varint(buf, i)
            return i
        if wire == 1:
            return i + 8
        if wire == 2:
            _, i = read_len_delimited(buf, i)
            return i
        if wire == 5:
            return i + 4
        return len(buf)

    def parse_bytes_list_value_list(msg):
        vals = []
        j = 0
        while j < len(msg):
            key, j = read_varint(msg, j)
            if key is None:
                break
            field = key >> 3
            wire = key & 7
            if field == 1 and wire == 2:
                chunk, j = read_len_delimited(msg, j)
                if chunk is None:
                    break
                vals.append(chunk)
            else:
                j = skip_field(msg, j, wire)
        return vals

    def parse_int64_list_values(msg):
        vals = []
        j = 0
        while j < len(msg):
            key, j = read_varint(msg, j)
            if key is None:
                break
            field = key >> 3
            wire = key & 7
            if field == 1 and wire == 0:
                v, j = read_varint(msg, j)
                if v is None:
                    break
                vals.append(int(v))
            elif field == 1 and wire == 2:
                packed, j = read_len_delimited(msg, j)
                if packed is None:
                    break
                k = 0
                while k < len(packed):
                    v, k = read_varint(packed, k)
                    if v is None:
                        break
                    vals.append(int(v))
            else:
                j = skip_field(msg, j, wire)
        return vals

    def parse_feature_value(msg):
        image_bytes = None
        label_val = None
        j = 0
        while j < len(msg):
            key, j = read_varint(msg, j)
            if key is None:
                break
            field = key >> 3
            wire = key & 7
            if wire != 2:
                j = skip_field(msg, j, wire)
                continue
            sub, j = read_len_delimited(msg, j)
            if sub is None:
                break
            if field == 1:  # bytes_list
                vals = parse_bytes_list_value_list(sub)
                if vals:
                    image_bytes = vals[0]
            elif field == 3:  # int64_list
                vals = parse_int64_list_values(sub)
                if vals:
                    label_val = vals[0]
        return image_bytes, label_val

    def parse_features_map(features_msg):
        out = {}
        j = 0
        while j < len(features_msg):
            key, j = read_varint(features_msg, j)
            if key is None:
                break
            field = key >> 3
            wire = key & 7
            if field != 1 or wire != 2:
                j = skip_field(features_msg, j, wire)
                continue
            entry, j = read_len_delimited(features_msg, j)
            if entry is None:
                break

            k = 0
            mkey = None
            mval = None
            while k < len(entry):
                ekey, k = read_varint(entry, k)
                if ekey is None:
                    break
                efield = ekey >> 3
                ewire = ekey & 7
                if efield == 1 and ewire == 2:
                    s, k = read_len_delimited(entry, k)
                    if s is None:
                        break
                    mkey = s.decode("utf-8", errors="ignore")
                elif efield == 2 and ewire == 2:
                    mval, k = read_len_delimited(entry, k)
                    if mval is None:
                        break
                else:
                    k = skip_field(entry, k, ewire)

            if mkey is not None and mval is not None:
                out[mkey] = mval
        return out

    def parse_example(example_bytes):
        i = 0
        features_msg = None
        while i < len(example_bytes):
            key, i = read_varint(example_bytes, i)
            if key is None:
                break
            field = key >> 3
            wire = key & 7
            if field == 1 and wire == 2:
                features_msg, i = read_len_delimited(example_bytes, i)
                break
            i = skip_field(example_bytes, i, wire)
        if not features_msg:
            return None, None

        fmap = parse_features_map(features_msg)
        if not fmap:
            return None, None

        img_b = None
        lab = None

        for k in ("image", "image_raw", "jpeg"):
            if k in fmap:
                ib, _ = parse_feature_value(fmap[k])
                if ib is not None:
                    img_b = ib
                    break

        for k in ("label", "target"):
            if k in fmap:
                _, lv = parse_feature_value(fmap[k])
                if lv is not None:
                    lab = int(lv)
                    break

        return img_b, lab

    count = 0
    for p in tfrec_paths:
        for rec in read_records(p):
            img_b, lab = parse_example(rec)
            if img_b is None or lab is None:
                continue
            yield img_b, int(lab)
            count += 1
            if count >= max_examples:
                return


def compute_class_prototypes_from_tfrecords(
    model, tfrec_glob: str, max_examples: int = 2500, batch_size: int = 16
):
    tfrec_paths = sorted(glob.glob(tfrec_glob))
    if not tfrec_paths:
        print(
            "No TFRecords found for prototype computation; will fall back to argmax of model output."
        )
        return None

    sums = None
    counts = torch.zeros((num_classes,), dtype=torch.long, device=device)

    batch_imgs = []
    batch_labs = []

    it = iter_tfrecord_images_and_labels(tfrec_paths, max_examples=max_examples)

    for img_b, lab in tqdm(
        it, total=max_examples, desc="Prototypes(tfrec)", leave=False
    ):
        try:
            img = Image.open(io.BytesIO(img_b)).convert("RGB")
        except Exception:
            continue

        x = val_transforms(img)
        batch_imgs.append(x)
        batch_labs.append(int(lab))

        if len(batch_imgs) >= batch_size:
            xb = torch.stack(batch_imgs, dim=0).to(device)
            yb = torch.tensor(batch_labs, device=device, dtype=torch.long)

            with torch.no_grad():
                feat = extract_features(model, xb, model_select)
                emb = l2_normalize(feat)

            if sums is None:
                sums = torch.zeros(
                    (num_classes, emb.shape[1]), device=device, dtype=emb.dtype
                )

            for c in range(num_classes):
                m = yb == c
                if m.any():
                    sums[c] += emb[m].sum(dim=0)
                    counts[c] += int(m.sum().item())

            batch_imgs, batch_labs = [], []

    if len(batch_imgs) > 0:
        xb = torch.stack(batch_imgs, dim=0).to(device)
        yb = torch.tensor(batch_labs, device=device, dtype=torch.long)
        with torch.no_grad():
            feat = extract_features(model, xb, model_select)
            emb = l2_normalize(feat)
        if sums is None:
            sums = torch.zeros(
                (num_classes, emb.shape[1]), device=device, dtype=emb.dtype
            )
        for c in range(num_classes):
            m = yb == c
            if m.any():
                sums[c] += emb[m].sum(dim=0)
                counts[c] += int(m.sum().item())

    if sums is None:
        return None

    protos = torch.zeros_like(sums)
    for c in range(num_classes):
        if counts[c] > 0:
            protos[c] = sums[c] / counts[c].to(sums.dtype)

    protos = l2_normalize(protos)
    print("Prototype counts per class:", counts.detach().cpu().tolist())
    return protos


sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
test_image_ids, resolved_test_dir = get_test_image_ids(test_data_directory, sample_path)
print(
    "Resolved test image dir:",
    resolved_test_dir,
    "| num test ids:",
    len(test_image_ids),
)

active_model = vit_model if model_select == "vit" else en_model

prototypes = None
if not using_custom_ckpt:
    prototypes = compute_class_prototypes_from_tfrecords(
        active_model,
        tfrec_glob="/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
        max_examples=2500,  # keep cap to fit runtime budget; logic unchanged
        batch_size=16,
    )

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(resolved_test_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if using_custom_ckpt:
            output = active_model(transformed_image)
            if output.shape[-1] == 5:
                pred_int = int(torch.argmax(output, dim=1).item())
            else:
                pred_int = int(torch.argmax(output, dim=1).item()) % 5
        elif prototypes is not None:
            feat = extract_features(active_model, transformed_image, model_select)
            emb = l2_normalize(feat)
            sims = emb @ prototypes.T  # [1, C]
            pred_int = int(torch.argmax(sims, dim=1).item())
        else:
            output = active_model(transformed_image)
            pred_int = int(torch.argmax(output, dim=1).item())
            pred_int = pred_int % 5  # keep valid label space

    predictions.append(pred_int)
    image_ids.append(image_name)



## === cell 5
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    if len(predictions) > 0:
        fill_label = int(pd.Series(predictions).mode().iloc[0])
    else:
        fill_label = 0
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print("Rows:", len(submission_df), "Columns:", list(submission_df.columns))
print(submission_df.head())
print("Any NA labels:", submission_df["label"].isna().any())
