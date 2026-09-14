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

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blockers by making checkpoint discovery work in this environment (your provided ViT/EN checkpoint paths don’t exist), and ensure `model_image_size` is always defined so transforms build reliably. To preserve the original inference-only core logic while still producing a valid submission, I fall back to torchvision’s pretrained weights when no custom `.pth` is found (same architectures, same forward pass; only the weight source changes). I also fix the glob to be recursive, and make the head/num_classes handling robust so the model can run even when pretrained weights have 1000 classes. Finally, I make sure we always generate `submission.csv` with the exact `sample_submission.csv` ordering.'
- What this solution (achieved 0.40209) has done: 'I fix the ViT input-size mismatch that causes the runtime `AssertionError` by resizing images to the actual `vit_model.image_size` (or EfficientNet’s expected size) rather than the hardcoded checkpoint image size. This is a correctness fix (not a model change) and should also materially improve accuracy because the model now receive inputs at the resolution it was designed for, instead of failing or being mismatched. I also make the classifier-head handling robust when falling back to torchvision pretrained weights so the model always outputs 5 classes without the current “mod num_classes” fallback that destroys accuracy. Finally, I keep the submission ordering aligned to `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.40209) is far below the target (0.88546), so we should improve accuracy with minimal, semantics-preserving fixes. The biggest issue is that when no custom cassava checkpoint is found, the code replaces the 1000-class ImageNet head with a fresh 5-class head, which makes predictions essentially random; instead, we should keep the pretrained head and map its logits to 5 cassava classes via a fixed ImageNet→cassava class mapping. We also switch normalization to use the model’s official pretrained weights transforms (when using torchvision pretrained weights), which is a correctness fix for inference-time preprocessing. These changes keep the same architectures and inference-only approach, and should move the score substantially toward the target.'

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
    For torchvision pretrained (ImageNet) inference fallback, we must NOT replace the head,
    otherwise we destroy all learned logits and get near-random accuracy.
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


IMAGENET_TO_CASSAVA = {
    0: [  # Cassava Bacterial Blight (CBB) - map to "blight/spot/lesion" type classes
        312,
        313,
        314,
        315,
        316,
        317,  # crickets (texture-y; weak)
        985,
        986,
        987,
        988,
        989,  # daisy-like; weak
        948,
        949,  # 'honeycomb' etc; weak
        999,  # 'toilet tissue'; weak
        825,
        826,
        827,  # fungus family; weak
        721,
        722,  # leaf beetle; weak
        875,
        876,
        877,  # "mushroom" like; weak
        309,
        310,
        311,  # "bee" like; weak
        947,  # mushroom; weak
        991,
        992,
        993,
        994,
        995,
        996,
        997,
        998,  # misc textures
    ],
    1: [  # Cassava Brown Streak Disease (CBSD) - "wilt/leaf curl/dry"
        948,
        949,
        950,
        951,
        952,
        323,
        324,
        325,
        326,
        327,
        328,  # monarch/butterfly-ish; weak
        983,
        984,
        985,
        986,
        815,
        816,
        817,
        818,  # plants/misc; weak
        760,
        761,
        762,  # "artichoke" etc; weak
    ],
    2: [  # Cassava Green Mottle (CGM) - "mottle/mosaic"
        985,
        986,
        987,
        988,
        989,
        815,
        816,
        817,
        818,
        537,
        538,
        539,  # 'leaf beetle' etc; weak
        936,
        937,
        938,  # "mushroom" etc; weak
        649,
        650,
        651,  # "maze" textures; weak
    ],
    3: [  # Cassava Mosaic Disease (CMD) - strongest "mosaic/variegation"
        815,
        816,
        817,
        818,
        985,
        986,
        987,
        988,
        989,
        936,
        937,
        938,
        652,
        653,
        654,
        999,
    ],
    4: [  # Healthy
        948,
        949,
        950,
        951,
        952,
        815,
        816,
        817,
        818,
        980,
        981,
        982,
        983,
        984,  # flowers/plant-like; weak
        644,
        645,
        646,
        647,
        648,  # "maze" textures; weak
    ],
}


def imagenet_logits_to_cassava_logits(logits_1000: torch.Tensor) -> torch.Tensor:
    bsz = logits_1000.shape[0]
    out = torch.empty((bsz, 5), device=logits_1000.device, dtype=logits_1000.dtype)
    for c in range(5):
        idx = IMAGENET_TO_CASSAVA.get(c, [])
        if len(idx) == 0:
            out[:, c] = logits_1000[:, 0] * 0.0  # zeros
        else:
            out[:, c] = torch.logsumexp(logits_1000[:, idx], dim=1)
    return out


resolved_vit_path = resolve_checkpoint_path(
    vit_model_path,
    "/kaggle/input/**/vit*.pth",
)
resolved_en_path = resolve_checkpoint_path(
    en_model_path,
    "/kaggle/input/**/efficientnet*.pth",
)

vit_model = None
en_model = None

model_image_size = None
weights_for_transforms = None
using_custom_ckpt = False
use_imagenet_head_mapping = False

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
                    use_imagenet_head_mapping = True
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
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                new_state[nk] = v
            state = new_state

        vit_model.load_state_dict(state, strict=True)
        use_imagenet_head_mapping = False

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
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                new_state[nk] = v
            state = new_state

        en_model.load_state_dict(state, strict=True)
        use_imagenet_head_mapping = False
        weights_for_transforms = None
    else:
        if hasattr(models, "EfficientNet_V2_L_Weights"):
            weights_for_transforms = models.EfficientNet_V2_L_Weights.DEFAULT
            en_model = models.efficientnet_v2_l(weights=weights_for_transforms)
        else:
            en_model = models.efficientnet_v2_l(weights=None)
            weights_for_transforms = None
        use_imagenet_head_mapping = True

    en_model.to(device)
    en_model.eval()
    model_image_size = int(en_image_size)

if model_image_size is None:
    raise RuntimeError("model_image_size was not set; cannot build transforms.")

print("Using device:", device)
print("Model:", model_select, "image_size:", model_image_size)
print("Resolved vit checkpoint:", resolved_vit_path)
print("Resolved en checkpoint:", resolved_en_path)
print(
    "Using custom ckpt:",
    using_custom_ckpt,
    "| Using ImageNet head mapping:",
    use_imagenet_head_mapping,
)
print(
    "Weights for transforms:",
    (
        type(weights_for_transforms).__name__
        if weights_for_transforms is not None
        else None
    ),
)



## === cell 4
if weights_for_transforms is not None and use_imagenet_head_mapping:
    val_transforms = weights_for_transforms.transforms()
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


sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
test_image_ids, resolved_test_dir = get_test_image_ids(test_data_directory, sample_path)
print(
    "Resolved test image dir:",
    resolved_test_dir,
    "| num test ids:",
    len(test_image_ids),
)

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(resolved_test_dir, image_name)

    image = Image.open(image_path).convert("RGB")
    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_select == "vit":
            output = vit_model(transformed_image)
        if model_select == "en":
            output = en_model(transformed_image)

        if use_imagenet_head_mapping:
            if output.shape[-1] == 1000:
                output = imagenet_logits_to_cassava_logits(output)

        _, predicted_class = torch.max(output, 1)
        pred_int = int(predicted_class.item())

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
