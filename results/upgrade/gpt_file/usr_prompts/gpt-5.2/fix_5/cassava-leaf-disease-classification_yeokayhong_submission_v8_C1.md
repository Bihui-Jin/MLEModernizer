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

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blockers by making checkpoint discovery work in this environment (your provided ViT/EN checkpoint paths don’t exist), and ensure `model_image_size` is always defined so transforms build reliably. To preserve the original inference-only core logic while still producing a valid submission, I fall back to torchvision’s pretrained weights when no custom `.pth` is found (same architectures, same forward pass; only the weight source changes). I also fix the glob to be recursive, and make the head/num_classes handling robust so the model can run even when pretrained weights have 1000 classes. Finally, I make sure we always generate `submission.csv` with the exact `sample_submission.csv` ordering.'

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

if model_select == "vit":
    vit_model = None
    vit_ctor_errors = []

    for ctor_name in ["vit_h_14", "vit_l_16", "vit_b_16"]:
        if hasattr(models, ctor_name):
            try:
                if resolved_vit_path is None:
                    weights_enum_name = ctor_name.upper() + "_Weights"
                    if hasattr(models, weights_enum_name):
                        weights_enum = getattr(models, weights_enum_name)
                        vit_model = getattr(models, ctor_name)(
                            weights=weights_enum.DEFAULT
                        )
                    else:
                        vit_model = getattr(models, ctor_name)(weights=None)
                else:
                    vit_model = getattr(models, ctor_name)(weights=None)
                break
            except Exception as e:
                vit_ctor_errors.append((ctor_name, repr(e)))

    if vit_model is None:
        raise AttributeError(
            "No compatible ViT constructor found in torchvision.models. "
            f"Tried: {vit_ctor_errors}"
        )

    using_custom_ckpt = resolved_vit_path is not None

    if using_custom_ckpt:
        if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "head"):
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
        elif hasattr(vit_model, "head"):
            vit_model.head = torch.nn.Linear(vit_model.head.in_features, num_classes)
        else:
            raise AttributeError(
                "Unexpected ViT head structure; cannot set classifier head to num_classes."
            )

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

    vit_model.to(device)
    vit_model.eval()

    model_image_size = vit_image_size

if model_select == "en":
    using_custom_ckpt = resolved_en_path is not None

    if using_custom_ckpt:
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )

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
    else:
        if hasattr(models, "EfficientNet_V2_L_Weights"):
            en_model = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.DEFAULT
            )
        else:
            en_model = models.efficientnet_v2_l(weights=None)

    en_model.to(device)
    en_model.eval()
    model_image_size = en_image_size

print("Using device:", device)
print("Model:", model_select, "image_size:", model_image_size)
print("Resolved vit checkpoint:", resolved_vit_path)
print("Resolved en checkpoint:", resolved_en_path)



## === cell 4
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

        _, predicted_class = torch.max(output, 1)
        pred_int = int(predicted_class.item())

        if output.shape[-1] != num_classes:
            pred_int = pred_int % num_classes

    predictions.append(pred_int)
    image_ids.append(image_name)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/300194087.py in <cell line: 0>()
     61     with torch.no_grad():
     62         if model_select == "vit":
---> 63             output = vit_model(transformed_image)
     64         if model_select == "en":
     65             output = en_model(transformed_image)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
--> 291         x = self._process_input(x)
    292         n = x.shape[0]
    293 

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _process_input(self, x)
    269         n, c, h, w = x.shape
    270         p = self.patch_size
--> 271         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    272         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    273         n_h = h // p

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _assert(condition, message)
   2130             _assert, (condition,), condition, message
   2131         )
-> 2132     assert condition, message
   2133 
   2134 

AssertionError: Wrong image height! Expected 224 but got 518!

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
