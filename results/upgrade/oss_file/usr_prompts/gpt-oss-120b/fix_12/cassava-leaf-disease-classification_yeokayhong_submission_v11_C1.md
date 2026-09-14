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

0.8872771229978845

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51009) has done: 'I fix the weight‑enum name for the ViT fallback, ensure the model is always moved to the same device as the inputs, and add a small safety check before writing the submission so its length matches the test set. These changes resolve the AttributeError, the device‑type mismatch, and the invalid‑submission error while keeping the original architecture and training logic intact.'
- What this solution (achieved 0.23244) has done: 'The change updates the preprocessing to use the standard ImageNet mean‑std values expected by both ViT‑H‑14 and EfficientNet‑V2‑L, which aligns the inputs with the fine‑tuned checkpoints and should raise validation accuracy toward the target score. No architectural or training logic is altered.'
- What this solution (achieved 0.15396) has done: 'I make two small but impactful fixes:  
1. Automatically locate the ViT and EfficientNet checkpoint files under any Kaggle input directory instead of using hard‑coded paths that may not exist, ensuring the model loads the fine‑tuned weights rather than falling back to random ImageNet heads.  
2. Remove the custom “invert_square_pad” preprocessing step, which distorts images and hurts accuracy, and keep a standard resize‑to‑model‑size pipeline that matches the training preprocessing of the checkpoints. These minimal changes keep the original architecture and inference flow while moving the validation accuracy much closer to the target score.'
- What this solution (achieved 0.0938) has done: 'I load both ViT‑H‑14 and EfficientNet‑V2‑L (using the same pretrained‑weight fallback logic) and ensemble their logits at inference time, which typically yields a modest boost in accuracy and therefore moves the current score closer to the target. The rest of the pipeline, preprocessing and CSV output, remains unchanged.'
- What this solution (achieved 0.09417) has done: 'I modify the checkpoint loading to use `strict=False` so that mismatched keys from the fine‑tuned `.pth` files no longer cause a fallback to ImageNet weights. This small change keeps the original architecture and inference pipeline intact while allowing the pretrained fine‑tuned weights to be applied, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.15396) has done: 'I adjust the preprocessing so each model gets images at the size it expects and then run only the selected model (default “vit”). This fixes the major mismatch that was driving the very low accuracy while keeping the overall architecture and inference flow unchanged.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
from PIL import Image

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.transforms.v2 as v2

from torchvision.models import ViT_H_14_Weights, EfficientNet_V2_L_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)

num_classes = 5
model_select = "ensemble"  # choose "vit", "en", or "ensemble"


def find_checkpoint(name_substring: str) -> str | None:
    for p in Path("/kaggle/input").rglob("*.pth"):
        if name_substring.lower() in p.name.lower():
            return str(p)
    return None


en_model_path = find_checkpoint("efficientnetv2")
en_image_size = 480
vit_model_path = find_checkpoint("vit_h_14")
vit_image_size = 224

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"




## === cell 1
def invert_square_pad(img):
    """Flip quadrants of the image and pad to a square shape."""
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

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )
    padded_img = TF.pad(flipped_img, padding, padding_mode="reflect")
    return padded_img




## === cell 2
vit_transform = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((vit_image_size, vit_image_size)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

en_transform = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((en_image_size, en_image_size)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
try:
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
    if vit_model_path is None:
        raise FileNotFoundError("ViT checkpoint not found.")
    state_dict = torch.load(vit_model_path, map_location=device, weights_only=True)
    vit_model.load_state_dict(state_dict, strict=False)
except Exception as e:
    print(f"Custom ViT checkpoint not loaded ({e}), using ImageNet pretrained weights.")
    vit_model = models.vit_h_14(weights=ViT_H_14_Weights.DEFAULT)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
vit_model = vit_model.to(device)
vit_model.eval()

try:
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    if en_model_path is None:
        raise FileNotFoundError("EfficientNet checkpoint not found.")
    state_dict = torch.load(en_model_path, map_location=device, weights_only=True)
    en_model.load_state_dict(state_dict, strict=False)
except Exception as e:
    print(
        f"Custom EfficientNet checkpoint not loaded ({e}), using ImageNet pretrained weights."
    )
    en_model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
en_model = en_model.to(device)
en_model.eval()



## === cell 4
predictions = []
image_ids = []

test_files = sorted(
    [f for f in os.listdir(test_data_directory) if f.lower().endswith(".jpg")]
)

batch_size = 32

if model_select not in {"vit", "en", "ensemble"}:
    raise ValueError("model_select must be 'vit', 'en', or 'ensemble'")

for start_idx in tqdm(range(0, len(test_files), batch_size), desc="Test"):
    batch_files = test_files[start_idx : start_idx + batch_size]

    vit_batch_tensors = []
    en_batch_tensors = []
    batch_names = []

    for image_name in batch_files:
        image_path = os.path.join(test_data_directory, image_name)
        image = Image.open(image_path).convert("RGB")
        batch_names.append(image_name)

        vit_tensor = vit_transform(image)
        en_tensor = en_transform(image)

        vit_batch_tensors.append(vit_tensor)
        en_batch_tensors.append(en_tensor)

    vit_input = torch.stack(vit_batch_tensors).to(device)
    en_input = torch.stack(en_batch_tensors).to(device)

    with torch.no_grad():
        if model_select == "vit":
            logits = vit_model(vit_input)
        elif model_select == "en":
            logits = en_model(en_input)
        else:  # ensemble
            vit_logits = vit_model(vit_input)
            en_logits = en_model(en_input)
            logits = (vit_logits + en_logits) / 2

    _, batch_pred = torch.max(logits, 1)
    predictions.extend(batch_pred.cpu().tolist())
    image_ids.extend(batch_names)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_54/1095148798.py in <cell line: 0>()
     38             logits = en_model(en_input)
     39         else:  # ensemble
---> 40             vit_logits = vit_model(vit_input)
     41             en_logits = en_model(en_input)
     42             logits = (vit_logits + en_logits) / 2

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

AssertionError: Wrong image height! Expected 518 but got 224!

## === cell 5
assert len(predictions) == len(test_files), "Prediction length mismatch!"

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path} ({len(submission_df)} rows)")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_54/2529372579.py in <cell line: 0>()
----> 1 assert len(predictions) == len(test_files), "Prediction length mismatch!"
      2 
      3 submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
      4 submission_path = "submission.csv"
      5 submission_df.to_csv(submission_path, index=False)

AssertionError: Prediction length mismatch!
