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

0.7805983680870353

# 6. Current score

0.19544

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1491) has done: 'I fix the immediate runtime/import failure by removing the TensorFlow/Keras dependency that triggers the `MessageFactory.GetPrototype` error, and I also remove the unused TFRecord parsing code. Next, I make the solution robust to missing external model weight files by falling back to a torchvision ViT with ImageNet weights (same inference loop, no training), so the notebook always runs end-to-end. Finally, I generate the submission using `sample_submission.csv` as the authoritative test image order/length (and only predict those ids), which fixes the “Invalid submission length” issue and guarantees correct formatting with a `.csv` suffix.'
- What this solution (achieved 0.179) has done: 'I fix the ViT loading crash by keeping the ImageNet-weighted fallback at its required `image_size=224` (torchvision enforces this), while still allowing custom cassava checkpoints to use `vit_image_size=512` if compatible. Then I make inference robust by ensuring `vit_model` is always defined (so cell 5 can run), and I guarantee the submission lengths match by deriving `test_image_ids` from `sample_submission.csv` and hard-aligning `predictions` to that length right before writing. These changes are execution-unblocking and should also increase score versus random/empty output by producing real model predictions. The core approach (inference-only ViT, argmax class) is preserved.'
- What this solution (achieved 0.19544) has done: 'Your current score is far below the target, and the biggest issue is that the ImageNet ViT fallback is being used with a randomly initialized classification head (5-way), which makes predictions near-random. To move accuracy toward the target with minimal core-logic change (still inference-only ViT + argmax), we build the ViT so it can load ImageNet weights while cleanly replacing the head, use the weights’ own preprocessing (which matches training normalization), and force `use_224_preprocess` whenever we’re not actually using a compatible custom 512px cassava checkpoint. These changes keep the same model family and inference semantics, but should substantially improve accuracy versus the current random-head behavior. The submission writing and `sample_submission.csv` alignment are preserved.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_data_directory = f"{DATA_DIR}/test_images"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"
resnet_model_path = "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"

vit_image_size = 512



## === cell 2
vit_preprocess = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def _strip_prefix_if_present(
    state_dict, prefixes=("module.", "model.", "net.", "encoder.")
):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    return new_sd


def _load_vit_model(device, num_classes, vit_image_size, vit_model_path):
    """
    Score-moving fix (keeps core logic: ViT inference-only, argmax):
    - When using torchvision ImageNet weights, ensure we *load the pretrained backbone*
      and only replace the classification head AFTER loading weights (so head is not random + backbone is pretrained).
    - Only use 512px preprocessing when a compatible custom checkpoint is actually loaded with non-trivial tensors.
      Otherwise force the 224px path (weights.meta["min_size"] == 224).
    Returns:
      vit_model, use_224_preprocess (bool)
    """
    use_224_preprocess = True
    weights_source = None

    if os.path.exists(vit_model_path):
        vit_model = models.vit_l_16(weights=None, image_size=vit_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )

        state = torch.load(vit_model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix_if_present(state)

        model_sd = vit_model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if (k in model_sd and model_sd[k].shape == v.shape)
        }

        vit_model.load_state_dict(filtered, strict=False)

        if len(filtered) > 50:
            use_224_preprocess = False
            weights_source = f"custom cassava weights @ {vit_image_size}px (loaded {len(filtered)}/{len(state)} tensors)"
        else:
            weights = models.ViT_L_16_Weights.DEFAULT
            vit_model = models.vit_l_16(weights=weights)  # image_size fixed to 224
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
            use_224_preprocess = True
            weights_source = "torchvision ImageNet weights @ 224px (fallback; checkpoint missing/incompatible)"
    else:
        weights = models.ViT_L_16_Weights.DEFAULT
        vit_model = models.vit_l_16(weights=weights)  # image_size fixed to 224
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        use_224_preprocess = True
        weights_source = (
            "torchvision ImageNet weights @ 224px (fallback; checkpoint missing)"
        )

    vit_model.to(device)
    vit_model.eval()
    print(f"ViT loaded using: {weights_source}")
    return vit_model, use_224_preprocess


vit_model, use_224_preprocess = _load_vit_model(
    device, num_classes, vit_image_size, vit_model_path
)



## === cell 4
use_resnet = False
resnet_model = None

if os.path.exists(resnet_model_path):
    use_resnet = False

print(
    f"ResNet enabled: {use_resnet} (model file exists: {os.path.exists(resnet_model_path)})"
)



## === cell 5
vit_preprocess_224 = models.ViT_L_16_Weights.DEFAULT.transforms()

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

predictions = []
missing_images = 0

for image_name in tqdm(test_image_ids, desc="Predict"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.exists(image_path):
        missing_images += 1
        predictions.append(0)
        continue

    image = Image.open(image_path).convert("RGB")
    if use_224_preprocess:
        vit_image = vit_preprocess_224(image).unsqueeze(0).to(device)
    else:
        vit_image = vit_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vit_output = vit_model(vit_image)
        vit_pred = int(torch.argmax(vit_output, dim=1).item())

    predictions.append(vit_pred)

if missing_images:
    print(
        f"Warning: {missing_images} test images were missing; filled with label=0 to keep valid length."
    )

n_expected = len(test_image_ids)
if len(predictions) != n_expected:
    print(
        f"Warning: predictions length {len(predictions)} != expected {n_expected}. Fixing by trim/pad."
    )
    if len(predictions) > n_expected:
        predictions = predictions[:n_expected]
    else:
        predictions = predictions + [0] * (n_expected - len(predictions))



## === cell 6
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_sub)})")
