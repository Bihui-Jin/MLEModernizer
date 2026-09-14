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

0.8301601692354186

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10874) has done: 'I fix the missing model-weight paths by automatically falling back to torchvision’s built-in pretrained weights when the Kaggle dataset paths don’t exist, which also makes the script run end-to-end and typically improves accuracy vs random init. I also fix the CUDA/CPU dtype/device mismatch by ensuring the loaded state_dict tensors and the model are on the same device before inference. Finally, I harden the prediction loop so it always produces exactly one prediction per `image_id` (even if an image read fails) to prevent the submission length mismatch and guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should increase accuracy while keeping the same inference-only approach and the same model family. The biggest issue is that when your fine-tuned checkpoint paths don’t exist you replace the classifier head with a new random Linear layer, which destroys the usefulness of pretrained weights (and typically yields ~random accuracy like your 0.108). I keep the core logic the same (single-model inference loop + same transform pipeline concept) but (1) stop reinitializing the head when using torchvision pretrained weights, (2) switch normalization to the correct ImageNet stats for these pretrained weights, and (3) use each weights’ recommended resize/crop size to match training-time preprocessing. These are minimal changes that should move the score substantially toward your 0.83 target without changing the overall approach.'
- What this solution (achieved 0.05643) has done: 'Your current 0.0 suggests the submission is valid but predictions are essentially unusable; the most likely cause is that your intended fine-tuned checkpoint isn’t being found/loaded, so you’re effectively relying on ImageNet weights that are badly mismatched to cassava labels. To move the score toward 0.83 with minimal changes and without altering your inference-only core logic, I (1) automatically fall back to using the competition’s `train.csv` to compute the ImageNet model’s confusion-matrix-based class mapping (a standard “label remapping” fix when using an out-of-domain pretrained classifier), and (2) apply that mapping to test predictions. This keeps the same model, transforms, and argmax inference, but fixes the label-ID semantics to align better with cassava classes; if the fine-tuned checkpoint exists, nothing changes. The script still writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that keep your inference-only pipeline intact. The biggest likely issue is the aggressive `invert_square_pad` transform: it creates an unnatural “quadrant swap” image that badly mismatches ImageNet pretraining, so I keep only the square reflect-pad (no quadrant inversion) while preserving your overall transform structure. Next, when falling back to torchvision pretrained weights, I use each weights object’s exact recommended preprocessing (mean/std + resize/crop) to better match the pretrained model’s expectations. Finally, I make the label-remap calibration more representative (stratified-ish via shuffle + larger but still fast cap) while keeping the same confusion-matrix argmax remapping logic.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import os
import random



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
en_image_size = 480

vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/3/vit_h_14_518_8927.pth"
vit_image_size = 518

model_select = "en"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size

random.seed(0)
torch.manual_seed(0)




## === cell 2
def square_reflect_pad(img):
    """
    Change is directly to improve accuracy: removing the quadrant inversion (which corrupts semantics)
    while preserving your core idea of making the image square via reflect padding.
    """
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )
    return transforms.functional.pad(img, padding, padding_mode="reflect")




## === cell 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose(
    [
        v2.Lambda(square_reflect_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)




## === cell 4
def safe_load_state_dict(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


using_finetuned_ckpt = False

if model_select == "vit":
    if os.path.exists(vit_model_path):
        using_finetuned_ckpt = True
        vit_model = models.vit_h_14(weights=None, image_size=518)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        vit_model.load_state_dict(
            safe_load_state_dict(vit_model_path, map_location=device)
        )
    else:
        vit_weights = models.ViT_H_14_Weights.DEFAULT
        vit_model = models.vit_h_14(weights=vit_weights, image_size=518)
        wt = vit_weights.transforms()
        val_transforms = transforms.Compose(
            [
                v2.Lambda(square_reflect_pad),
                wt,  # includes resize/crop + to_tensor + normalize as expected by these weights
            ]
        )

    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    if os.path.exists(en_model_path):
        using_finetuned_ckpt = True
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        en_model.load_state_dict(
            safe_load_state_dict(en_model_path, map_location=device)
        )
    else:
        en_weights = models.EfficientNet_V2_L_Weights.DEFAULT
        en_model = models.efficientnet_v2_l(weights=en_weights)
        wt = en_weights.transforms()
        val_transforms = transforms.Compose(
            [
                v2.Lambda(square_reflect_pad),
                wt,  # includes resize/crop + to_tensor + normalize as expected by these weights
            ]
        )

    en_model.to(device)
    en_model.eval()




## === cell 5
def build_label_remap_from_train(
    model,
    transforms_fn,
    train_csv,
    train_images_dir,
    device,
    num_classes=5,
    max_samples=6000,
):
    """
    Same core logic as your remap (confusion-matrix argmax), but with a more representative subset:
    shuffle before slicing and use a somewhat larger cap to reduce noise and improve accuracy.
    """
    df = pd.read_csv(train_csv)
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    df = df.iloc[: min(len(df), max_samples)].copy()

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # [pred, true]
    model.eval()

    for img_id, y in tqdm(
        zip(df["image_id"].tolist(), df["label"].tolist()),
        total=len(df),
        desc="Calibrating label map",
    ):
        path = os.path.join(train_images_dir, img_id)
        try:
            img = Image.open(path).convert("RGB")
            x = transforms_fn(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(x)
                p = int(torch.argmax(logits, dim=1).item())
            y = int(y)
            if 0 <= p < num_classes and 0 <= y < num_classes:
                cm[p, y] += 1
        except Exception:
            continue

    remap = list(range(num_classes))
    for p in range(num_classes):
        if cm[p].sum().item() > 0:
            remap[p] = int(torch.argmax(cm[p]).item())
        else:
            remap[p] = p
    return remap, cm


label_remap = None
if not using_finetuned_ckpt:
    train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if model_select == "vit":
        label_remap, _cm = build_label_remap_from_train(
            vit_model,
            val_transforms,
            train_csv_path,
            train_images_dir,
            device,
            num_classes=num_classes,
            max_samples=6000,
        )
    else:
        label_remap, _cm = build_label_remap_from_train(
            en_model,
            val_transforms,
            train_csv_path,
            train_images_dir,
            device,
            num_classes=num_classes,
            max_samples=6000,
        )
    print("Using label remap (pred->label):", label_remap)



## === cell 6
sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

predictions = []
for image_name in tqdm(image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)

    try:
        image = Image.open(image_path).convert("RGB")
        transformed_image = val_transforms(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            if model_select == "vit":
                output = vit_model(transformed_image)
            else:
                output = en_model(transformed_image)
            predicted_class = int(torch.argmax(output, dim=1).item())

        if label_remap is not None:
            predicted_class = int(label_remap[predicted_class])

        predictions.append(predicted_class)
    except Exception:
        predictions.append(0)

assert len(predictions) == len(
    image_ids
), f"predictions ({len(predictions)}) != image_ids ({len(image_ids)})"



## === cell 7
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
