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

0.8553943789664551

# 6. Current score

0.32922

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11958) has done: 'I fix the environment-breaking TensorFlow import issue by removing TensorFlow/Keras usage (it isn’t actually used for inference in your current pipeline) to avoid the protobuf `MessageFactory` crash. I also make the test image directory resolution robust to the nested `test_images/test_images` folder and ensure we only iterate over actual image files, which fixes the `IsADirectoryError` and the wrong submission length. Finally, since the EfficientNet weight path you referenced doesn’t exist in this environment, I fall back to a standard torchvision EfficientNetV2-M pretrained model (same architecture call) so the notebook runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved 0.18348) has done: 'Your current score is extremely low because the model you use for inference has a randomly initialized 5-class head (you overwrite the pretrained 1000-class head but never load cassava-trained weights), so predictions are close to random. To move the accuracy up toward the 0.855 target while keeping your core approach (single EfficientNet inference loop + argmax) unchanged, I load cassava-trained weights if they exist, but otherwise I *not* replace the pretrained head and instead map ImageNet-1000 predictions into 5 classes via a small, fixed grouping heuristic (this keeps the same model and inference semantics but makes predictions far less random). I also ensure deterministic file ordering and keep the submission aligned to `sample_submission.csv` exactly as you already do. These are minimal changes aimed at a large lift from ~0.12 toward the target without changing the training approach (still none) or introducing extra packages.'
- What this solution (achieved 0.32922) has done: 'Your score is low because when cassava weights aren’t found you’re effectively producing near-random 5-class outputs by taking `argmax` over 1000 ImageNet classes and then doing `% 5`, which has no relationship to cassava diseases. To move accuracy up toward the 0.855 target while keeping the same single-model inference + argmax core logic, I keep the EfficientNetV2-M backbone exactly as-is but replace the `% 5` hack with a minimal, data-driven mapping from ImageNet-1000 predictions into 5 labels using nearest-class prototypes computed from the cassava training images (no training loop, no new architecture). This keeps evaluation semantics (still argmax-based classification) while making predictions meaningfully aligned to the cassava label space. I also keep the submission aligned to `sample_submission.csv` exactly, and ensure deterministic ordering.'

# 9. Code solution

## === cell 0
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = (
    "/kaggle/input/efficientnetv2-large-test/pytorch/default/1/ENL_V2 (test).pth"
)
vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"
vit_image_size = 518
resnet_model_path = "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"



## === cell 2
vit_preprocess = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def _resolve_test_dir(base_dir: str) -> str:
    """
    Kaggle dataset sometimes contains nested folder test_images/test_images.
    Return the directory that actually contains jpg/png files.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Test directory not found: {base_dir}")

    nested = os.path.join(base_dir, "test_images")

    def has_images(d):
        if not os.path.isdir(d):
            return False
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(
                os.path.join(d, fn)
            ):
                return True
        return False

    if has_images(base_dir):
        return base_dir
    if has_images(nested):
        return nested

    for fn in os.listdir(base_dir):
        p = os.path.join(base_dir, fn)
        if os.path.isdir(p) and has_images(p):
            return p

    raise FileNotFoundError(f"No image files found under: {base_dir}")


test_data_directory = _resolve_test_dir(test_data_directory)
print("Using test image directory:", test_data_directory)



## === cell 4
using_cassava_head = False

if os.path.exists(en_model_path):
    en_model = models.efficientnet_v2_m(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    state = torch.load(en_model_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    en_model.load_state_dict(state, strict=False)
    using_cassava_head = True
    print("Loaded EfficientNet cassava weights from:", en_model_path)
else:
    en_model = models.efficientnet_v2_m(
        weights=models.EfficientNet_V2_M_Weights.DEFAULT
    )
    using_cassava_head = False
    print(
        "Custom cassava weights not found; using torchvision ImageNet pretrained model with original 1000-class head."
    )

en_model.to(device)
en_model.eval()



## === cell 5
en_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 6
def _resolve_train_dir(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Train directory not found: {base_dir}")

    nested = os.path.join(base_dir, "train_images")

    def has_images(d):
        if not os.path.isdir(d):
            return False
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(
                os.path.join(d, fn)
            ):
                return True
        return False

    if has_images(base_dir):
        return base_dir
    if has_images(nested):
        return nested

    for fn in os.listdir(base_dir):
        p = os.path.join(base_dir, fn)
        if os.path.isdir(p) and has_images(p):
            return p

    raise FileNotFoundError(f"No image files found under: {base_dir}")


def _build_imagenet_to_cassava_mapping(
    model,
    preprocess,
    device,
    train_csv_path: str,
    train_images_dir: str,
    num_classes: int = 5,
    per_class_limit: int = 80,
) -> np.ndarray:
    """
    Returns mapping array of shape [1000], where mapping[i] in {0..4} is the cassava label
    assigned to ImageNet class i, based on nearest label prototype in the model's penultimate
    embedding space.
    """
    model.eval()

    backbone = model.features
    avgpool = model.avgpool

    def embed_one(pil_img: Image.Image) -> np.ndarray:
        x = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(x)
            feat = avgpool(feat)
            feat = torch.flatten(feat, 1)
        v = feat.detach().cpu().float().numpy()[0]
        n = np.linalg.norm(v) + 1e-12
        return v / n

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.sort_values("image_id").reset_index(drop=True)

    sums = None
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, row in tqdm(
        train_df.iterrows(), total=len(train_df), desc="Build prototypes"
    ):
        y = int(row["label"])
        if y < 0 or y >= num_classes:
            continue
        if counts[y] >= per_class_limit:
            continue

        img_path = os.path.join(train_images_dir, row["image_id"])
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        v = embed_one(img)

        if sums is None:
            sums = np.zeros((num_classes, v.shape[0]), dtype=np.float64)
        sums[y] += v.astype(np.float64)
        counts[y] += 1

        if int(counts.min()) >= per_class_limit:
            break

    if sums is None or (counts == 0).any():
        raise RuntimeError(
            f"Failed to build prototypes (counts per class: {counts.tolist()})."
        )

    protos = sums / counts[:, None]
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)

    W = model.classifier[1].weight.detach().cpu().float().numpy()  # [1000, dim]
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    sims = W @ protos.T  # [1000, 5]
    mapping = sims.argmax(axis=1).astype(np.int64)  # [1000]
    return mapping




## === cell 7
imagenet_to_cassava = None
if not using_cassava_head:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_images_dir = _resolve_train_dir(
        "/kaggle/input/cassava-leaf-disease-classification/train_images"
    )
    print("Using train image directory for prototypes:", train_images_dir)

    imagenet_to_cassava = _build_imagenet_to_cassava_mapping(
        model=en_model,
        preprocess=en_preprocess,
        device=device,
        train_csv_path=train_csv_path,
        train_images_dir=train_images_dir,
        num_classes=num_classes,
        per_class_limit=80,
    )
    print("Built ImageNet->cassava mapping:", imagenet_to_cassava.shape)



## === cell 8
en_predictions = []
image_ids = []

test_files = [
    fn
    for fn in os.listdir(test_data_directory)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(test_data_directory, fn))
]
test_files = sorted(test_files)

for image_name in tqdm(test_files, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)
    image = Image.open(image_path).convert("RGB")

    x = en_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = en_model(x)
        pred_idx = int(out.argmax(dim=1).item())

    if using_cassava_head:
        pred = pred_idx
    else:
        pred = int(imagenet_to_cassava[pred_idx])

    en_predictions.append(int(pred))
    image_ids.append(image_name)

print(
    "Predictions:",
    len(en_predictions),
    "Images:",
    len(image_ids),
    "using_cassava_head:",
    using_cassava_head,
)



## === cell 9
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)

pred_map = dict(zip(image_ids, en_predictions))

sample_df["label"] = sample_df["image_id"].map(pred_map).fillna(0).astype(int)

sample_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(sample_df.head())
print("Submission rows:", len(sample_df))
