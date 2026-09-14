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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.8925657298277425

# 6. Current score

0.1009

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.321) has done: 'I fix the runtime failure by making the code robust to the missing external weights directory (`/kaggle/input/cassva-models-se50-640`) and to the unavailable pip install source, so the notebook runs in this environment. Since the pretrained weights aren’t available, I fall back to using timm’s built-in pretrained ImageNet weights (same model architecture) to produce reasonable predictions and a valid `submission.csv`. I also fix the TTA bug where the flipped/rotated variants were incorrectly generated from the original image, and ensure deterministic, sorted test filenames so the output is stable. Finally, I ensure the script always writes a properly formatted submission CSV with `image_id,label` columns.'
- What this solution (achieved 0.06764) has done: 'Your current 0.321 score is consistent with a model whose classifier head is random (ImageNet pretrained backbone + brand-new 5-class head), so the smallest meaningful fix is to load cassava-trained weights if they exist anywhere in `/kaggle/input` (your code only checks one hardcoded, likely-misspelled directory). To keep core logic identical, I only (1) add a robust search for `.pth/.pt/.bin` weights across the competition dataset folders and (2) make weight loading handle common checkpoint formats (`state_dict`, `model`, `module.` prefixes) so the same architecture can actually use the weights. If no weights are found, it still fall back to the current behavior (ImageNet backbone) and write a valid `submission.csv`. This should move accuracy sharply upward toward your ~0.892 target if any reasonable cassava checkpoint is present.'
- What this solution (achieved 0.1009) has done: 'Your current score (0.06764) is far below the target (0.8926), which is consistent with running a cassava architecture but effectively predicting with an untrained/random 5-class head (or not actually loading a compatible cassava checkpoint). The smallest change that should move accuracy sharply upward (without changing the model/training logic) is to (1) pick the *most likely compatible* checkpoint file rather than iterating over every `.pth/.pt/.bin`, and (2) make checkpoint loading more robust by filtering to matching tensor shapes (so a cassava-trained head loads while unrelated weights are ignored). If no compatible checkpoint exists in `/kaggle/input`, the behavior stays the same (ImageNet backbone + random head), but if one exists anywhere under `/kaggle/input`, this should substantially increase score toward your target. I also ensure the test output is strictly aligned to `sample_submission.csv` ordering to avoid any accidental row-order mismatches that can destroy accuracy.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd



## === cell 1
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import cv2




## === cell 2
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=5, use_pretrained_backbone=True):
        super().__init__()

        self.model = timm.create_model(
            "seresnext50_32x4d",
            pretrained=use_pretrained_backbone,
            num_classes=0,  # use as feature extractor
            global_pool="",  # we'll pool ourselves
        )

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, inputs):
        x = inputs / 255.0
        bs = x.size(0)
        x = self.model.forward_features(x)
        fm = self._avg_pooling(x).view(bs, -1)
        fm = self.dropout(fm)
        x = self._fc(fm)
        return x




## === cell 4
class DatasetTest:
    def __init__(self, test_data_dir):
        self.root_dir = test_data_dir
        self.ds = self.get_list(test_data_dir)

    def get_list(self, dir_path):
        pic_list = [f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")]
        pic_list.sort()
        return pic_list

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))

        image_90 = np.rot90(image, 1)
        image_180 = np.rot90(image, 2)
        image_270 = np.rot90(image, 3)

        image_fliplr = np.fliplr(image)
        image_fliplr_90 = np.rot90(image_fliplr, 1)
        image_fliplr_180 = np.rot90(image_fliplr, 2)
        image_fliplr_270 = np.rot90(image_fliplr, 3)

        image_batch = np.stack(
            [
                image,
                image_90,
                image_180,
                image_270,
                image_fliplr,
                image_fliplr_90,
                image_fliplr_180,
                image_fliplr_270,
            ],
            axis=0,
        )

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]
        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        image, float_image = self.preprocess_func(image)
        return fname, image, float_image




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/test_images"
)
if not os.path.isdir(test_datadir):
    test_datadir = os.path.join(kaggle_root, "test_images")
if not os.path.isdir(test_datadir):
    raise FileNotFoundError(f"Could not find test_images directory under {kaggle_root}")

dataiter = DatasetTest(test_datadir)

sample_path = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_path):
    sample_path = os.path.join(kaggle_root, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError("Could not find sample_submission.csv under /kaggle/input")

sample_sub = pd.read_csv(sample_path)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)


def find_weights_under(root_dir):
    exts = (".pth", ".pt", ".bin")
    found = []
    if not os.path.isdir(root_dir):
        return found
    for base, dirs, files in os.walk(root_dir):
        bn = os.path.basename(base)
        if bn in ("train_images", "test_images", "train_tfrecords", "test_tfrecords"):
            dirs[:] = []
            continue
        for fn in files:
            lf = fn.lower()
            if lf.endswith(exts):
                found.append(os.path.join(base, fn))
    found.sort()
    return found


model_dir = os.path.join(
    kaggle_root, "cassva-models-se50-640"
)  # original (possibly misspelled) path
weights = []
if os.path.isdir(model_dir):
    weights = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    weights = [
        w
        for w in weights
        if os.path.isfile(w)
        and (
            w.lower().endswith(".pth")
            or w.lower().endswith(".pt")
            or w.lower().endswith(".bin")
        )
    ]
    weights.sort()

if len(weights) == 0:
    weights = find_weights_under(kaggle_root)

print(f"Found {len(weights)} weight file(s).")
if len(weights) > 0:
    print("First few weights:", weights[:5])




## === cell 6
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj  # may already be a state_dict


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _filter_state_dict_by_shape(model, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict, []
    model_sd = model.state_dict()
    kept = {}
    dropped = []
    for k, v in state_dict.items():
        if (
            k in model_sd
            and hasattr(v, "shape")
            and hasattr(model_sd[k], "shape")
            and tuple(v.shape) == tuple(model_sd[k].shape)
        ):
            kept[k] = v
        else:
            dropped.append(k)
    return kept, dropped


def load_checkpoint_flexible(model, weight_path, device):
    obj = torch.load(weight_path, map_location=device)
    sd = _extract_state_dict(obj)
    if isinstance(sd, dict):
        for pref in ("module.", "model.", "net."):
            sd = _strip_prefix_if_present(sd, pref)

        sd, dropped = _filter_state_dict_by_shape(model, sd)
    else:
        dropped = []

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected, dropped


def score_weight_compatibility(model, weight_path, device):
    try:
        obj = torch.load(weight_path, map_location=device)
        sd = _extract_state_dict(obj)
        if isinstance(sd, dict):
            for pref in ("module.", "model.", "net."):
                sd = _strip_prefix_if_present(sd, pref)
            filt, _ = _filter_state_dict_by_shape(model, sd)
            return len(filt)
    except Exception:
        return -1
    return -1


def predict_with_model(model, weights, dataset, device, sample_sub):
    chosen_weight = None
    if len(weights) > 0:
        scored = []
        for w in weights:
            scored.append((score_weight_compatibility(model, w, device), w))
        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_w = scored[0]
        if best_score > 0:
            chosen_weight = best_w
            print(
                f"Chose checkpoint with best compatibility score={best_score}: {chosen_weight}"
            )
        else:
            print(
                "No compatible checkpoint found among discovered files; will run with ImageNet backbone + random head."
            )
    else:
        print("No weights found; will run with ImageNet backbone + random head.")

    if chosen_weight is not None:
        missing, unexpected, dropped = load_checkpoint_flexible(
            model, chosen_weight, device
        )
        print(f"Loaded weights: {chosen_weight}")
        if len(dropped) > 0:
            print(f"  Dropped incompatible keys (showing up to 10): {dropped[:10]}")
        if len(missing) > 0:
            print(f"  Missing keys (showing up to 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"  Unexpected keys (showing up to 10): {unexpected[:10]}")
        model.eval()
    else:
        model.eval()

    image_ids = []
    predictions = []

    for i in range(len(dataset)):
        fname, _, float_image = dataset[i]

        inp = torch.from_numpy(float_image).to(device).float()
        with torch.no_grad():
            out = model(inp)
            out = torch.softmax(out, dim=-1).detach().cpu().numpy()
            out = np.mean(out, axis=0)

        image_ids.append(fname)
        predictions.append(int(np.argmax(out)))

    cur_result = pd.DataFrame({"image_id": image_ids, "label": predictions})

    cur_result["image_id"] = cur_result["image_id"].astype(str)
    cur_result = sample_sub[["image_id"]].merge(cur_result, on="image_id", how="left")

    if cur_result["label"].isna().any():
        mode_label = (
            int(pd.Series(predictions).mode().iloc[0]) if len(predictions) else 0
        )
        cur_result["label"] = cur_result["label"].fillna(mode_label).astype(int)
    else:
        cur_result["label"] = cur_result["label"].astype(int)

    if chosen_weight is not None:
        cur_result.to_csv(f"{os.path.basename(chosen_weight)}.csv", index=False)

    cur_result.to_csv("submission.csv", index=False)
    return cur_result




## === cell 7
model = Net(num_classes=5, use_pretrained_backbone=True).to(device)

_ = predict_with_model(model, weights, dataiter, device, sample_sub)

sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, 4).all()
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
