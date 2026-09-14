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

albumentations==2.0.8
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

0.8942278634028408

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the crash by ensuring the test dataset only contains valid image files and by raising a clear error if `cv2.imread` returns `None` (which happens when the directory listing includes subfolders or non-image entries). I also make the augmentation compatible with albumentations v2 by removing deprecated `always_apply` usage while keeping the exact same transforms and behavior. Finally, I make the inference deterministic-ish and faster/safer by disabling gradient tracking globally during prediction and by ensuring the submission strictly follows `sample_submission.csv` ordering and columns, writing `submission.csv` to the working directory.'
- What this solution (achieved 0.0994) has done: 'Your current score (0.61099) is far below the target (0.8942), and the main reason is that you are running `pretrained=False` and (likely) with no usable weights found, so predictions are effectively random-ish. The smallest change that preserves your exact model/training-free inference logic is to (1) enable ImageNet pretrained weights for the same backbone and (2) make the test-time augmentation deterministic and evaluation-appropriate by removing *random* flips/crops/color jitter from the **test** pipeline (keeping the same multi-view averaging but using a fixed resize/center crop). These two changes typically jump accuracy substantially in Cassava without changing architecture or adding training. I also make weight loading a bit more robust (handles `module.` prefixes) but otherwise keep your inference semantics the same and still write `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your code fails because it hard-errors when no finetuned `.pth/.pt` weights are found under `/kaggle/input`, which is true in this environment. To make the notebook run end-to-end and produce a valid `submission.csv`, I keep the same model and inference logic but add a safe fallback: run inference with the ImageNet-pretrained backbone and a deterministic prior-based classifier head initialized from the training label distribution (so it’s better than random without requiring external weights). I also ensure the test file listing and submission ordering strictly follow `sample_submission.csv` as you already intended. This should yield a non-trivial accuracy (not near target without finetuning, but meaningfully higher than random) while preserving your pipeline and producing a valid CSV.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
import sys

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "":
    pass



## === cell 2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import cv2
import timm
import albumentations as A




## === cell 3
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )




## === cell 4
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = timm.create_model("seresnext50_32x4d", pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, inputs):
        input_iid = inputs
        input_iid = input_iid / 255.0
        bs = input_iid.size(0)
        x = self.model.forward_features(input_iid)
        fm = self._avg_pooling(x)
        fm = fm.view(bs, -1)
        feature = self.dropout(fm)
        x = self._fc(feature)
        return x




## === cell 5
class DatasetTest:
    def __init__(self, test_data_dir):
        self.ds = self.get_list(test_data_dir)
        self.root_dir = test_data_dir

        self.val_trans = A.Compose(
            [
                A.LongestMaxSize(max_size=640, p=1.0),
                A.PadIfNeeded(
                    min_height=640,
                    min_width=640,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                A.CenterCrop(height=640, width=640, p=1.0),
            ]
        )

    def get_list(self, dir):
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        out = []
        for name in os.listdir(dir):
            full = os.path.join(dir, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_ext:
                out.append(name)
        out = sorted(out)
        return out

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.val_trans(image=image)["image"]
        image2 = self.val_trans(image=image)["image"]
        image3 = self.val_trans(image=image)["image"]
        image4 = self.val_trans(image=image)["image"]
        image5 = self.val_trans(image=image)["image"]
        image6 = self.val_trans(image=image)["image"]
        image7 = self.val_trans(image=image)["image"]
        image8 = self.val_trans(image=image)["image"]

        image_batch = np.stack(
            [image1, image2, image3, image4, image5, image6, image7, image8]
        )
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]
        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        if image is None:
            raise FileNotFoundError(
                f"Failed to read image with cv2.imread: {image_path}"
            )
        image, float_image = self.preprocess_func(image)
        return fname, image, float_image




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/test_images"
)
if not os.path.isdir(test_datadir):
    test_datadir = os.path.join(
        kaggle_root,
        "cassava-leaf-disease-classification",
        "cassava-leaf-disease-classification",
        "test_images",
    )
assert os.path.isdir(test_datadir), f"Test image directory not found: {test_datadir}"

dataiter = DatasetTest(test_datadir)


def _find_weight_files(kaggle_root="/kaggle/input"):
    """
    Search common locations for .pth/.pt weights.
    """
    candidates = [
        os.path.join(kaggle_root, "cassva-models-se50-640"),
        os.path.join(kaggle_root, "cassava-models-se50-640"),
        os.path.join(kaggle_root, "cassava-leaf-disease-classification"),
    ]
    weight_files = []
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                lf = f.lower()
                if lf.endswith(".pth") or lf.endswith(".pt") or lf.endswith(".bin"):
                    weight_files.append(os.path.join(d, f))
    weight_files = sorted(weight_files)
    return weight_files


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    return ckpt


def _has_fc_weights(sd):
    if not isinstance(sd, dict):
        return False
    keys = set(sd.keys())
    return any(
        k.endswith("_fc.weight")
        or k.endswith("_fc.bias")
        or k.endswith("._fc.weight")
        or k.endswith("._fc.bias")
        for k in keys
    )


def _init_fc_from_train_prior(model, kaggle_root="/kaggle/input"):
    """
    Bugfix / robustness: when no finetuned weights exist, prevent a crash and make
    predictions non-random by initializing the classifier bias to the log prior
    of training labels. This preserves the same architecture and inference flow.
    """
    train_path = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/train.csv"
    )
    if not os.path.isfile(train_path):
        train_path = os.path.join(kaggle_root, "train.csv")
    if not os.path.isfile(train_path):
        prior = np.ones(5, dtype=np.float64) / 5.0
    else:
        tr = pd.read_csv(train_path)
        vc = tr["label"].value_counts().sort_index()
        prior = np.zeros(5, dtype=np.float64)
        for i in range(5):
            prior[i] = float(vc.get(i, 0))
        prior = prior / max(prior.sum(), 1.0)

    prior = np.clip(prior, 1e-6, 1.0)
    log_prior = np.log(prior)

    with torch.no_grad():
        model._fc.weight.zero_()
        model._fc.bias.copy_(
            torch.tensor(
                log_prior, dtype=model._fc.bias.dtype, device=model._fc.bias.device
            )
        )


weights = _find_weight_files(kaggle_root)
print(f"Found {len(weights)} weight file(s).")


def predict_with_model(model, weights):
    merge_res_dict = {}

    sample_path = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/sample_submission.csv"
    )
    if not os.path.isfile(sample_path):
        sample_path = os.path.join(kaggle_root, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    if len(weights) == 0:
        print(
            "WARNING: No finetuned model weights found under /kaggle/input. "
            "Falling back to training-label prior initialization for the classifier head."
        )
        _init_fc_from_train_prior(model, kaggle_root=kaggle_root)
        valid_weight_paths = [None]
    else:
        valid_weight_paths = []
        for w in weights:
            try:
                ckpt = torch.load(w, map_location="cpu")
                sd = _extract_state_dict(ckpt)
                sd = _strip_module_prefix(sd)
                if _has_fc_weights(sd):
                    valid_weight_paths.append(w)
            except Exception as e:
                print(f"Skipping unreadable checkpoint {w}: {repr(e)}")

        if len(valid_weight_paths) == 0:
            print(
                "WARNING: Weight files were found but none contained '_fc' classifier weights compatible with this model. "
                "Falling back to training-label prior initialization for the classifier head."
            )
            _init_fc_from_train_prior(model, kaggle_root=kaggle_root)
            valid_weight_paths = [None]
        else:
            print(
                f"Using {len(valid_weight_paths)} checkpoint(s) that contain classifier head weights."
            )

    for j, weight in enumerate(valid_weight_paths):
        if weight is not None:
            ckpt = torch.load(weight, map_location=device)
            state = _extract_state_dict(ckpt)
            state = _strip_module_prefix(state)
            model.load_state_dict(state, strict=False)

        model.eval()

        len_data = len(dataiter)
        for i in range(len_data):
            if (i + 1) % 200 == 0 or i == 0 or (i + 1) == len_data:
                print(f"weight {j+1}/{len(valid_weight_paths)}: data {i+1}/{len_data}")
            fname, _, float_image = dataiter.__getitem__(i)

            input_tensor = torch.from_numpy(float_image).to(device).float()
            with torch.inference_mode():
                output = model(input_tensor)
                output = torch.nn.functional.softmax(output, dim=-1).cpu().numpy()
                output = np.mean(output, axis=0)

            if fname not in merge_res_dict:
                merge_res_dict[fname] = output
            else:
                merge_res_dict[fname] += output

    preds = []
    for fname in sample["image_id"].tolist():
        if fname in merge_res_dict:
            preds.append(int(np.argmax(merge_res_dict[fname])))
        else:
            preds.append(0)

    sub = pd.DataFrame({"image_id": sample["image_id"].tolist(), "label": preds})
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())


model = Net().to(device)
predict_with_model(model, weights)
