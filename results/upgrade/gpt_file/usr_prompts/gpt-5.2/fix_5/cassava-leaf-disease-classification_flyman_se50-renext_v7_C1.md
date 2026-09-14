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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the crash by ensuring the test dataset only contains valid image files and by raising a clear error if `cv2.imread` returns `None` (which happens when the directory listing includes subfolders or non-image entries). I also make the augmentation compatible with albumentations v2 by removing deprecated `always_apply` usage while keeping the exact same transforms and behavior. Finally, I make the inference deterministic-ish and faster/safer by disabling gradient tracking globally during prediction and by ensuring the submission strictly follows `sample_submission.csv` ordering and columns, writing `submission.csv` to the working directory.'
- What this solution (achieved 0.0994) has done: 'Your current score (0.61099) is far below the target (0.8942), and the main reason is that you are running `pretrained=False` and (likely) with no usable weights found, so predictions are effectively random-ish. The smallest change that preserves your exact model/training-free inference logic is to (1) enable ImageNet pretrained weights for the same backbone and (2) make the test-time augmentation deterministic and evaluation-appropriate by removing *random* flips/crops/color jitter from the **test** pipeline (keeping the same multi-view averaging but using a fixed resize/center crop). These two changes typically jump accuracy substantially in Cassava without changing architecture or adding training. I also make weight loading a bit more robust (handles `module.` prefixes) but otherwise keep your inference semantics the same and still write `submission.csv` aligned to `sample_submission.csv`.'

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


weights = _find_weight_files(kaggle_root)
print(f"Found {len(weights)} weight file(s).")


def predict_with_model(model, weights):
    merge_res_dict = {}
    image_ids_all = []

    if len(weights) == 0:
        raise RuntimeError(
            "No model weights found under /kaggle/input. "
            "With an untrained classifier head, accuracy will be near random. "
            "Please add finetuned Cassava weights as a Kaggle dataset and re-run."
        )

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
        raise RuntimeError(
            "Weight files were found but none contained '_fc' classifier weights compatible with this model. "
            "Using such checkpoints would keep predictions near-random; please provide correct finetuned weights."
        )

    print(
        f"Using {len(valid_weight_paths)} checkpoint(s) that contain classifier head weights."
    )

    for j, weight in enumerate(valid_weight_paths):
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
                if j == 0:
                    image_ids_all.append(fname)
            else:
                merge_res_dict[fname] += output

    sample_path = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/sample_submission.csv"
    )
    if not os.path.isfile(sample_path):
        sample_path = os.path.join(kaggle_root, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2225726733.py in <cell line: 0>()
    161 
    162 model = Net().to(device)
--> 163 predict_with_model(model, weights)

/tmp/ipykernel_55/2225726733.py in predict_with_model(model, weights)
     84     # an almost-random submission far from the target score.
     85     if len(weights) == 0:
---> 86         raise RuntimeError(
     87             "No model weights found under /kaggle/input. "
     88             "With an untrained classifier head, accuracy will be near random. "

RuntimeError: No model weights found under /kaggle/input. With an untrained classifier head, accuracy will be near random. Please add finetuned Cassava weights as a Kaggle dataset and re-run.
