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

0.8159564823209429

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the manual wheel installs and instead using only libraries that are already available in Kaggle (notably `timm`). I keep the ensemble logic the same (two models averaged) by loading both models via `timm` and then loading the provided `.pth` weights with `map_location='cpu'` for robustness before moving to GPU. I also fix the softmax/argmax call (`dim=1` is required) and add safe handling for CPU-only environments so the notebook always completes. Finally, I ensure the submission is written as `submission.csv` with exactly `image_id,label` and the correct test set order (sorted by filename, matching sample submission ordering expectations).'
- What this solution (achieved 0.1278) has done: 'I fix the immediate runtime failure by removing the hard dependency on missing `../input/ensemblev3/*.pth` files and instead loading robust pretrained `timm` weights (same two-model averaging ensemble logic) so the notebook runs end-to-end. I also correct the input preprocessing to match what these `timm` models expect (RGB + ImageNet normalization at the correct resolution) to move accuracy up substantially toward the target. Finally, I keep the exact submission schema/order by merging with `sample_submission.csv` and always writing a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current low score is mainly because you’re using ImageNet-pretrained heads (random for 5 classes) and applying CLAHE+single-view inference, so predictions are essentially untrained for cassava labels. To move the accuracy up toward the target with minimal core-logic change, I keep the same two-model averaging ensemble and same preprocessing, but load actual Cassava fine-tuned weights if they exist in common Kaggle input locations; otherwise we fall back to your current behavior. I also switch inference to use `timm`’s `resolve_data_config`/`create_transform` per model so normalization and resizing exactly match the pretrained checkpoints (this is a small, metric-aligned change that typically yields a large jump). Finally, I ensure correct test order by strictly following `sample_submission.csv` image_id ordering, not relying on filesystem sorting.'

# 9. Code solution

## === cell 0
import os
import glob
import cv2
import torch
import timm
import pandas as pd
import tqdm

torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("DEVICE:", DEVICE)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print(
    "TEST_IMG_DIR exists:",
    os.path.exists(TEST_IMG_DIR),
    "num_files:",
    len(glob.glob(os.path.join(TEST_IMG_DIR, "*"))),
)




## === cell 1
def load_state_dict_flexible(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(unexpected) > 0:
        print(
            f"Warning: unexpected keys while loading {os.path.basename(path)} (showing up to 5): {unexpected[:5]}"
        )
    if len(missing) > 0:
        print(
            f"Warning: missing keys while loading {os.path.basename(path)} (showing up to 5): {missing[:5]}"
        )
    return model


def find_weight_file(preferred_substrings):
    candidates = []
    search_roots = ["../input", DATA_DIR]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pth"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))

    def score(p):
        lp = p.lower()
        s = 0
        for i, sub in enumerate(preferred_substrings):
            if sub in lp:
                s += 10 - i
        s -= lp.count(os.sep) * 0.01
        return s

    scored = [(score(p), p) for p in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    for sc, p in scored:
        lp = p.lower()
        if any(sub in lp for sub in preferred_substrings):
            return p
    return None


efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
hrnet = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)

eff_w = find_weight_file(["cassava", "b5", "efficientnet", "tf_efficientnet_b5"])
hr_w = find_weight_file(
    ["cassava", "seresnext101", "se_resnext101", "32x4d", "resnext101"]
)

if eff_w is not None:
    print("Found efficientnet weights:", eff_w)
    efficient = load_state_dict_flexible(efficient, eff_w)
else:
    print(
        "No cassava efficientnet weights found; using ImageNet pretrained backbone + random 5-class head."
    )

if hr_w is not None:
    print("Found seresnext101 weights:", hr_w)
    hrnet = load_state_dict_flexible(hrnet, hr_w)
else:
    print(
        "No cassava seresnext101 weights found; using ImageNet pretrained backbone + random 5-class head."
    )

efficient.to(DEVICE).eval()
hrnet.to(DEVICE).eval()

print("Models have been loaded...\n")
print("efficient:", efficient.__class__.__name__)
print("hrnet:", hrnet.__class__.__name__)



## === cell 2
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np

eff_cfg = resolve_data_config(efficient.default_cfg, model=efficient)
hr_cfg = resolve_data_config(hrnet.default_cfg, model=hrnet)

eff_tf = create_transform(**eff_cfg, is_training=False)
hr_tf = create_transform(**hr_cfg, is_training=False)

print("Efficient cfg:", eff_cfg)
print("HR cfg:", hr_cfg)


def _clahe_bgr(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def to_tensor_via_timm_transform(image_bgr, transform):
    img = _clahe_bgr(image_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_like = img  # create_transform accepts numpy HWC uint8
    x = transform(pil_like)  # CHW float tensor normalized as required
    x = x.unsqueeze(0).to(DEVICE)
    return x




## === cell 3
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if not os.path.exists(sample_path):
    raise RuntimeError(f"Missing sample_submission.csv at {sample_path}")
sample = pd.read_csv(sample_path)
test_image_ids = sample["image_id"].tolist()

names, labels = [], []

with torch.no_grad():
    for image_id in tqdm.tqdm(test_image_ids, total=len(test_image_ids)):
        file = os.path.join(TEST_IMG_DIR, image_id)
        img = cv2.imread(file)
        if img is None:
            names.append(image_id)
            labels.append(0)
            continue

        x_hr = to_tensor_via_timm_transform(img, hr_tf)
        x_eff = to_tensor_via_timm_transform(img, eff_tf)

        hr_out = hrnet(x_hr)
        eff_out = efficient(x_eff)

        total = (hr_out + eff_out) / 2.0
        pred = int(torch.argmax(total, dim=1).item())

        names.append(image_id)
        labels.append(pred)

print(
    "Predictions made for:", len(names), "images (expected:", len(test_image_ids), ")"
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3936272142.py in <cell line: 0>()
     18             continue
     19 
---> 20         x_hr = to_tensor_via_timm_transform(img, hr_tf)
     21         x_eff = to_tensor_via_timm_transform(img, eff_tf)
     22 

/tmp/ipykernel_55/462233306.py in to_tensor_via_timm_transform(image_bgr, transform)
     29     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     30     pil_like = img  # create_transform accepts numpy HWC uint8
---> 31     x = transform(pil_like)  # CHW float tensor normalized as required
     32     x = x.unsqueeze(0).to(DEVICE)
     33     return x

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in __call__(self, img)
     93     def __call__(self, img):
     94         for t in self.transforms:
---> 95             img = t(img)
     96         return img
     97 

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

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in forward(self, img)
    352             PIL Image or Tensor: Rescaled image.
    353         """
--> 354         return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
    355 
    356     def __repr__(self) -> str:

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in resize(img, size, interpolation, max_size, antialias)
    463             )
    464 
--> 465     _, image_height, image_width = get_dimensions(img)
    466     if isinstance(size, int):
    467         size = [size]

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in get_dimensions(img)
     78         return F_t.get_dimensions(img)
     79 
---> 80     return F_pil.get_dimensions(img)
     81 
     82 

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/_functional_pil.py in get_dimensions(img)
     29         width, height = img.size
     30         return [channels, height, width]
---> 31     raise TypeError(f"Unexpected type {type(img)}")
     32 
     33 

TypeError: Unexpected type <class 'numpy.ndarray'>

## === cell 4
df = pd.DataFrame({"image_id": names, "label": labels}, columns=["image_id", "label"])

df = sample[["image_id"]].merge(df, on="image_id", how="left")
df["label"] = df["label"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape}")
print(df.head())
print("Label value counts:\n", df["label"].value_counts(dropna=False).sort_index())
