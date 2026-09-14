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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.8845572680568148

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.25673) has done: 'I make the pipeline run end-to-end without requiring an external weights dataset by falling back to a torchvision pretrained ViT when the requested `.pth` file isn’t available, keeping the same ViT family and inference flow. I also fix the device/dtype mismatch that caused `Input type ... and weight type ... should be the same` by ensuring the model is moved to `device` after any potential weight loading and that inputs use the same device. Finally, I keep the submission formatting aligned to `sample_submission.csv` and always write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.25673) is far below the target (0.88456), and the biggest issue is that your fallback path uses an ImageNet-pretrained ViT with a freshly initialized 5-class head, so predictions are essentially near-random. To move the score sharply toward the target while keeping the same core ViT inference pipeline, the minimal fix is to *require* a valid cassava fine-tuned `.pth/.pt` and load it correctly (including common checkpoint key-prefix patterns like `module.`). I also switch test-time preprocessing to the official ViT_H_14 weights’ transforms when using torchvision weights (kept as a last-resort fallback), and I add a lightweight batch inference (no change to semantics) to reduce overhead and stay within runtime. The output submission format/path stays identical and still always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def torch_load_state_dict(path: str, map_location):
    """
    Compatibility: environments differ on torch.load(weights_only=...).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(state):
    """
    Score-improvement fix: handle common checkpoint wrappers & key prefixes so we actually
    load the trained weights into the ViT (otherwise accuracy collapses).
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state_dict = state["model"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_sd[nk] = v
        state_dict = new_sd

    return state_dict


def state_dict_head_out_features(sd: dict) -> int | None:
    """
    Score-improvement fix: prefer checkpoints that clearly have a 5-class head
    (to avoid picking generic ImageNet weights or incompatible fine-tunes).
    """
    for k in ("heads.head.weight", "head.weight", "fc.weight", "classifier.weight"):
        if k in sd and hasattr(sd[k], "shape") and len(sd[k].shape) == 2:
            return int(sd[k].shape[0])
    return None


def find_best_cassava_weights_path(preferred_path: str) -> str | None:
    """
    Score-improvement fix (minimal, preserves ViT inference flow):
    - Do not silently fall back to ImageNet+random head (near-random predictions).
    - Search /kaggle/input for likely cassava fine-tuned checkpoints, including .ckpt.
    - Prefer checkpoints that include a 5-class classification head.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = []
    for pattern in [
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.ckpt",
    ]:
        candidates.extend(glob.glob(pattern, recursive=True))

    candidates = sorted(set(candidates))

    def quick_text_score(p: str) -> int:
        base = os.path.basename(p).lower()
        full = p.lower()
        s = 0
        if "cassava" in full:
            s += 50
        if "leaf" in full:
            s += 10
        if "vit" in full:
            s += 10
        if "efficientnet" in full or "resnet" in full:
            s -= 5
        if "imagenet" in full:
            s -= 20
        if "swa" in full:
            s += 2
        if "best" in base:
            s += 2
        return s

    prefiltered = sorted(
        candidates, key=lambda p: (quick_text_score(p), p), reverse=True
    )[:50]

    best = None
    best_score = -(10**9)

    for p in prefiltered:
        try:
            state = torch_load_state_dict(p, map_location="cpu")
            sd = unwrap_state_dict(state)
            if not isinstance(sd, dict):
                continue
            out_features = state_dict_head_out_features(sd)
            s = quick_text_score(p)
            if out_features == num_classes:
                s += 1000
            elif out_features is None:
                s -= 50
            else:
                s -= 200
            if s > best_score:
                best_score = s
                best = p
        except Exception:
            continue

    return best


weights_path = find_best_cassava_weights_path(model_path)
if weights_path is None:
    raise FileNotFoundError(
        "No cassava fine-tuned checkpoint (.pth/.pt/.ckpt) was found under /kaggle/input.\n"
        "Your current score indicates the model is effectively untrained for cassava (random head).\n"
        "Please add the correct trained weights dataset and set model_path accordingly."
    )
else:
    print(f"Using weights: {weights_path}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/21615044.py in <cell line: 0>()
    153 if weights_path is None:
    154     # Score-improvement fix: fail fast instead of silently producing a near-random submission.
--> 155     raise FileNotFoundError(
    156         "No cassava fine-tuned checkpoint (.pth/.pt/.ckpt) was found under /kaggle/input.\n"
    157         "Your current score indicates the model is effectively untrained for cassava (random head).\n"

FileNotFoundError: No cassava fine-tuned checkpoint (.pth/.pt/.ckpt) was found under /kaggle/input.
Your current score indicates the model is effectively untrained for cassava (random head).
Please add the correct trained weights dataset and set model_path accordingly.

## === cell 1
val_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = models.vit_h_14(weights=None, image_size=image_size)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

state = torch_load_state_dict(weights_path, map_location="cpu")
state_dict = unwrap_state_dict(state)

model_sd = model.state_dict()
filtered_sd = {}
for k, v in state_dict.items():
    if (
        k in model_sd
        and hasattr(v, "shape")
        and hasattr(model_sd[k], "shape")
        and tuple(v.shape) == tuple(model_sd[k].shape)
    ):
        filtered_sd[k] = v

missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
if missing or unexpected:
    print(
        f"Warning: non-strict load_state_dict after shape-filter. "
        f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )

model = model.to(device)
model.eval()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    850     try:
--> 851         f.seek(f.tell())
    852         return True

AttributeError: 'NoneType' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2209291027.py in <cell line: 0>()
     13 model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
     14 
---> 15 state = torch_load_state_dict(weights_path, map_location="cpu")
     16 state_dict = unwrap_state_dict(state)
     17 

/tmp/ipykernel_55/21615044.py in torch_load_state_dict(path, map_location)
     28     """
     29     try:
---> 30         return torch.load(path, map_location=map_location, weights_only=True)
     31     except TypeError:
     32         return torch.load(path, map_location=map_location)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    754             return _open_buffer_writer(name_or_buffer)
    755         elif "r" in mode:
--> 756             return _open_buffer_reader(name_or_buffer)
    757         else:
    758             raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, buffer)
    739     def __init__(self, buffer):
    740         super().__init__(buffer)
--> 741         _check_seekable(buffer)
    742 
    743 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    852         return True
    853     except (io.UnsupportedOperation, AttributeError) as e:
--> 854         raise_err_msg(["seek", "tell"], e)
    855     return False
    856 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in raise_err_msg(patterns, e)
    845                     + " try to load from it instead."
    846                 )
--> 847                 raise type(e)(msg)
    848         raise e
    849 

AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

## === cell 2
sample_sub = pd.read_csv(sample_submission_path)
image_ids = sample_sub["image_id"].tolist()

batch_size = 16 if device.type == "cuda" else 8

test_predictions = []
batch_imgs = []

with torch.no_grad():
    for image_name in tqdm(image_ids, desc="Test"):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_t = val_transforms(img)
        batch_imgs.append(img_t)

        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            logits = model(x)
            preds = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            test_predictions.extend(preds)
            batch_imgs = []

    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0).to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        test_predictions.extend(preds)

if len(test_predictions) != len(image_ids):
    raise RuntimeError(
        f"Prediction length mismatch: {len(test_predictions)} vs {len(image_ids)}"
    )

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_df = submission_df[["image_id", "label"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/386620306.py in <cell line: 0>()
     19         if len(batch_imgs) == batch_size:
     20             x = torch.stack(batch_imgs, dim=0).to(device)
---> 21             logits = model(x)
     22             preds = (
     23                 torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()

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
    275 
    276         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
--> 277         x = self.conv_proj(x)
    278         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    279         x = x.reshape(n, self.hidden_dim, n_h * n_w)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
