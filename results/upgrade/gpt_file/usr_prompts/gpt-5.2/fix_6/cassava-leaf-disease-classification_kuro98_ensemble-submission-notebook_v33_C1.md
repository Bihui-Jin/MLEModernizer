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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8937745542459957

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it can run end-to-end in the Kaggle environment by (1) making model loading robust to missing `/kaggle/input/vit-*` assets and (2) ensuring images are read as RGB to avoid transform/model shape errors. I also fix submission validity by iterating over the test set in a deterministic order (`shuffle=False`) and then aligning predictions to `sample_submission.csv` so the output has exactly the required 2676 rows in the correct `image_id` order. These changes preserve the existing inference/ensemble logic when the provided models are available, and otherwise fall back to a standard torchvision ViT-based baseline so a valid submission is always produced. Finally, I ensure the submission file is written with the required `.csv` suffix and correct columns.'
- What this solution (achieved 0.05531) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT is created with an `image_size` that matches the pipeline’s resized inputs (384), because the current fallback expects 224 and asserts on forward. I keep your existing dataset/transforms, ensemble weighting, and submission alignment logic unchanged. I also make the model-loading helper robust to cases where a loaded model is a checkpoint dict (common in Kaggle assets) so it still runs end-to-end. These changes should both unblock execution and substantially improve accuracy vs the current broken/fallback mismatch that led to an extremely low score.'
- What this solution (achieved 0.15583) has done: 'I fix the `IsADirectoryError` by ensuring the dataset only indexes actual image files (and not the nested `test_images/` directory that exists inside the folder). To keep submission ordering stable and avoid any silent row mismatches, I also build the test file list directly from `sample_submission.csv` (falling back to directory listing if needed) and filter missing entries. These changes are score-neutral (they don’t change the model/inference math) but unblock end-to-end execution so a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _safe_torch_load(path: str, map_location):
    obj = torch.load(path, map_location=map_location)
    return obj


def _resize_vit_pos_embedding_(
    model: torch.nn.Module, image_size: int, patch_size: int = 16
):
    """
    Score-impacting fix (minimal core-logic change): when we use ImageNet-pretrained ViT-B/16
    at 384px, the positional embedding length must match the new number of patches.
    We resize the grid part of pos_embed with bicubic interpolation, keeping cls token intact.
    This keeps the same ViT architecture and preserves evaluation semantics (inference-only).
    """
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "pos_embedding"):
        return model

    pos = model.encoder.pos_embedding  # [1, 1+N, D]
    if not isinstance(pos, torch.nn.Parameter):
        return model

    with torch.no_grad():
        pos_data = pos.data
        if pos_data.ndim != 3 or pos_data.shape[0] != 1:
            return model

        n_tokens, dim = pos_data.shape[1], pos_data.shape[2]
        if n_tokens < 2:
            return model

        cls_pos = pos_data[:, :1, :]  # [1,1,D]
        grid_pos = pos_data[:, 1:, :]  # [1,N,D]

        old_n = grid_pos.shape[1]
        old_g = int(old_n**0.5)
        if old_g * old_g != old_n:
            return model

        new_g = image_size // patch_size
        if new_g * new_g == old_n:
            return model

        grid_pos = grid_pos.reshape(1, old_g, old_g, dim).permute(
            0, 3, 1, 2
        )  # [1,D,H,W]
        grid_pos = torch.nn.functional.interpolate(
            grid_pos, size=(new_g, new_g), mode="bicubic", align_corners=False
        )
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(
            1, new_g * new_g, dim
        )  # [1,N',D]

        new_pos = torch.cat([cls_pos, grid_pos], dim=1)
        model.encoder.pos_embedding = torch.nn.Parameter(new_pos)

    return model


def _build_fallback_vit(num_classes: int, image_size: int):
    """
    Score-moving change: previous fallback used random weights (weights=None), yielding near-random accuracy.
    We instead use ImageNet-pretrained ViT-B/16 weights and resize positional embeddings to support 384 inputs.
    Core logic preserved: still ViT-B/16 with a linear classification head to 5 classes.
    """
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    m = vit_b_16(
        weights=weights
    )  # default expects 224, but we will adapt pos-embeds for 384
    m = _resize_vit_pos_embedding_(m, image_size=image_size, patch_size=16)

    if hasattr(m, "heads") and hasattr(m.heads, "head"):
        in_features = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = m.classifier.in_features
        m.classifier = torch.nn.Linear(in_features, num_classes)
    return m


def _unwrap_loaded_model(obj):
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], torch.nn.Module):
            return obj["model"]
    return None


def _maybe_load_models(device):
    path_a = "/kaggle/input/vit-v1/vit_v1.pt"
    path_b = "/kaggle/input/vit-boosted/vit_boosted.pt"
    path_head = "/kaggle/input/linear-head/linear_cls.pt"

    model_a = None
    model_b = None
    linear_head = None

    try:
        if os.path.exists(path_a):
            model_a = _unwrap_loaded_model(_safe_torch_load(path_a, map_location="cpu"))
        if os.path.exists(path_b):
            model_b = _unwrap_loaded_model(_safe_torch_load(path_b, map_location="cpu"))
        if os.path.exists(path_head):
            lh_obj = _safe_torch_load(path_head, map_location="cpu")
            linear_head = lh_obj if isinstance(lh_obj, torch.nn.Module) else None
    except Exception as e:
        print(
            f"Warning: failed to load provided .pt models, will use fallback. Error: {e}"
        )
        model_a = None
        model_b = None
        linear_head = None

    if model_a is None:
        model_a = _build_fallback_vit(num_classes, image_size=model_a_img_size)
    if model_b is None:
        model_b = _build_fallback_vit(num_classes, image_size=model_b_img_size)

    if linear_head is None:
        linear_head = torch.nn.Identity()

    model_a = model_a.to(device)
    model_b = model_b.to(device)
    linear_head = linear_head.to(device)
    return model_a, model_b, linear_head


model_a, model_b, linear_head = _maybe_load_models(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        img_size=384,
        image_ids=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        if image_ids is not None:
            images = list(image_ids)
        else:
            images = sorted(os.listdir(data_dir))

        self.images = [
            fn for fn in images if os.path.isfile(os.path.join(data_dir, fn))
        ]

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample = pd.read_csv(sample_sub_path)
test_image_ids = sample["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_ids=test_image_ids,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)


def _as_logits(model_out):
    if isinstance(model_out, torch.Tensor):
        return model_out
    if (
        isinstance(model_out, (tuple, list))
        and len(model_out) > 0
        and isinstance(model_out[0], torch.Tensor)
    ):
        return model_out[0]
    if isinstance(model_out, dict):
        for k in ("logits", "out", "output"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs, model_b_inputs = (
                torch.cat(model_a_inputs, dim=0).to(device),
                torch.cat(model_b_inputs, dim=0).to(device),
            )
            filenames = list(filenames)

            model_a_outputs = _as_logits(model_a(model_a_inputs))
            model_b_outputs = _as_logits(model_b(model_b_inputs))

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (0.95 * model_a_mean_logits + 0.05 * model_b_mean_logits) / 2
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = _as_logits(model_a(model_a_inputs))
            model_b_outputs = _as_logits(model_b(model_b_inputs))

            outputs = (0.97 * model_a_outputs + 0.03 * model_b_outputs) / 2
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predicted:", len(all_preds), "files:", len(all_names))

pred_map = dict(zip(all_names, all_preds))
labels = [int(pred_map.get(img_id, 0)) for img_id in sample["image_id"].tolist()]
my_submission = pd.DataFrame({"image_id": sample["image_id"], "label": labels})

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", my_submission.shape)
print(my_submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3796346187.py in <cell line: 0>()
     92             filenames = list(filenames)
     93 
---> 94             model_a_outputs = _as_logits(model_a(model_a_inputs))
     95             model_b_outputs = _as_logits(model_b(model_b_inputs))
     96 

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

AssertionError: Wrong image height! Expected 224 but got 384!
