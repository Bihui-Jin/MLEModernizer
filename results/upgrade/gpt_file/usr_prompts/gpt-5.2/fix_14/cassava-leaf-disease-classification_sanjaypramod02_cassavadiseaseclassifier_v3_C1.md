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

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.8268359020852222

# 6. Current score

0.61659

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'We fix the TensorFlow/Protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` incompatibility seen with protobuf 6.x). Then we remove the hard dependency on a missing external weight file by training the same ViT classifier architecture on `train_tfrecords` for a small, fixed number of steps (so the pipeline always produces a model and a submission). Finally, we ensure the dataloaders are stable in Kaggle (no multiprocessing) and that the submission is aligned to `sample_submission.csv` and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'We fix the immediate crash by ensuring the protobuf runtime is forced to the pure-Python implementation *before* any TensorFlow/Transformers-related import happens, and by restarting the protobuf module state in-process to avoid the already-imported C++ runtime. Then we keep your training/inference logic identical but add a tiny, score-improving (and stability-safe) change: load official ImageNet-pretrained ViT weights from the already-installed `transformers` cache if available locally; otherwise fall back to your current random-init config (no internet required). Finally, we make the TFRecord parsing robust to minor schema differences and guarantee the submission is fully aligned to `sample_submission.csv` and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'We fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation and (critically) importing `google.protobuf` once to lock that runtime before importing TensorFlow. Then we make TFRecord parsing more robust by accepting either `image` or `jpeg` keys (both appear in different Cassava TFRecord variants), which prevents silent failures/empty outputs. Finally, we keep your model/training core logic intact but add lightweight, deterministic training improvements (shuffle+repeat in the TF input pipeline and correct label fallback) that should move accuracy up toward the target without changing the architecture or loss.'
- What this solution (achieved 0.61099) has done: 'We fix the TensorFlow import crash caused by protobuf 6.x by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow is imported before `transformers` (which triggers TF/protobuf descriptor code paths). We keep your ViT architecture and training/inference loop intact, only adjusting import order and adding a small, safe environment guard (`TF_CPP_MIN_LOG_LEVEL`) to reduce noisy logs. This should restore end-to-end execution and allow training on TFRecords as intended, producing `/kaggle/working/submission.csv`. With the crash fixed, the existing training improvements (shuffle+repeat, robust TFRecord keys, cached pretrained backbone if locally available) can again help move accuracy upward toward the target.'
- What this solution (achieved 0.61099) has done: 'We fix the immediate crash by ensuring the protobuf runtime is locked to the pure-Python implementation before importing TensorFlow/transformers, and by sanitizing any already-imported protobuf modules (the current order still allows the C++ runtime to be used, causing the `MessageFactory.GetPrototype` error). Then we keep your training/inference logic the same, but add a safe guard to prevent transformers from trying to import TensorFlow at all (we don’t need TF via transformers, only TF for TFRecord reading), which avoids the fragile TF↔protobuf code path. Finally, we keep submission generation identical but ensure the pipeline always reaches the CSV write step.'
- What this solution (achieved 0.61099) has done: 'We fix the TensorFlow↔protobuf crash by hard-blocking TensorFlow imports entirely (we only used TF to read TFRecords), and replace TFRecord reading with a pure-Python `tfrecord` reader so the same streaming train/test pipeline still works. We keep the exact ViT model architecture, optimizer, batch size, and training loop semantics, but switch the input pipeline to read images/labels from the existing JPEG folders and `train.csv`/`sample_submission.csv` instead. This removes the failing dependency path and should also improve accuracy toward the target because training reliably see correct labels and images without TFRecord schema/key issues. Finally, we keep submission alignment to `sample_submission.csv` and ensure `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.61099) has done: 'We fix the immediate crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *before* importing anything that might touch `google.protobuf` (including `transformers`). Then we keep your exact ViT model/training/inference logic, but make the import order deterministic and set `HF_HUB_OFFLINE=1` so `from_pretrained(..., local_files_only=True)` never tries to hit the network (stability-only). Finally, we ensure the script always reaches submission writing and that the submission is aligned to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.61099) has done: 'The crash is happening before training because importing `transformers` triggers a protobuf API path that is incompatible with protobuf 6.x (`MessageFactory.GetPrototype`). I fix this by (1) forcing the pure-Python protobuf runtime earlier and purging any already-imported protobuf modules, and (2) hard-blocking any TensorFlow/JAX/Flax import paths that `transformers` might touch, so the ViT import becomes stable. Then I keep your model, training loop, and preprocessing semantics the same, but switch to `ViTModel.from_pretrained(..., local_files_only=True)` as the primary path and fall back to the same random-init config as you already do, ensuring end-to-end execution and a valid `/kaggle/working/submission.csv`. This is expected to restore functionality and (when cached weights exist locally) improve accuracy toward the target without changing the core approach.'
- What this solution (achieved 0.61659) has done: 'We keep your ViT model, loss, optimizer, and basic training loop intact, but make two minimal changes that typically move Cassava accuracy up: (1) switch the image normalization to the standard ImageNet mean/std expected by pretrained ViT (your current 0.5/0.5 scaling mismatches the pretrained backbone), and (2) use a small, deterministic train/val split to pick the best checkpoint during the fixed 600 steps (no early stopping; we still run all steps, we just keep the best weights). These are calibration/selection tweaks rather than architecture or training-approach changes, and they should improve generalization toward your target score. Submission writing, ordering, and paths remain unchanged.'

# 9. Code solution

## === cell 0
import os, sys
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_JAX"] = "0"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

np.random.seed(0)

import google.protobuf  # noqa: F401

print(
    "Environment guards set; protobuf forced to pure-Python; transformers TF/Flax/JAX disabled."
)



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from transformers import ViTModel, ViTConfig

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Torch version:", torch.__version__)



## === cell 2
BASE1 = "/kaggle/input/cassava-leaf-disease-classification"
BASE2 = "/kaggle/input"

train_csv_path = (
    f"{BASE1}/train.csv"
    if os.path.exists(f"{BASE1}/train.csv")
    else f"{BASE2}/train.csv"
)
sample_path = (
    f"{BASE1}/sample_submission.csv"
    if os.path.exists(f"{BASE1}/sample_submission.csv")
    else f"{BASE2}/sample_submission.csv"
)

train_img_dir = (
    f"{BASE1}/train_images"
    if os.path.exists(f"{BASE1}/train_images")
    else f"{BASE2}/train_images"
)
test_img_dir = (
    f"{BASE1}/test_images"
    if os.path.exists(f"{BASE1}/test_images")
    else f"{BASE2}/test_images"
)

assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.exists(sample_path), f"Missing sample_submission.csv at {sample_path}"
assert os.path.isdir(train_img_dir), f"Missing train_images dir at {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

train_df = pd.read_csv(train_csv_path)
sample_df = pd.read_csv(sample_path)

print("train_df:", train_df.shape, "sample_df:", sample_df.shape)
print("train_img_dir:", train_img_dir)
print("test_img_dir:", test_img_dir)




## === cell 3
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_labels=5):
        super(ViTForImageClassification, self).__init__()
        self.num_labels = num_labels

        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,  # dropout handled below
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            num_labels=num_labels,
        )

        self.vit = None
        try:
            self.vit = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                local_files_only=True,
                ignore_mismatched_sizes=True,
            )
            print("Loaded cached pretrained ViT backbone.")
        except Exception as e:
            self.vit = ViTModel(config)
            print(
                "No cached pretrained ViT backbone found; using random init. Reason:",
                repr(e),
            )

        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return logits, loss




## === cell 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CassavaImageFolderTrainDataset(Dataset):
    """
    Direct JPEG loading from train_images + train.csv.
    Preprocessing: resize 224 and normalize with ImageNet mean/std (matches pretrained ViT).
    """

    def __init__(
        self, df: pd.DataFrame, image_dir: str, mean=IMAGENET_MEAN, std=IMAGENET_STD
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
        arr = (arr - self.mean) / self.std
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return ten

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])
        path = os.path.join(self.image_dir, image_id)
        x = self._load_image(path)
        return x, label


class CassavaImageFolderTestDataset(Dataset):
    """
    Direct JPEG loading from test_images + sample_submission ordering for stable submission alignment.
    """

    def __init__(self, image_ids, image_dir: str, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return ten

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = os.path.join(self.image_dir, image_id)
        x = self._load_image(path)
        return x, image_id




## === cell 5
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=0)
train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["label"]))
train_df_tr = train_df.iloc[train_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification(num_labels=5).to(device)

train_dataset = CassavaImageFolderTrainDataset(train_df_tr, train_img_dir)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)

val_dataset = CassavaImageFolderTrainDataset(train_df_val, train_img_dir)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
model.train()

max_steps = 600
running_loss = 0.0

best_val_acc = -1.0
best_state = None


@torch.no_grad()
def evaluate_accuracy(m: torch.nn.Module, loader: DataLoader) -> float:
    m.eval()
    correct = 0
    total = 0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.long)
        logits, _ = m(pixel_values, None)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    m.train()
    return float(correct) / float(total) if total > 0 else 0.0


step = 0
while step < max_steps:
    for pixel_values, labels in train_loader:
        step += 1
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(pixel_values, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        if step % 50 == 0:
            print(f"train step {step}/{max_steps} - loss {running_loss/50:.4f}")
            running_loss = 0.0

        if step % 200 == 0 or step == max_steps:
            val_acc = evaluate_accuracy(model, val_loader)
            print(f"val acc @ step {step}: {val_acc:.4f} (best {best_val_acc:.4f})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if step >= max_steps:
            break

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
print("Training complete; best_val_acc:", best_val_acc, "; model ready on", device)



## === cell 6
test_ids = sample_df["image_id"].tolist()
test_dataset = CassavaImageFolderTestDataset(test_ids, test_img_dir)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

print("Num test images:", len(test_dataset))



## === cell 7
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        predictions.extend(preds.tolist())
        image_ids.extend(list(ids))

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

out_path, submission_df.shape, submission_df.head()



## === cell 8
import subprocess

if os.path.exists("/kaggle/working/submission.csv"):
    subprocess.check_call(["bash", "-lc", "head -n 5 /kaggle/working/submission.csv"])
    subprocess.check_call(["bash", "-lc", "ls -l /kaggle/working/submission.csv"])
else:
    raise FileNotFoundError("submission.csv was not created.")
