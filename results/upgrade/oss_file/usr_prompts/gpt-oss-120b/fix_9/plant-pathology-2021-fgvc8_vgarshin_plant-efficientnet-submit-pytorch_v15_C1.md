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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8245798707294568

# 6. Current score

0.17301

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The fix adds a proper EfficientNet wrapper so the model can be constructed when the external library is missing, and filters the test‑image list to include only actual image files, preventing mismatched row counts in the submission. These minimal changes resolve the import error, ensure all predicted rows align with the submission file, and allow the notebook to run end‑to‑end producing a valid `submission.csv`.'
- What this solution (achieved 0.28786) has done: 'I fixed the EfficientNet wrapper so it correctly extracts the feature size from the torchvision classifier, added proper label mapping based on the training CSV, and implemented a lightweight fine‑tuning step that trains the model on the available training data when pre‑trained weights are missing. These changes resolve the attribute error, produce meaningful predictions, and raise the validation‑style F1 score toward the target while still outputting a correct `submission.csv`.'
- What this solution (achieved 0.24727) has done: 'I fixed the NameError by removing the unconditional deletion of `state_dict` and `model`, and I added a safe conditional clean‑up. I also increased the lightweight fine‑tuning epochs from 2 to 5 so the models learn more from the training data, which should boost the mean F1‑score while keeping the overall architecture unchanged. The rest of the pipeline remains the same and now reliably writes a proper `submission.csv`.'
- What this solution (achieved 0.17301) has done: 'The script was spending excessive time because it fell back to full training when any model checkpoint was missing, which can take many minutes. The fix removes this costly training path: it now aborts with a clear error if a required checkpoint isn’t found, ensuring only inference runs (the intended fast path). No core logic, architecture, or inference steps are altered, preserving exact predictions when the checkpoints exist.'

# 9. Code solution

## === cell 0
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

torch.set_num_threads(os.cpu_count() or 1)

torch.backends.cudnn.benchmark = True

try:
    from efficientnet_pytorch import model as enet

    _HAS_EFFNET = True
except ModuleNotFoundError:
    _HAS_EFFNET = False

    class _EffNetWrapper:
        class EfficientNet:
            @staticmethod
            def from_name(name):
                backbone = name.replace("-", "_")
                if "b0" in backbone:
                    return torchvision.models.efficientnet_b0(pretrained=True)
                elif "b1" in backbone:
                    return torchvision.models.efficientnet_b1(pretrained=True)
                else:
                    return torchvision.models.efficientnet_b0(pretrained=True)

    enet = _EffNetWrapper

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
try:
    get_ipython().system(
        "pip install ../input/efficientnet-pytorch/EfficientNet-PyTorch-1.0 -f ./ --no-index"
    )
except Exception:
    pass




## === cell 2
TEST = True
VER = "v6"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"
TTAS = [0, 1]  # test‑time augmentations
FOLDS = [0, 1, 2]  # model folds
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()

params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.exists(params_path):
    with open(params_path) as f:
        params = json.load(f)
else:
    params = {
        "labels_": {"healthy": 0},
        "labels": {"0": "healthy"},
        "workers": 2,
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
        "ths": {"0": 0.5},
    }
    print("Warning: params.json not found – using fallback parameters.")
LABELS_ = params["labels_"]
LABELS = params["labels"]
DEFAULT_WORKERS = os.cpu_count() or 2
WORKERS = min(8, DEFAULT_WORKERS) if KAGGLE else params.get("workers", 2)

train_csv_path = os.path.join(DATA_PATH, "train.csv")
if os.path.exists(train_csv_path):
    df_train_full = pd.read_csv(train_csv_path)
    unique_labels = set()
    for lab_str in df_train_full["labels"].fillna("").values:
        unique_labels.update(lab_str.split())
    LABELS_ = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
    LABELS = {str(idx): lbl for lbl, idx in LABELS_.items()}
    params["labels_"] = LABELS_
    params["labels"] = LABELS
else:
    print("Warning: train.csv not found – using fallback label mapping.")

WORKERS = min(8, DEFAULT_WORKERS) if KAGGLE else params.get("workers", 2)




## === cell 3
image_files = [
    f for f in os.listdir(IMGS_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
df_sub = pd.DataFrame(image_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder; will be overwritten after inference




## === cell 4
def flip_axis_tensor(tensor, axis):
    if axis == 1:
        return torch.flip(tensor, dims=[2])
    elif axis == 2:
        return torch.flip(tensor, dims=[3])
    elif axis == 3:
        return torch.flip(tensor, dims=[2, 3])
    else:
        return tensor


class PlantDataset(data.Dataset):
    _cache = {}

    def __init__(
        self, df, size, labels_dict, transform=None, tta=0, is_train=False, img_dir=None
    ):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels_dict = labels_dict
        self.transform = transform
        self.tta = tta
        self.is_train = is_train
        self.img_dir = img_dir if img_dir is not None else IMGS_PATH

    def __len__(self):
        return self.df.shape[0]

    def _load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = os.path.join(self.img_dir, img_name)

        if not self.is_train:
            if img_path in PlantDataset._cache:
                img = PlantDataset._cache[img_path]
            else:
                img = self._load_image(img_path)
                PlantDataset._cache[img_path] = img
        else:
            img = self._load_image(img_path)

        if self.is_train:
            img = img.transpose(2, 0, 1)
            label_vec = np.zeros(len(self.labels_dict), dtype=np.float32)
            for lbl in row.labels.split():
                if lbl in self.labels_dict:
                    label_vec[self.labels_dict[lbl]] = 1.0
            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                label_vec, dtype=torch.float32
            )
        else:
            img = img.transpose(2, 0, 1)
            return torch.tensor(img, dtype=torch.float32)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.EfficientNet.from_name(params["backbone"])
        if hasattr(self.enet, "_fc"):
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
        elif hasattr(self.enet, "classifier"):
            if isinstance(self.enet.classifier, nn.Sequential):
                linear_layer = None
                for m in self.enet.classifier.modules():
                    if isinstance(m, nn.Linear):
                        linear_layer = m
                        break
                if linear_layer is None:
                    raise AttributeError("Linear layer not found in classifier.")
                nc = linear_layer.in_features
                self.enet.classifier = nn.Identity()
            else:
                nc = self.enet.classifier.in_features
                self.enet.classifier = nn.Identity()
        else:
            raise AttributeError(
                "Backbone does not have a fully‑connected layer attribute."
            )
        self.myfc = nn.Sequential(
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
models = []
missing_checkpoints = []
for n_fold in FOLDS:
    try:
        model = EffNet(params, out_dim=len(LABELS_))
        path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {path}")
        loaded = True
    except Exception as e:
        print(f"Could not load model for fold {n_fold} ({e});")
        missing_checkpoints.append(path)
        model = EffNet(params, out_dim=len(LABELS_))
        loaded = False
    model = model.to(DEVICE).eval()
    models.append((model, loaded))

if missing_checkpoints:
    raise RuntimeError(
        f"Missing model checkpoints: {missing_checkpoints}. "
        "Provide all required .pth files to avoid expensive training fallback."
    )

if "state_dict" in globals():
    del state_dict
if "model" in globals():
    del model
gc.collect()





## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/742780763.py in <cell line: 0>()
     25 
     26 if missing_checkpoints:
---> 27     raise RuntimeError(
     28         f"Missing model checkpoints: {missing_checkpoints}. "
     29         "Provide all required .pth files to avoid expensive training fallback."

RuntimeError: Missing model checkpoints: ['../input/plant-models-v6/model_best_0.pth', '../input/plant-models-v6/model_best_1.pth', '../input/plant-models-v6/model_best_2.pth']. Provide all required .pth files to avoid expensive training fallback.

## === cell 6
inference_dataset = PlantDataset(
    df=df_sub,
    size=params["img_size"],
    labels_dict=None,
    transform=None,
    tta=0,
    is_train=False,
)
loader = torch.utils.data.DataLoader(
    inference_dataset,
    batch_size=params["batch_size"],
    sampler=SequentialSampler(inference_dataset),
    num_workers=WORKERS,
    pin_memory=True,
)




## === cell 7
def get_labels(row, labels_dict, thresholds):
    try:
        idx = [i for i, x in enumerate(row) if x > thresholds.get(str(i), 0.5)]
        names = [labels_dict.get(str(i), "") for i in idx]
        return "healthy" if ("healthy" in names or len(names) == 0) else " ".join(names)
    except Exception as e:
        print(f"Label extraction error: {e}")
        return "healthy"


all_fold_preds = []
for model, _ in models:
    tta_preds = []
    for tta in TTAS:
        batch_preds = []
        for batch in loader:
            if tta != 0:
                batch = flip_axis_tensor(batch, tta)
            batch = batch.to(DEVICE)
            with torch.no_grad():
                out = torch.sigmoid(model(batch))
            batch_preds.append(out.cpu().numpy())
        tta_preds.append(np.concatenate(batch_preds, axis=0))  # (N, C)
    tta_preds = np.mean(np.stack(tta_preds, axis=0), axis=0)  # (N, C)
    all_fold_preds.append(tta_preds)

logits = np.mean(np.stack(all_fold_preds, axis=0), axis=0)  # (N, C)

df_sub["labels"] = [get_labels(row, LABELS, params["ths"]) for row in logits]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")




## === cell 8
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
