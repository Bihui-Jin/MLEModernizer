# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings

import cv2
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 1
def get_tta(image_size, p=0.5):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    augs = A.Compose(
        [
            A.Resize(600, 600),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.PiecewiseAffine(p=0.3),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                ],
                p=p,
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    return augs




## === cell 2
class CassavaLeafDataset(Dataset):
    def __init__(
        self, root_dir, transforms, image_ids, cache_images=True, cache_tensors=True
    ):
        self.root_dir = root_dir
        self.transform = transforms
        self.image_ids = image_ids  # numpy array / list of strings

        self.cache_images = bool(cache_images)
        self.cache_tensors = False
        self._tensor_cache = None
        self._epoch = 0

        self._paths = [os.path.join(self.root_dir, iid) for iid in self.image_ids]

        self._images = None
        if self.cache_images:
            self._preload_images()

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.image_ids)

    def _imread_rgb_fast(self, img_path: str):
        data = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if data is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        return data[:, :, ::-1]  # BGR -> RGB

    def _preload_images(self):
        imgs = [None] * len(self._paths)
        for i, p in enumerate(self._paths):
            imgs[i] = self._imread_rgb_fast(p)
        self._images = imgs

    def _read_rgb(self, idx, img_path):
        if self.cache_images and self._images is not None:
            return self._images[idx]
        return self._imread_rgb_fast(img_path)

    def __getitem__(self, idx):
        img_path = self._paths[idx]
        image = self._read_rgb(idx, img_path)
        tensor = self.transform(image=image)["image"]
        return tensor, idx




## === cell 3
DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

test_transforms = get_tta(546)

df = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = df["image_id"].to_numpy()

try:
    cv2.setUseOptimized(True)
    cv2.setCacheSize(256 * 1024 * 1024)  # 256MB
except Exception:
    pass

test_data = CassavaLeafDataset(
    TEST_IMG_DIR, test_transforms, image_ids, cache_images=True, cache_tensors=False
)

cpu_cnt = os.cpu_count() or 2

num_workers = min(8, max(2, cpu_cnt // 2))

batch_size = 192 if torch.cuda.is_available() else 32

loader_kwargs = dict(
    batch_size=batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

try:
    test_loader = DataLoader(test_data, multiprocessing_context="fork", **loader_kwargs)
except Exception:
    test_loader = DataLoader(test_data, **loader_kwargs)

print(
    "Test samples:",
    len(test_data),
    "| num_workers:",
    num_workers,
    "| batch_size:",
    batch_size,
    "| cache_images:",
    True,
)



## === cell 4
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(in_features=1280, out_features=5, bias=True)

ckpt_candidates = [
    "../input/models/my-mobile.pth",
    "../input/my-mobile.pth",
    "../input/cassava-leaf-disease-classification/my-mobile.pth",
]
ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)

loaded = False
if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=True)
    loaded = True
    print("Loaded checkpoint:", ckpt_path)
else:
    try:
        imagenet_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        model.features.load_state_dict(
            imagenet_model.features.state_dict(), strict=True
        )
        loaded = True
        print("Loaded ImageNet pretrained weights for features (fallback).")
    except Exception as e:
        print("Warning: Could not load ImageNet weights fallback:", repr(e))

model = model.to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)
model.eval()
print("Weights loaded:", loaded)

model_forward = model
if hasattr(torch, "compile"):
    try:
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", fullgraph=False
        )
        print("torch.compile enabled for inference.")
    except Exception as e:
        print("torch.compile not used:", repr(e))



## === cell 5
n = df.shape[0]
epochs = 10

res = torch.zeros((n, 5), device=device)

torch.set_grad_enabled(False)

for ep in range(epochs):
    test_data.set_epoch(
        ep
    )  # keeps original intent; augmentations remain stochastic as before

    with torch.no_grad():
        for images, idx in test_loader:
            if device.type == "cuda":
                images = images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device, non_blocking=True)

            outputs = model_forward(images)  # [bs, 5]
            res.index_add_(0, idx.to(device, non_blocking=True), outputs)

res.div_(epochs)
pred = torch.argmax(res, dim=1)
print(pred[:20], pred.shape)



## === cell 6
labels = pred.detach().cpu().numpy().astype(int).tolist()
df["label"] = labels
df.to_csv("./submission.csv", index=False)
print(df.head())
print("Wrote ./submission.csv with shape:", df.shape)
