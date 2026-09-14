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

0.8757932910244787

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'The script failed because the `efficientnet_pytorch` package is not available and the `crop_image` function referenced an undefined variable. I added a safe import that falls back to a timm EfficientNet model, introduced a device‑agnostic setup (CPU/GPU), corrected the variable name in `crop_image`, and updated all tensors to use the chosen device. The rest of the workflow stays unchanged, and the script now writes a proper `submission.csv` file.'
- What this solution (achieved 0.21263) has done: 'I switch the EfficientNet and SEResNeXt models to use pretrained weights (instead of random initialization) and fix the image preprocessing to correctly scale pixel values to the [0,1] range before normalisation. These small adjustments keep the original architecture and workflow unchanged while providing much more meaningful predictions, which should raise the accuracy from the current ~0.12 toward the target score.'
- What this solution (achieved 0.15209) has done: 'I simplify the preprocessing by removing the depth‑based cropping (which often discards useful leaf area) and resize every image to the standard 224 × 224 size expected by the ImageNet‑pretrained models. This keeps the same model architecture and ensemble logic but provides cleaner inputs, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.11323) has done: 'Implemented robust checkpoint loading, corrected image channel ordering, added reproducibility seed, and refined preprocessing. These adjustments ensure the pretrained fine‑tuned weights are actually used (if present), improve input normalization, and keep the existing model architecture unchanged, moving the validation accuracy toward the target score.'
- What this solution (achieved 0.78587) has done: 'I add a lightweight fine‑tuning stage that trains the EfficientNet model on the provided training images for a couple of epochs before performing inference. This keeps the original inference pipeline and model architectures unchanged, but adapts the pretrained weights to the cassava dataset, which should raise the validation accuracy markedly and move the score closer to the target. The new cells locate the training CSV and image folder, define a simple `Dataset`, run a short training loop, and then keep the existing ensemble inference unchanged.'

# 9. Code solution

## === cell 0
import os, glob, cv2, json, numpy as np, pandas as pd
import torch, torch.nn.functional as F
import timm
from torchvision import models

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True  # ensure cuDNN optimizations when GPU is present


def load_checkpoint(model, pattern):
    """Load checkpoint matching pattern and return True if found."""
    search_paths = [
        pattern,
        os.path.join("/kaggle/input", pattern),
        os.path.join("../input", pattern),
    ]
    for p in search_paths:
        for ckpt_path in glob.glob(p, recursive=True):
            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                print(f"Loaded checkpoint: {ckpt_path}")
                return True
    print(f"No checkpoint found for pattern: {pattern}")
    return False


try:
    from efficientnet_pytorch import EfficientNet

    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
except Exception:
    efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)

ckpt_eff = load_checkpoint(efficient, "**/eff_best.pth")
efficient.to(device).eval()

seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
ckpt_ser = load_checkpoint(seresnext, "**/seresnext_best.pth")
seresnext.to(device).eval()

print("Models have been loaded...")




## === cell 1
def processor(image):
    """Resize to 224×224, convert BGR→RGB, normalize, and return a torch tensor."""
    img = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(device)
    )
    return tensor




## === cell 2
def find_existing_path(candidates):
    for cand in candidates:
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return None


train_csv_candidates = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "train.csv",
]
train_csv_path = find_existing_path(train_csv_candidates)
if train_csv_path is None:
    raise RuntimeError("train.csv not found.")

train_img_candidates = [
    "../input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "train_images",
]
train_img_dir = find_existing_path(train_img_candidates)
if train_img_dir is None:
    raise RuntimeError("train_images directory not found.")


class CassavaDataset(torch.utils.data.Dataset):
    """
    Pre‑load all images into memory to avoid per‑sample disk I/O.
    Random horizontal flip augmentation is still applied at __getitem__ time.
    """

    def __init__(self, csv_path, img_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        self.preproc_imgs = []  # list of np.ndarray with shape (3,224,224)
        for img_name in self.df["image_id"]:
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )
            self.preproc_imgs.append(img.transpose(2, 0, 1).astype(np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_np = self.preproc_imgs[
            idx
        ].copy()  # copy so augmentation does not affect cache
        if self.augment and np.random.rand() < 0.5:
            img_np = img_np[:, :, ::-1]  # horizontal flip on numpy array
        tensor = torch.tensor(img_np, dtype=torch.float)
        label = int(self.df.iloc[idx]["label"])
        return tensor, label


if not (ckpt_eff and ckpt_ser) and device.type == "cpu":
    print(
        "No checkpoints found and running on CPU – skipping training to meet time limit."
    )
else:
    train_dataset = CassavaDataset(train_csv_path, train_img_dir, augment=True)

    num_workers = min(4, os.cpu_count() or 0) if device.type == "cuda" else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_eff = torch.optim.Adam(efficient.parameters(), lr=1e-4)
    optimizer_ser = torch.optim.Adam(seresnext.parameters(), lr=1e-4)

    efficient.train()
    seresnext.train()
    num_epochs = 4
    for epoch in range(num_epochs):
        running_loss_eff = 0.0
        running_loss_ser = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer_eff.zero_grad()
            optimizer_ser.zero_grad()

            outputs_eff = efficient(imgs)
            outputs_ser = seresnext(imgs)

            loss_eff = criterion(outputs_eff, targets)
            loss_ser = criterion(outputs_ser, targets)

            loss_eff.backward()
            loss_ser.backward()

            optimizer_eff.step()
            optimizer_ser.step()

            running_loss_eff += loss_eff.item() * imgs.size(0)
            running_loss_ser += loss_ser.item() * imgs.size(0)

        epoch_loss_eff = running_loss_eff / len(train_loader.dataset)
        epoch_loss_ser = running_loss_ser / len(train_loader.dataset)
        print(
            f"EfficientNet Epoch [{epoch+1}/{num_epochs}] - Training loss: {epoch_loss_eff:.4f}"
        )
        print(
            f"SEResNeXt Epoch [{epoch+1}/{num_epochs}] - Training loss: {epoch_loss_ser:.4f}"
        )

efficient.eval()
seresnext.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2634147803.py in <cell line: 0>()
     90         running_loss_eff = 0.0
     91         running_loss_ser = 0.0
---> 92         for imgs, targets in train_loader:
     93             imgs = imgs.to(device)
     94             targets = targets.to(device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2634147803.py", line 60, in __getitem__
    tensor = torch.tensor(img_np, dtype=torch.float)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 


## === cell 3
test_dir_candidates = [
    "../input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "test_images",
]
test_dir = None
for cand in test_dir_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    raise RuntimeError("Test images directory not found.")

files = sorted(glob.glob(os.path.join(test_dir, "*")))
batch_size = 32
names, labels = [], []

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

num_test = len(files)
test_np = np.empty((num_test, 3, 224, 224), dtype=np.float32)
for idx, p in enumerate(files):
    img = cv2.imread(p)
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - mean) / std
    test_np[idx] = img.transpose(2, 0, 1)  # C, H, W

with torch.no_grad():
    for i in range(0, num_test, batch_size):
        batch_tensor = torch.from_numpy(test_np[i : i + batch_size]).to(device)

        se_out = seresnext(batch_tensor)
        eff_out = efficient(batch_tensor)

        se_prob = F.softmax(se_out, dim=1)
        eff_prob = F.softmax(eff_out, dim=1)
        avg_prob = (se_prob + eff_prob) / 2.0

        preds = torch.argmax(avg_prob, dim=1).cpu().numpy()
        names.extend([os.path.basename(p) for p in files[i : i + batch_size]])
        labels.extend(preds.tolist())

submission = pd.DataFrame({"image_id": names, "label": labels})
submission.to_csv("submission.csv", index=False, header=True)
print("Submission file written to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
