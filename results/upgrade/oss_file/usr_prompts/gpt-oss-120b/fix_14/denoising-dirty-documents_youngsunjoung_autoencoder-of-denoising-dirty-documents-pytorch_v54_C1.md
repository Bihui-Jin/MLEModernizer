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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.13

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.45275

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, subprocess, zipfile
from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


def maybe_unzip(zip_path, extract_to):
    if not os.path.isfile(zip_path):
        return
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        try:
            subprocess.check_call(["unzip", "-q", zip_path, "-d", extract_to])
        except Exception:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)


base_extract_dir = "/kaggle/working/denoising-dirty-documents"

if not (
    os.path.isdir(os.path.join(base_extract_dir, "train"))
    and os.path.isdir(os.path.join(base_extract_dir, "train_cleaned"))
    and os.path.isdir(os.path.join(base_extract_dir, "test"))
):
    maybe_unzip("/kaggle/input/denoising-dirty-documents/train.zip", base_extract_dir)
    maybe_unzip("/kaggle/input/denoising-dirty-documents/test.zip", base_extract_dir)
    maybe_unzip(
        "/kaggle/input/denoising-dirty-documents/train_cleaned.zip",
        base_extract_dir,
    )


def find_dir(keyword):
    """
    Walk the extraction directory and return the first sub‑folder whose name contains `keyword`
    (case‑insensitive) and that holds at least one PNG file.
    """
    for root, dirs, _ in os.walk(base_extract_dir):
        for d in dirs:
            if keyword.lower() in d.lower():
                candidate = os.path.join(root, d)
                if any(f.lower().endswith(".png") for f in os.listdir(candidate)):
                    return candidate
    fallback = os.path.join(base_extract_dir, "denoising-dirty-documents", keyword)
    if os.path.isdir(fallback) and any(
        f.lower().endswith(".png") for f in os.listdir(fallback)
    ):
        return fallback
    raise FileNotFoundError(f"Unable to locate directory containing '{keyword}'")


train_dir = find_dir("train")
train_cleaned_dir = find_dir("train_cleaned")
test_dir = find_dir("test")



## === cell 1
train_files = sorted(
    [
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.lower().endswith(".png")
    ]
)
cleaned_files = sorted(
    [
        os.path.join(train_cleaned_dir, f)
        for f in os.listdir(train_cleaned_dir)
        if f.lower().endswith(".png")
    ]
)

train_files, val_files, cleaned_train, cleaned_val = train_test_split(
    train_files, cleaned_files, test_size=2 / 9, random_state=42
)


class ImagePairDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths=None, transform=None):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.noisy_paths)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # normalize to [0,1]
        return img

    def __getitem__(self, idx):
        noisy_img = self._load_image(self.noisy_paths[idx])
        noisy_tensor = self.transform(noisy_img)
        if self.clean_paths is not None:
            clean_img = self._load_image(self.clean_paths[idx])
            clean_tensor = self.transform(clean_img)
            return noisy_tensor, clean_tensor
        else:
            return noisy_tensor, os.path.basename(self.noisy_paths[idx])


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = ImagePairDataset(train_files, cleaned_train, transform=transform)
val_dataset = ImagePairDataset(val_files, cleaned_val, transform=transform)
test_dataset = ImagePairDataset(
    sorted(
        [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(".png")
        ]
    ),
    clean_paths=None,
    transform=transform,
)

batch_size = 8
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # -> /2
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # -> /4
        self.dec1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # -> /2
        self.dec2 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)  # -> original
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x


model = SimpleAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

epochs = 5
patience = 5
early_stop_counter = 0
best_val_loss = float("inf")
best_model_state = None

epoch = 0
while epoch < epochs:
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    for train_imgs, clean_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(train_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_rmse += RMSELoss()(outputs, clean_imgs).item()
    train_loss /= len(train_loader)
    train_rmse /= len(train_loader)
    print(
        f"Epoch [{epoch+1}/{epochs}] Train loss: {train_loss:.4f} RMSE: {train_rmse:.4f}"
    )

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for val_imgs, val_clean in val_loader:
            val_imgs = val_imgs.to(device)
            val_clean = val_clean.to(device)
            outputs = model(val_imgs)
            loss = criterion(outputs, val_clean)
            val_loss += loss.item()
            val_rmse += RMSELoss()(outputs, val_clean).item()
    val_loss /= len(val_loader)
    val_rmse /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print(f"  New best val loss: {val_loss:.4f} (RMSE {val_rmse:.4f})")
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop cnt: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered")
        break
    epoch += 1

if best_model_state is not None:
    torch.save(best_model_state, "best_model.pth")
    print("Best model saved.")
else:
    print("No model saved.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1880951875.py in <cell line: 0>()
     42     train_loss = 0.0
     43     train_rmse = 0.0
---> 44     for train_imgs, clean_imgs in train_loader:
     45         train_imgs = train_imgs.to(device)
     46         clean_imgs = clean_imgs.to(device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     53         else:
     54             data = self.dataset[possibly_batched_index]
---> 55         return self.collate_fn(data)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in default_collate(batch)
    396         >>> default_collate(batch)  # Handle `CustomType` automatically
    397     """
--> 398     return collate(batch, collate_fn_map=default_collate_fn_map)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    209 
    210         if isinstance(elem, tuple):
--> 211             return [
    212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in <listcomp>(.0)
    210         if isinstance(elem, tuple):
    211             return [
--> 212                 collate(samples, collate_fn_map=collate_fn_map)
    213                 for samples in transposed
    214             ]  # Backwards compatibility.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate(batch, collate_fn_map)
    153     if collate_fn_map is not None:
    154         if elem_type in collate_fn_map:
--> 155             return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
    156 
    157         for collate_type in collate_fn_map:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py in collate_tensor_fn(batch, collate_fn_map)
    270         storage = elem._typed_storage()._new_shared(numel, device=elem.device)
    271         out = elem.new(storage).resize_(len(batch), *list(elem.size()))
--> 272     return torch.stack(batch, 0, out=out)
    273 
    274 

RuntimeError: stack expects each tensor to be equal size, but got [1, 420, 540] at entry 0 and [1, 258, 540] at entry 5

## === cell 3
if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()

submission_rows = []
with torch.no_grad():
    for img_tensor, img_name in test_loader:
        img_tensor = img_tensor.to(device)
        pred = model(img_tensor).cpu().squeeze(0).squeeze(0).numpy()  # shape HxW
        orig_path = os.path.join(test_dir, img_name)
        orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        h, w = orig_img.shape
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        for i in range(h):
            for j in range(w):
                pixel_id = f"{os.path.splitext(img_name)[0]}_{i+1}_{j+1}"
                pixel_val = float(pred_resized[i, j])
                submission_rows.append((pixel_id, pixel_val))

submission_df = pd.DataFrame(submission_rows, columns=["id", "value"])
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2449319474.py in <cell line: 0>()
     10         pred = model(img_tensor).cpu().squeeze(0).squeeze(0).numpy()  # shape HxW
     11         # Load original image to get dimensions (they may have been resized by transforms)
---> 12         orig_path = os.path.join(test_dir, img_name)
     13         orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
     14         h, w = orig_img.shape

/usr/lib/python3.11/posixpath.py in join(a, *p)

/usr/lib/python3.11/genericpath.py in _check_arg_types(funcname, *args)

TypeError: join() argument must be str, bytes, or os.PathLike object, not 'tuple'
