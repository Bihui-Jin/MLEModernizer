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

0.36165

# 6. Current score

0.02067

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41668) has done: 'I fixed the syntax problems (removed stray text and backticks), replaced the notebook‑style `!` commands with safe Python subprocess calls, and added a fallback import for `torchinfo`. These changes let the script run end‑to‑end, produce predictions, crop the padding correctly, and write a valid `submission.csv` matching the required format.'
- What this solution (achieved 0.42481) has done: 'I make four minimal adjustments that should reduce the RMSE without changing the model architecture.  
1) Simplify the training transform by removing the random blur and color‑jitter augmentations, which can hurt denoising performance.  
2) Give the loss function full weight to the RMSE term (the competition metric) by setting λ‑rmse = 1.0 and λ‑l1 = 0.0.  
3) Lower the optimizer learning rate to 1e‑3 for more stable convergence.  
4) Allow a longer training run (500 epochs) and increase the early‑stopping patience to 10 so the model can keep improving.'
- What this solution (achieved 0.49396) has done: 'I add a simple mean‑std normalization to the image tensors (so the network sees centered inputs) and remove the weight‑decay regularization from the Adam optimizer to let the model fit the data more closely. Both changes are tiny hyper‑parameter tweaks that keep the overall architecture and training loop unchanged, and they are expected to lower the RMSE toward the target score.'
- What this solution (achieved 0.41111) has done: 'I remove the mean‑std normalization from the image transforms so that both inputs and targets stay in the original 0‑1 range. The network already ends with a sigmoid, thus training on normalized values (‑1 to 1) forces it to approximate negatives it cannot output, hurting RMSE. By keeping the data un‑normalized the loss now compares like‑scaled tensors, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 0.42613) has done: 'I adjust the early‑stopping logic so that the counter is only increased when validation loss does not improve, and not altered when the learning‑rate scheduler updates the LR. This prevents premature stopping and lets the model train longer, which should lower the RMSE toward the target.'
- What this solution (achieved 0.43685) has done: 'I added the missing imports, made the zip‑extraction robust with `zipfile`, correctly set the training/validation/test directories, and imported all libraries required by the later cells (torch, torchvision, numpy, matplotlib, cv2, sklearn, etc.). These fixes unblock the entire pipeline, allow the model to be defined, trained, and used for inference, and finally produce a properly‑formatted `submission.csv` file.'
- What this solution (achieved 0.02067) has done: 'Implemented fixes to resolve size mismatches during training and ensure the submission contains exactly the required number of rows.  
- Added a safe cropping step in the training/validation loops so model outputs match the ground‑truth dimensions, preventing runtime errors.  
- Adjusted the test inference loop to load each original test image, retrieve its true height‑width, and crop the network prediction accordingly before flattening.  
These changes keep the original model and training logic intact while producing a correctly sized `submission.csv` matching the expected row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

torch.manual_seed(42)


def get_base_path():
    possible_paths = [
        os.path.join("input", "denoising-dirty-documents"),
        os.path.join("/kaggle", "input", "denoising-dirty-documents"),
    ]
    for p in possible_paths:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Dataset directory not found in expected locations.")


BASE = get_base_path()


class HybridLoss(nn.Module):
    """Weighted combination of RMSE (MSE) and L1 loss."""

    def __init__(self, lambda_rmse: float = 1.0, lambda_l1: float = 0.0):
        super().__init__()
        self.lambda_rmse = lambda_rmse
        self.lambda_l1 = lambda_l1
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        if self.lambda_rmse != 0:
            loss = loss + self.lambda_rmse * torch.sqrt(self.mse(pred, target) + 1e-8)
        if self.lambda_l1 != 0:
            loss = loss + self.lambda_l1 * self.l1(pred, target)
        return loss


class SimpleCNN(nn.Module):
    """A tiny encoder‑decoder suitable for the small image set."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # ½ size
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # ¼ size
        self.dec1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # ×2
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # ×2
        self.out = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.out(x))  # keep output in [0,1]
        return x


class DenoisingDataset(Dataset):
    """Dataset returning (noisy, clean) tensors."""

    def __init__(self, noisy_dir: str, clean_dir: str = None, transform=None):
        self.noisy_paths = sorted(
            [
                os.path.join(noisy_dir, f)
                for f in os.listdir(noisy_dir)
                if f.endswith(".png")
            ]
        )
        self.clean_dir = clean_dir
        self.transform = transform

    def __len__(self):
        return len(self.noisy_paths)

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # grayscale
        img = np.array(img).astype(np.float32) / 255.0  # to [0,1]
        return img

    def __getitem__(self, idx):
        noisy = self._load_image(self.noisy_paths[idx])
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        if self.transform:
            noisy = self.transform(noisy)

        if self.clean_dir is not None:
            clean_path = os.path.join(
                self.clean_dir, os.path.basename(self.noisy_paths[idx])
            )
            clean = self._load_image(clean_path)
            clean = torch.from_numpy(clean).unsqueeze(0)
            if self.transform:
                clean = self.transform(clean)
            return noisy, clean
        else:
            img_name = os.path.basename(self.noisy_paths[idx])
            return noisy, img_name


TRAIN_NOISY = os.path.join(BASE, "train")
TRAIN_CLEAN = os.path.join(BASE, "train_cleaned")
TEST_NOISY = os.path.join(BASE, "test")

full_dataset = DenoisingDataset(TRAIN_NOISY, TRAIN_CLEAN)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    random_state=42,
)

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = DataLoader(
    full_dataset, batch_size=1, sampler=train_sampler, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    full_dataset, batch_size=1, sampler=val_sampler, num_workers=0, pin_memory=True
)

test_dataset = DenoisingDataset(TEST_NOISY, clean_dir=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = SimpleCNN()
criterion = HybridLoss(lambda_rmse=1.0, lambda_l1=0.0)
optimizer = optim.Adam(
    model.parameters(), lr=5e-4, weight_decay=0.0
)  # slightly lower LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=False
)

best_state = model.state_dict()




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

best_val_loss = float("inf")
patience = 20  # increased patience for more stable training
early_stop_counter = 0
MAX_EPOCHS = 300  # allow more epochs for better convergence


def crop_to_match(out, tgt):
    """Crop the larger tensor to the size of the smaller one (height & width)."""
    _, _, h_out, w_out = out.shape
    _, _, h_tgt, w_tgt = tgt.shape
    h = min(h_out, h_tgt)
    w = min(w_out, w_tgt)
    out_cropped = out[:, :, :h, :w]
    tgt_cropped = tgt[:, :, :h, :w]
    return out_cropped, tgt_cropped


for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []
    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(noisy)

        if output.shape != clean.shape:
            output, clean = crop_to_match(output, clean)

        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            output = model(noisy)

            if output.shape != clean.shape:
                output, clean = crop_to_match(output, clean)

            loss = criterion(output, clean)
            val_losses.append(loss.item())

    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    scheduler.step(avg_val)

    if avg_val < best_val_loss - 1e-5:
        best_val_loss = avg_val
        early_stop_counter = 0
        best_state = model.state_dict()
    else:
        early_stop_counter += 1

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | train loss: {avg_train:.5f} | val loss: {avg_val:.5f}"
        )

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
model.eval()




## === cell 2
submission_rows = []

with torch.no_grad():
    for noisy, img_name in test_loader:
        img_name = img_name[0]  # e.g., "110.png"
        orig_path = os.path.join(TEST_NOISY, img_name)
        orig_img = Image.open(orig_path).convert("L")
        w_orig, h_orig = orig_img.size  # PIL returns (width, height)

        noisy = noisy.to(device)
        pred = model(noisy)  # shape (1,1,H_pred,W_pred)
        pred_np = pred.squeeze().cpu().numpy()  # (H_pred,W_pred) in [0,1]

        if pred_np.shape[0] != h_orig or pred_np.shape[1] != w_orig:
            pred_np = pred_np[:h_orig, :w_orig]

        img_id = os.path.splitext(img_name)[0]  # e.g., "110"
        h, w = pred_np.shape
        for r in range(h):
            for c in range(w):
                pixel_id = f"{img_id}_{r+1}_{c+1}"
                submission_rows.append([pixel_id, float(pred_np[r, c])])

submission_df = pd.DataFrame(submission_rows, columns=["id", "value"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Created submission file at {submission_path} with {len(submission_df)} rows.")
