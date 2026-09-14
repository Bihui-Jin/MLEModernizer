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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.9

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

# 4. Data file paths

```
/
    kaggle/
        data/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.9160158399570948

# 6. Current score

0.74629

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We fix the CUDA‑only device handling (fallback to CPU), correctly map model weights to the chosen device, guard model loading so the script continues even if a weight file is missing, and ensure the prediction lists are always defined before they are used. This makes the pipeline run end‑to‑end and produces a valid `submission.csv` while keeping the original architecture and ensembling logic intact, which should move the score toward the target.'
- What this solution (achieved 0.0) has done: 'The changes add a more sensible default for missing model predictions by using the training set’s average diagnosis, and replace the manual thresholding with a simple rounding after clipping to the valid range. This should produce a more realistic distribution of predictions, moving the quadratic weighted kappa score closer to the target while keeping the core model logic unchanged.'
- What this solution (achieved 0.0) has done: 'The changes fix the incorrect relative paths for the pretrained model files by constructing absolute paths under `/kaggle/input`. This lets the script actually load the provided weights, generate meaningful predictions, and therefore move the quadratic weighted kappa score much closer to the target.'
- What this solution (achieved 0.0) has done: 'I add a lightweight fallback model that is trained quickly on the available training data when the custom pretrained weights are missing. This model (a ResNet‑18 with ImageNet weights) provides reasonable continuous predictions, which are then merged into the submission in place of the all‑zero defaults, moving the quadratic weighted kappa score far above 0 and closer to the target while keeping the original architecture and ensemble logic unchanged.'
- What this solution (achieved 0.0) has done: 'The changes batch‑process the test images during inference and enable CuDNN benchmarking, which greatly speeds up the forward passes on GPU while keeping the exact same model logic and prediction order.'
- What this solution (achieved 0.0) has done: 'The changes introduce a simple image‑tensor cache so each test image is read, resized, and transformed only once per required size, eliminating repeated I/O and Pillow processing across the three models. The prediction loop now works directly on these cached tensors (with a larger batch size) while preserving the original flip‑augmentation logic, ensuring identical outputs but much faster runtime.'
- What this solution (achieved 0.0) has done: 'The fixes add missing imports, define the device and base input path, replace the undefined `se_resnet50` with a simple ResNet‑50 implementation, correct the `torch.flip` call, and import utilities (`glob`, `re`, etc.) so that every cell can run. With these minimal, targeted changes the script now runs end‑to‑end and writes a valid `submission.csv` file, moving the solution toward the target score.'
- What this solution (achieved 0.0) has done: 'I added the missing imports, defined a safe device fallback, set the correct base input path, guarded model loading against undefined helper functions, and created a fallback prediction that uses the mean diagnosis from the training set when no model predictions are available. The script now builds a complete predictions list for all test IDs and writes a properly‑formatted `submission.csv`, ensuring the pipeline runs end‑to‑end and produces a valid Kaggle submission file.'
- What this solution (achieved 0.14259) has done: 'I add a lightweight, intensity‑based fallback that derives per‑class average grayscale values from the training images and uses these to predict each test image when the ensemble models provide no output. This keeps the original inference pipeline unchanged, adds only a small, deterministic computation, and replaces the constant‑mean prediction with a varied one that should move the quadratic weighted‑kappa score toward the target.'
- What this solution (achieved 0.74629) has done: 'I speed up the pipeline by (1) adding a resize step to the Densenet transform so every model works on a small 224 × 224 image, (2) increasing the DataLoader `num_workers` and enabling `persistent_workers` for all loaders, (3) raising the batch size to 256 where memory permits, (4) fixing a typo that prevented the 512‑pixel SE‑ResNet model from loading (which caused the fallback training to run), and (5) using a few workers for the fallback training DataLoader. These changes keep the exact model architectures, loss, and inference logic unchanged while removing unnecessary I/O and reducing per‑image computation, ensuring the script finishes well under the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn = torch.backends.cudnn
cudnn.benchmark = True

BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"



## === cell 1
TEST_IMAGE_DIR = os.path.join(BASE_INPUT, "test_images")
TEST_IDS = None  # cached list of ids in deterministic order


def get_test_ids():
    """Return sorted list of test image ids; cache for reuse."""
    global TEST_IDS
    if TEST_IDS is None:
        paths = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.png")))
        TEST_IDS = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return TEST_IDS


class TestImageDataset(Dataset):
    """Dataset that loads a test image, applies a given transform and returns (tensor, id)."""

    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_id


def make_predictions(model, transform, batch_size=256, num_workers=8):
    """Run inference over the test set using a DataLoader.

    Increased `batch_size`, `num_workers`, and enabled `persistent_workers`
    to reduce I/O and compute overhead while preserving the exact inference logic.
    """
    model.eval()
    ids = get_test_ids()
    dataset = TestImageDataset(ids, TEST_IMAGE_DIR, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    predictions = []
    with torch.no_grad():
        for batch_imgs, batch_ids in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            out = model(batch_imgs)
            out_flip = model(torch.flip(batch_imgs, dims=(3,)))  # horizontal flip
            batch_pred = (out.squeeze() + out_flip.squeeze()) / 2.0
            if batch_pred.dim() == 2:  # shape [B, 5]
                probs = torch.softmax(batch_pred, dim=1)
                batch_pred = (
                    probs * torch.arange(5, device=device, dtype=torch.float32)
                ).sum(dim=1)
            batch_pred = batch_pred.cpu().numpy()
            for img_id, pred in zip(batch_ids, batch_pred):
                predictions.append((img_id, float(pred)))
    return predictions




## === cell 2
predictions_densenet = []
MODEL_PATH_DENSENET = os.path.join(
    BASE_INPUT, "densenet121", "model_densenet121_bs64_30.pth"
)
if os.path.isfile(MODEL_PATH_DENSENET):
    if "get_densenet121_gem" in globals():
        model_dn = get_densenet121_gem(pretrain=False)
        model_dn = model_dn.to(device)
        state_dn = torch.load(MODEL_PATH_DENSENET, map_location=device)
        model_dn.load_state_dict(state_dn)
        norm_dn = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        predictions_densenet = make_predictions(
            model_dn, transform=norm_dn, batch_size=256, num_workers=8
        )
    else:
        print("Warning: get_densenet121_gem not defined; skipping Densenet121.")
else:
    print("Warning: Densenet121 weights not found; skipping this model.")




## === cell 3
predictions_seresnet = []
MODEL_PATH_SERESNET = os.path.join(BASE_INPUT, "seresnet50testpseudo", "model10.pth")
if os.path.isfile(MODEL_PATH_SERESNET):
    if "get_se_resnet50_gem" in globals():
        model_sr = get_se_resnet50_gem(pretrain=False)
        model_sr = model_sr.to(device)
        state_sr = torch.load(MODEL_PATH_SERESNET, map_location=device)
        model_sr.load_state_dict(state_sr)
        norm_sr = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        predictions_seresnet = make_predictions(
            model_sr, transform=norm_sr, batch_size=256, num_workers=8
        )
    else:
        print("Warning: get_se_resnet50_gem not defined; skipping SE-ResNet50.")
else:
    print("Warning: SE-ResNet50 weights not found; skipping this model.")




## === cell 4
predictions_seresnet_512 = []
MODEL_PATH_SERESNET_512 = os.path.join(BASE_INPUT, "seresnet50-512", "model30_512.pth")
if os.path.isfile(MODEL_PATH_SERESNET_512):
    if "get_se_resnet50_gem" in globals():
        model_sr512 = get_se_resnet50_gem(pretrain=False)
        model_sr512 = model_sr512.to(device)
        state_sr512 = torch.load(MODEL_PATH_SERESNET_512, map_location=device)
        model_sr512.load_state_dict(state_sr512)
        norm_sr512 = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        predictions_seresnet_512 = make_predictions(
            model_sr512, transform=norm_sr512, batch_size=256, num_workers=8
        )
    else:
        print("Warning: get_se_resnet50_gem not defined; skipping SE-ResNet50 512px.")
else:
    print("Warning: SE-ResNet50 512px weights not found; skipping this model.")




## === cell 5
train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
mean_label = train_df["diagnosis"].mean()

all_preds = predictions_densenet + predictions_seresnet + predictions_seresnet_512

if not all_preds:
    print(
        "No pretrained model predictions found – training lightweight fallback model."
    )

    class TrainImageDataset(Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_id = row["id_code"]
            label = int(row["diagnosis"])
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_image_dir = os.path.join(BASE_INPUT, "train_images")
    train_dataset = TrainImageDataset(train_df, train_image_dir, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    fallback_model = models.resnet18(pretrained=True)
    fallback_model.fc = nn.Linear(fallback_model.fc.in_features, 5)
    fallback_model = fallback_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fallback_model.parameters(), lr=1e-3)

    fallback_model.train()
    for epoch in range(2):  # lightweight training, just a couple of epochs
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = fallback_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Fallback model training completed.")

    fallback_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    fallback_preds = make_predictions(
        fallback_model, transform=fallback_transform, batch_size=256, num_workers=8
    )
    all_preds = fallback_preds

pred_dict = {}
count_dict = {}
for img_id, pred in all_preds:
    pred_dict[img_id] = pred_dict.get(img_id, 0.0) + pred
    count_dict[img_id] = count_dict.get(img_id, 0) + 1

final_preds = {img_id: pred_dict[img_id] / count_dict[img_id] for img_id in pred_dict}

submission = pd.DataFrame(
    {
        "id_code": list(final_preds.keys()),
        "diagnosis": [int(round(min(max(v, 0), 4))) for v in final_preds.values()],
    }
)

submission = (
    submission.set_index("id_code").loc[sorted(submission["id_code"])].reset_index()
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
