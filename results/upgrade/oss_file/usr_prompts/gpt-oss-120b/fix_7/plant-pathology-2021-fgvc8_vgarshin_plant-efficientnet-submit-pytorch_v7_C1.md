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

0.8039335180055411

# 6. Current score

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.30566) has done: 'I filtered the test image list to include only valid image files and added a safeguard in the dataset loader to raise a clear error if an image cannot be read. This prevents the OpenCV assertion failure, ensures the DataLoader processes only proper images, and allows the submission dataframe (`df_sub`) to be created so the final CSV is written correctly. No changes were made to the core modeling logic, preserving the original approach while fixing the runtime errors.'
- What this solution (achieved 0.30565) has done: 'I lowered the cosine‑similarity threshold a bit (to 0.2) and, when building the class centroids, accumulated *already‑normalized* feature vectors. This aligns the centroid construction with the cosine‑similarity scoring used at inference and typically yields higher recall without altering the core model architecture or training loop.'
- What this solution (achieved 0.30565) has done: 'I keep the overall centroid‑based pipeline but improve the feature extractor and inference a bit to raise the F1 score.  
1) Switch the backbone from EfficientNet‑B0 to a stronger EfficientNet‑B4 (still pretrained, same feature‑extraction logic).  
2) Add a lightweight test‑time augmentation: compute features for the original and horizontally‑flipped images and average their cosine similarities before label assignment.  
3) Raise the cosine‑similarity threshold slightly (to 0.30) to better balance precision/recall after the stronger features and TTA.  
These minimal edits preserve the core centroid methodology while expectedly moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os, json, time, gc
import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

try:
    from efficientnet_pytorch import model as enet
except ModuleNotFoundError:

    class _EffNetWrapper:
        @staticmethod
        def from_name(name):
            backbone_map = {
                "efficientnet-b0": torchvision.models.efficientnet_b0,
                "efficientnet-b1": torchvision.models.efficientnet_b1,
                "efficientnet-b2": torchvision.models.efficientnet_b2,
                "efficientnet-b3": torchvision.models.efficientnet_b3,
                "efficientnet-b4": torchvision.models.efficientnet_b4,
                "efficientnet-b5": torchvision.models.efficientnet_b5,
                "efficientnet-b6": torchvision.models.efficientnet_b6,
                "efficientnet-b7": torchvision.models.efficientnet_b7,
            }
            if name not in backbone_map:
                raise ValueError(
                    f"Backbone {name} not recognized for fallback EfficientNet"
                )
            model = backbone_map[name](pretrained=True)
            return model

    enet = _EffNetWrapper

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
TEST = True
VER = "v1"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.30
IMG_SIZE = 224

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
TRAIN_IMGS_PATH = f"{DATA_PATH}/train_images"
start_time = time.time()



## === cell 2
train_csv_path = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv_path)

unique_labels = set()
for lbls in train_df["labels"]:
    unique_labels.update(lbls.split())
unique_labels = sorted(unique_labels)
LABELS_ = {i: lbl for i, lbl in enumerate(unique_labels)}  # idx → label
LABELS = {lbl: i for i, lbl in enumerate(unique_labels)}  # label → idx
NUM_LABELS = len(unique_labels)




## === cell 3
class PlantDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W

        if self.is_train:
            label_str = row["labels"]
            label_idxs = [LABELS[l] for l in label_str.split()]
            multi_hot = torch.zeros(NUM_LABELS, dtype=torch.float)
            multi_hot[label_idxs] = 1.0
            return img, multi_hot
        else:
            return img, img_name


def identity_transform(x):
    return x


train_dataset = PlantDataset(
    train_df, TRAIN_IMGS_PATH, transform=identity_transform, is_train=True
)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

valid_exts = (".jpg", ".jpeg", ".png")
test_files = [
    f for f in sorted(os.listdir(IMGS_PATH)) if f.lower().endswith(valid_exts)
]
test_df = pd.DataFrame(test_files, columns=["image"])
test_dataset = PlantDataset(
    test_df, IMGS_PATH, transform=identity_transform, is_train=False
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 4
backbone = torchvision.models.efficientnet_b4(pretrained=True).to(DEVICE).eval()


def extract_features(x):
    with torch.no_grad():
        x = backbone.features(x)
        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
    return x


feat_dim = backbone.classifier[1].in_features  # 1792 for B4



## === cell 5
sum_feats = torch.zeros((NUM_LABELS, feat_dim), device=DEVICE)
cnts = torch.zeros((NUM_LABELS, 1), device=DEVICE)

for imgs, lbls_batch in train_loader:
    imgs = imgs.to(DEVICE)
    lbls_batch = lbls_batch.to(DEVICE)  # shape (B, NUM_LABELS)
    feats = extract_features(imgs)  # (B, D)
    feats_norm = nn.functional.normalize(feats, dim=1)  # L2‑normed

    batch_size = imgs.size(0)
    for i in range(batch_size):
        f = feats_norm[i]  # already normalized
        label_mask = lbls_batch[i] == 1.0
        label_indices = label_mask.nonzero(as_tuple=False).view(-1)
        for l in label_indices.tolist():
            sum_feats[l] += f
            cnts[l] += 1

cnts = torch.clamp(cnts, min=1.0)
centroids = sum_feats / cnts  # (L, D)
centroids_norm = nn.functional.normalize(centroids, dim=1)  # unit‑norm



## === cell 6
pred_rows = []
for imgs, names in test_loader:
    imgs = imgs.to(DEVICE)

    feats = extract_features(imgs)
    feats_norm = nn.functional.normalize(feats, dim=1)

    imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
    feats_flipped = extract_features(imgs_flipped)
    feats_flipped_norm = nn.functional.normalize(feats_flipped, dim=1)

    sims_orig = torch.mm(feats_norm, centroids_norm.t())
    sims_flip = torch.mm(feats_flipped_norm, centroids_norm.t())
    sims = (sims_orig + sims_flip) / 2.0
    sims = sims.cpu().numpy()

    for img_name, sim_vec in zip(names, sims):
        label_idxs = np.where(sim_vec > TH)[0]
        if len(label_idxs) == 0:
            label_idxs = [int(np.argmax(sim_vec))]
        labels_str = " ".join([LABELS_[idx] for idx in label_idxs])
        pred_rows.append({"image": img_name, "labels": labels_str})

df_sub = pd.DataFrame(pred_rows)



## === cell 7
output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
elapsed = time.time() - start_time
print(f"Total elapsed time: {int(elapsed // 60)} min {int(elapsed % 60)} sec")
