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

3.9

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

0.8375642187972197

# 6. Current score

0.49963

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the dataset/IO bugs that prevent the notebook from running: the test image folder contains a nested `test_images/` directory that must be filtered out, and `MyTestDataset.__len__` incorrectly references `self.labels`. I also fix the missing model weights path by using torchvision’s built-in pretrained SqueezeNet weights (keeping the same model architecture and inference logic) so the code can run end-to-end in this environment. Finally, I ensure inference runs in `eval()` mode with `torch.no_grad()` and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.53587) has done: 'Your 0.0 score is because the current code uses an ImageNet-pretrained SqueezeNet (1000 classes) to predict cassava labels (0–4), so predictions are essentially random and won’t approach the target accuracy. To move the score toward 0.8376 with minimal changes while preserving the core approach (single pretrained CNN, no training loop), I keep SqueezeNet inference but add a lightweight “nearest-prototype” classifier: compute class prototypes from the provided `train_images/train.csv` using the same pretrained model, then classify each test image by nearest prototype in feature space. This keeps the model architecture and loss/training logic unchanged (still no training), but makes predictions aligned to the competition’s 5 classes. I also switch inference to use a DataLoader for correctness/performance and keep the submission format identical.'
- What this solution (achieved 0.53401) has done: 'To move your score up toward 0.8376 without changing the core “pretrained SqueezeNet + nearest-prototype (no training loop)” approach, I (1) build prototypes from a more representative subset of the training data by sampling uniformly per class instead of taking the first N rows (which can be order-biased), and (2) compute prototypes more efficiently by accumulating per-class sums without an inner loop. I also (3) use a slightly larger but still time-safe cap and (4) add lightweight test-time augmentation (simple horizontal flip) and average the similarities, which keeps the same inference semantics but typically improves accuracy for image classification. These are minimal, legitimate changes aimed at improving feature-prototype quality and stability, which should increase accuracy toward the target. The script still writes a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.49402) has done: 'Your current gap to the target is large (0.534 → 0.837), so we need a real accuracy lift while keeping the same core “pretrained SqueezeNet feature extractor + nearest-prototype, no training loop” approach. The biggest issue is that using only the mean prototype per class is too coarse; a minimal extension that preserves the same logic is to use multiple prototypes per class (k-means centroids) and classify by nearest centroid, which better represents intra-class variation. I also make the prototype computation use both original and hflip features (same TTA you already use at test time) so train/test feature distributions match better. Finally, I keep runtime under the 600s limit by capping the total prototype-build images and using a lightweight, torch-based k-means on the per-image feature vectors.'
- What this solution (achieved 0.53587) has done: 'Your current score (0.494) is far below the target (0.8376), so we should improve accuracy while keeping the same core “pretrained SqueezeNet features + (multi-)prototype nearest-centroid, no training loop” approach. The biggest likely issue is that the current k-means implementation is effectively “random init + cosine assignment” without k-means++ and can yield weak centroids; I switch to a minimal k-means++ init and a vectorized centroid update to get better, stabler prototypes without changing the overall method. I also compute centroids on CPU (as you do) but run assignment/update more efficiently, allowing a slightly higher K_PER_CLASS within the same time budget. Finally, I add a very small, metric-aligned calibration tweak: classify by the best centroid similarity but break ties by class-mean similarity (still prototype-based), which typically reduces noisy centroid effects without changing the evaluation semantics.'
- What this solution (achieved 0.52093) has done: 'Your current score (0.5359) is far below the target (0.8376), so we should improve accuracy while keeping the same core approach: SqueezeNet feature extractor + (multi-)prototype nearest-centroid with no training loop. The most impactful minimal change is to use a stronger pretrained feature extractor while preserving the same pipeline semantics; swapping SqueezeNet1_0 to SqueezeNet1_1 keeps the same architecture family and identical inference/prototype logic, but typically yields better ImageNet features. I also increase the per-class prototype data cap slightly and bump K_PER_CLASS modestly, which improves centroid coverage without changing the method and should move accuracy upward toward the target. Finally, I keep submission alignment stable by ordering predictions to exactly match `sample_submission.csv` image order.'
- What this solution (achieved 0.5071) has done: 'Your current score (0.52093) is far below the target (0.83756), so we should improve accuracy while keeping the same “pretrained CNN feature extractor + multi-prototype nearest-centroid (k-means) + light TTA, no training loop” core logic. The biggest minimal gain here is to make the similarity computation consistent with your k-means++ seeding objective: you seed centroids using cosine-style “1 - best_sim”, but you cluster using raw dot products; switching the assignment/update to true cosine distance (i.e., use L2 distance on normalized vectors) makes clustering/prototypes noticeably more meaningful. Second, we increase centroid coverage slightly (K_PER_CLASS) and k-means iterations a bit while staying within the time budget; this should move accuracy upward without changing the overall method. Finally, we keep your submission alignment logic intact and only adjust the prototype-building math and a small hyperparameter bump.'
- What this solution (achieved 0.50897) has done: 'I keep your same pretrained SqueezeNet1_1 feature extractor + multi-prototype (k-means) nearest-centroid pipeline, but fix a score-relevant inconsistency: your k-means description says cosine-consistent, yet the assignment step is still raw dot-product argmax; I switch it to true cosine/L2-on-normalized distance (equivalently, maximize cosine similarity but computed consistently) and add a tiny centroid-temperature smoothing when combining centroid score with class-mean score to reduce noisy centroid wins. I also make the train/test feature extraction path identical by using the same `model.eval()` and ensuring centroids/class_means are computed in float32 consistently (avoids subtle dtype drift). These are minimal changes that preserve your core logic (no training loop, same model, same feature pooling, same k-means + TTA + submission merge) and are aimed at improving accuracy from ~0.51 toward the 0.84 target.'
- What this solution (achieved 0.49851) has done: 'Your current score (0.50897) is far below the target (0.83756), so we should push accuracy upward with minimal, score-relevant tweaks while keeping the same core pipeline: pretrained SqueezeNet feature extractor + multi-prototype k-means + cosine nearest-centroid + light hflip TTA and the same submission logic. The biggest low-risk improvement is to fix the k-means++ seeding to be truly cosine-consistent (use squared L2 distance on normalized vectors, i.e., `2-2*cos`, instead of `1-cos`) and to update centroids using a stable “mean then normalize” step that matches the assignment geometry. Additionally, make the small “centroid boost” combine step class-aware (add the boost to all centroids of that predicted class via class-max centroid similarity), which reduces noise from a single centroid win without changing the method. These are small changes aimed at making prototypes and the final decision rule more consistent, which typically increases accuracy toward your target.'
- What this solution (achieved 0.49813) has done: 'Your current pipeline is already producing a valid submission, but the score is far below target, so the smallest likely lift (without changing the core “pretrained SqueezeNet feature extractor + k-means multi-prototypes + cosine retrieval + hflip TTA” approach) is to make the train prototypes more representative and slightly improve the decision fusion. I (1) rebalance the prototype-building sample to use **all** training images when feasible (it is ~18.7k total, which is still time-safe here) instead of capped per-class sampling, (2) increase centroid coverage slightly and make k-means a bit more stable, and (3) replace the fixed fusion weight `alpha` with a very small data-driven calibration computed from the training features (still no training loop; just choosing a scalar that best matches labels on the prototype-build set). These changes keep the same model, same feature extraction, same nearest-centroid/class-mean logic, and same submission semantics, but should move accuracy upward toward the target.'
- What this solution (achieved 0.49813) has done: 'Your score is far below the target, so we should push accuracy upward with the smallest changes that keep your core pipeline intact (pretrained SqueezeNet1_1 feature extractor → k-means multi-centroids → cosine retrieval + class-mean fusion + hflip TTA). The biggest low-risk issue is that the fusion/alpha is tuned and applied on the *same* features used to build centroids, which tends to overfit and can hurt generalization; I switch to a simple stratified holdout split for alpha selection only (no training loop added). I also make k-means slightly more stable by using a couple more iterations and using deterministic class-balanced shuffling for the prototype build order, without changing the model or inference semantics. Finally, I keep the submission alignment exactly matching `sample_submission.csv`.'
- What this solution (achieved 0.49813) has done: 'Your current score is far below the target (need to increase accuracy), and the most likely score-limiting issue in the current pipeline is that the train features are “normalized after averaging” instead of “average of normalized views,” which can distort class structure for k-means and nearest-centroid matching. I make feature aggregation consistent by averaging the two already-normalized view features and then normalizing once (same for train and test), which is a minimal semantic correction within your existing SqueezeNet + multi-prototype + cosine retrieval + hflip TTA approach. I also make k-means slightly less noisy (without changing the method) by increasing iterations a bit and using a tiny epsilon in centroid normalization to avoid rare NaNs/degeneracy that can hurt accuracy. The submission writing and sample_submission alignment remain unchanged.'
- What this solution (achieved 0.49963) has done: 'Your current pipeline is underperforming largely because the “alpha fusion” is selected on a holdout but the centroids/class-means are still built using *all* images (including the holdout), which leaks information into the alpha-selection objective and can pick a harmful alpha that generalizes poorly. I make a minimal, score-relevant change: build centroids and class means using only the stratified “build” split, then tune alpha on the disjoint “holdout” split (same features, same k-means, same decision rule). I also keep everything deterministic with the existing seeds and preserve the same submission alignment to `sample_submission.csv`. This keeps your core logic identical (pretrained SqueezeNet1_1 features → k-means multi-centroids → cosine similarity + hflip TTA + class-mean fusion), but should improve accuracy toward the target by making calibration consistent and less overfit.'
- What this solution (achieved 0.49963) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that keep your core pipeline intact (pretrained SqueezeNet1_1 feature extractor + k-means multi-centroids + cosine retrieval + hflip TTA + class-mean fusion, no training loop). The main score-limiting issue is that your k-means update averages unit vectors and then normalizes, which is fine, but it does not reweight for within-cluster similarity; a very small, metric-aligned improvement is to compute **spherical k-means** updates (weights proportional to cosine similarity to the centroid) while keeping the same assignment rule and cosine geometry. I also slightly widen the alpha grid and temperature grid on the holdout (still just calibration, not training) to reduce under/over-weighting of centroid evidence, which should move accuracy upward toward the target. All I/O paths and the submission alignment to `sample_submission.csv` remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images/"

files = sorted(
    [f for f in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f))]
)

assert len(files) > 0, (
    f"No test image files found in {TEST_PATH}. "
    f"Found entries: {os.listdir(TEST_PATH)[:10]}"
)



## === cell 2
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)

preprocess_hflip = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]
)


def default_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(img_pil)
    return img_tensor


def hflip_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess_hflip(img_pil)
    return img_tensor




## === cell 3
class MyTestDataset(Dataset):
    def __init__(self, files, loader=default_loader):
        self.files = files
        self.loader = loader

    def __getitem__(self, index):
        img_path = os.path.join(TEST_PATH, self.files[index])
        img = self.loader(img_path)
        return self.files[index], img

    def __len__(self):
        return len(self.files)




## === cell 4
testset = MyTestDataset(files)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
model = torchvision.models.squeezenet1_1(weights=weights)
model = model.to(device)
model.eval()

feature_extractor = model.features
feature_extractor.eval()



## === cell 6
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images/"

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)


class MyTrainDataset(Dataset):
    def __init__(self, df, img_dir, loader=default_loader):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.loader = loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        x = self.loader(img_path)
        y = int(row["label"])
        return x, y


num_classes = 5


def stratified_split_df(df, label_col="label", val_frac=0.20, seed=0):
    g = np.random.RandomState(seed)
    val_parts = []
    build_parts = []
    for c, gdf in df.groupby(label_col):
        idx = np.arange(len(gdf))
        g.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(gdf))))
        val_parts.append(gdf.iloc[idx[:n_val]])
        build_parts.append(gdf.iloc[idx[n_val:]])
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    build_df = (
        pd.concat(build_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return build_df, val_df


train_df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
build_df, holdout_df = stratified_split_df(train_df, val_frac=0.20, seed=0)

print("Total:", len(train_df), "Build:", len(build_df), "Holdout:", len(holdout_df))
print("Build per-class:", build_df["label"].value_counts().sort_index().to_dict())
print("Holdout per-class:", holdout_df["label"].value_counts().sort_index().to_dict())

buildset = MyTrainDataset(build_df, TRAIN_IMG_DIR)
buildloader = DataLoader(
    buildset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
buildset_hflip = MyTrainDataset(build_df, TRAIN_IMG_DIR, loader=hflip_loader)
buildloader_hflip = DataLoader(
    buildset_hflip,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

holdoutset = MyTrainDataset(holdout_df, TRAIN_IMG_DIR)
holdoutloader = DataLoader(
    holdoutset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
holdoutset_hflip = MyTrainDataset(holdout_df, TRAIN_IMG_DIR, loader=hflip_loader)
holdoutloader_hflip = DataLoader(
    holdoutset_hflip,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

feat_dim = 512  # SqueezeNet final conv channels


def extract_feats_labels(loader1, loader2):
    feats = []
    labels = []
    with torch.no_grad():
        for (xb, yb), (xb2, yb2) in zip(loader1, loader2):
            assert torch.equal(yb, yb2)
            xb = xb.to(device, non_blocking=True)
            xb2 = xb2.to(device, non_blocking=True)

            f1 = feature_extractor(xb).mean(dim=(2, 3))
            f1 = torch.nn.functional.normalize(f1, dim=1)

            f2 = feature_extractor(xb2).mean(dim=(2, 3))
            f2 = torch.nn.functional.normalize(f2, dim=1)

            f = (f1 + f2) * 0.5
            f = torch.nn.functional.normalize(f, dim=1)

            feats.append(f.detach().float().cpu())
            labels.append(yb.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0).long()
    return feats, labels


build_feats, build_labels = extract_feats_labels(buildloader, buildloader_hflip)
holdout_feats, holdout_labels = extract_feats_labels(holdoutloader, holdoutloader_hflip)

print("Collected BUILD features:", build_feats.shape, "labels:", build_labels.shape)
print(
    "BUILD per-class counts:",
    [(build_labels == c).sum().item() for c in range(num_classes)],
)
print(
    "Collected HOLDOUT features:", holdout_feats.shape, "labels:", holdout_labels.shape
)
print(
    "HOLDOUT per-class counts:",
    [(holdout_labels == c).sum().item() for c in range(num_classes)],
)


def kmeans_torch_spherical(x, k, iters=30, seed=0, weighted_update=True):
    """
    Minimal improvement to prototype quality while preserving core logic:
    cosine assignment + centroid update on normalized vectors, using spherical k-means.
    weighted_update=True performs a small, cosine-consistent reweighting inside clusters.
    x: [N, D] float32 CPU tensor (assumed L2-normalized)
    returns centroids: [k, D] normalized
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    n, d = x.shape
    if n == 0:
        return torch.empty((0, d), dtype=x.dtype)
    if n <= k:
        c = torch.nn.functional.normalize(x.clone(), dim=1, eps=1e-12)
        return c

    first = int(torch.randint(0, n, (1,), generator=g).item())
    centroids = [x[first].clone()]

    best_sim = (x @ centroids[0].unsqueeze(1)).squeeze(1)  # [N]
    for _ in range(1, k):
        dist2_l2 = (2.0 - 2.0 * best_sim).clamp_min(0.0)
        probs = dist2_l2 / (dist2_l2.sum() + 1e-12)
        idx = int(torch.multinomial(probs, 1, generator=g).item())
        centroids.append(x[idx].clone())
        best_sim = torch.maximum(best_sim, (x @ centroids[-1].unsqueeze(1)).squeeze(1))

    centroids = torch.stack(centroids, dim=0)  # [k, D]
    centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-12)

    for _ in range(iters):
        sims = x @ centroids.T  # [N, k]
        assign = torch.argmax(sims, dim=1)  # [N]

        new_centroids = torch.zeros((k, d), dtype=x.dtype)

        if weighted_update:
            w = sims.gather(1, assign.view(-1, 1)).squeeze(1).clamp_min(0.0)  # [N]
            xw = x * w.unsqueeze(1)
            new_centroids.index_add_(0, assign, xw)
            counts = torch.zeros((k,), dtype=x.dtype)
            counts.index_add_(0, assign, w)
            empty = counts <= 1e-12
            if empty.any():
                num_empty = int(empty.sum().item())
                ridx = torch.randint(0, n, (num_empty,), generator=g)
                new_centroids[empty] = x[ridx]
                counts[empty] = 1.0
            new_centroids = new_centroids / counts.unsqueeze(1).clamp_min(1e-12)
        else:
            new_centroids.index_add_(0, assign, x)
            counts = torch.bincount(assign, minlength=k).to(x.dtype)  # [k]
            empty = counts == 0
            if empty.any():
                num_empty = int(empty.sum().item())
                ridx = torch.randint(0, n, (num_empty,), generator=g)
                new_centroids[empty] = x[ridx]
                counts[empty] = 1.0
            new_centroids = new_centroids / counts.unsqueeze(1).clamp_min(1e-12)

        centroids = torch.nn.functional.normalize(new_centroids, dim=1, eps=1e-12)

    return centroids


K_PER_CLASS = 32

centroids_list = []
centroids_labels = []
for c in range(num_classes):
    x_c = build_feats[build_labels == c].float()
    x_c = torch.nn.functional.normalize(x_c, dim=1, eps=1e-12)

    cents = kmeans_torch_spherical(
        x_c, k=K_PER_CLASS, iters=60, seed=0 + c, weighted_update=True
    )

    centroids_list.append(cents)
    centroids_labels.append(torch.full((cents.shape[0],), c, dtype=torch.long))

centroids = torch.cat(centroids_list, dim=0)  # [M, 512]
centroids_y = torch.cat(centroids_labels, dim=0)  # [M]
centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-12)

print("Built centroids:", centroids.shape, "Centroid labels:", centroids_y.shape)

class_means = []
for c in range(num_classes):
    x_c = build_feats[build_labels == c].float()
    x_c = torch.nn.functional.normalize(x_c, dim=1, eps=1e-12)
    mu = x_c.mean(dim=0, keepdim=True)
    mu = torch.nn.functional.normalize(mu, dim=1, eps=1e-12)
    class_means.append(mu)
class_means = torch.cat(class_means, dim=0)  # [C, D]


def pick_alpha_temp_on_holdout(
    holdout_feats_cpu,
    holdout_labels_cpu,
    centroids_cpu,
    centroids_y_cpu,
    class_means_cpu,
):
    xh = torch.nn.functional.normalize(holdout_feats_cpu.float(), dim=1, eps=1e-12)
    yh = holdout_labels_cpu.long()

    cents = torch.nn.functional.normalize(centroids_cpu.float(), dim=1, eps=1e-12)
    cy = centroids_y_cpu.long()
    means = torch.nn.functional.normalize(class_means_cpu.float(), dim=1, eps=1e-12)

    sims = xh @ cents.T  # [Nh,M]
    mean_sims = xh @ means.T  # [Nh,C]

    per_class_cent_best = torch.full((xh.shape[0], num_classes), -1e9, dtype=xh.dtype)
    per_class_cent_best.scatter_reduce_(
        1,
        cy.view(1, -1).expand(xh.shape[0], -1),
        sims,
        reduce="amax",
        include_self=True,
    )

    alpha_grid = [0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25]
    temp_grid = [0.15, 0.20, 0.25]  # keep close to original 0.20

    best_a = 0.10
    best_t = 0.20
    best_acc = -1.0

    for t in temp_grid:
        cent_term = torch.tanh(per_class_cent_best / float(t))
        for a in alpha_grid:
            pred = torch.argmax(mean_sims + float(a) * cent_term, dim=1)
            acc = (pred == yh).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_a = float(a)
                best_t = float(t)

    print(
        f"Picked alpha={best_a:.2f}, temp={best_t:.2f} by HOLDOUT accuracy={best_acc:.4f} "
        f"over alpha_grid={alpha_grid}, temp_grid={temp_grid}"
    )
    return best_a, best_t


alpha, temp = pick_alpha_temp_on_holdout(
    holdout_feats, holdout_labels, centroids, centroids_y, class_means
)

centroids = centroids.to(device)
centroids_y = centroids_y.to(device)
class_means = class_means.to(device)



## === cell 7
testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

testset_hflip = MyTestDataset(files, loader=hflip_loader)
testloader_hflip = DataLoader(
    testset_hflip,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred_labels = []
pred_files = []

with torch.no_grad():
    for (file_batch, xb), (file_batch2, xb2) in zip(testloader, testloader_hflip):
        assert list(file_batch) == list(file_batch2)

        xb = xb.to(device, non_blocking=True)
        xb2 = xb2.to(device, non_blocking=True)

        feats1 = feature_extractor(xb).mean(dim=(2, 3))
        feats1 = torch.nn.functional.normalize(feats1, dim=1)

        feats2 = feature_extractor(xb2).mean(dim=(2, 3))
        feats2 = torch.nn.functional.normalize(feats2, dim=1)

        feats = (feats1 + feats2) * 0.5
        feats = torch.nn.functional.normalize(feats, dim=1)

        sims = feats @ centroids.T  # [B, M]
        mean_sims = feats @ class_means.T  # [B, C]

        per_class_cent_best = torch.full(
            (feats.shape[0], num_classes), -1e9, device=device, dtype=feats.dtype
        )
        per_class_cent_best.scatter_reduce_(
            1,
            centroids_y.view(1, -1).expand(feats.shape[0], -1),
            sims,
            reduce="amax",
            include_self=True,
        )

        combined = mean_sims + float(alpha) * torch.tanh(
            per_class_cent_best / float(temp)
        )
        yhat = torch.argmax(combined, dim=1).detach().cpu().numpy().astype(int).tolist()

        pred_labels.extend(yhat)
        pred_files.extend(list(file_batch))

submission = pd.DataFrame({"image_id": pred_files, "label": pred_labels})



## === cell 8
SAMPLE_SUB = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(SAMPLE_SUB)
submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notna().all(), "Some test images were not predicted."
submission["label"] = submission["label"].astype(int)

submission



## === cell 9
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
