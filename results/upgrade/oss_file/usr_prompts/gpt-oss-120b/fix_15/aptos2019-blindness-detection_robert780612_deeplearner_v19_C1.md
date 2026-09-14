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

0.910943132658642

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix removes the non‑code description that caused a syntax error, deletes stray markdown backticks, and restructures the notebook into proper Python cells so it runs from start to finish. No core model logic is changed; only the parsing/IO issues are corrected, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.70391) has done: 'The update batches all feature extraction with a DataLoader, avoids per‑image forward passes, and computes nearest‑neighbor distances for the whole test set in one NumPy operation. This cuts the dominant loop from thousands of separate GPU calls to a few batched calls while keeping the exact model, preprocessing, loss, and K‑NN logic unchanged.'
- What this solution (achieved 0.71234) has done: 'The update adds a safe fallback when loading the ImageNet‑pretrained SE‑ResNet‑50 model: if the external download fails (as it does in the restricted Kaggle environment), the code now falls back to initializing the model with random weights and proceeds to load any local fine‑tuned weights. This prevents the HTTP 503 error and ensures a `submission.csv` is produced, while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved 0.67425) has done: 'I keep the overall architecture and inference pipeline unchanged but replace the simple mean of the k‑nearest neighbor labels with a majority‑vote (mode) that better matches the discrete diagnosis classes. Using a mode often yields more accurate categorical predictions than averaging and rounding, which should move the Quadratic Weighted Kappa toward the target value without altering the model or preprocessing steps.'
- What this solution (achieved 0.75772) has done: 'I keep the overall architecture and feature extraction unchanged, but replace the simple majority‑vote aggregation with a distance‑weighted average of the 5 nearest neighbour labels (rounded to the nearest integer and clipped to 0‑4). Using the actual distances gives more nuanced predictions and is expected to raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

__all__ = [
    "SENet",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
]

pretrained_settings = {
    "senet154": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet50": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
}





## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    """Create SE‑ResNet‑50 with GeM pooling and a single‑output head."""
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)  # regression head for diagnosis score
    return model




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3703926879.py in <cell line: 0>()
----> 1 class GeM(nn.Module):
      2     def __init__(self, p=3, eps=1e-6):
      3         super(GeM, self).__init__()
      4         self.p = Parameter(torch.ones(1) * p)
      5         self.eps = eps

NameError: name 'nn' is not defined

## === cell 2
MODEL_PATH = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    model = get_se_resnet50_gem(pretrain="imagenet")
except Exception as e:
    print(
        f"Warning: could not load ImageNet pretrained weights ({e}); using randomly initialized model."
    )
    model = get_se_resnet50_gem(pretrain=None)

model.to(device)

try:
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print("Fine‑tuned weights loaded successfully.")
except Exception as e:
    print(
        f"Warning: could not load fine‑tuned weights ({e}); continuing with base model."
    )

model.eval()

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class ImageFeatureDataset(data.Dataset):
    """Dataset that returns transformed image tensor and its id."""

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256), resample=Image.BILINEAR)
        return self.transform(img), img_id


def compute_features(loader):
    """Extract GeM‑pooled features for all images in the loader."""
    feats_list = []
    ids = []
    with torch.no_grad():
        for imgs, batch_ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = model.features(imgs)
            pooled = model.avg_pool(feats)
            pooled = pooled.view(pooled.size(0), -1)
            feats_list.append(pooled.cpu().numpy())
            ids.extend(batch_ids)
    return np.vstack(feats_list), ids


print("Preparing training features for KNN inference...")
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_dataset = ImageFeatureDataset(train_df, TRAIN_IMAGE_PATH, norm)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

train_features, train_ids = compute_features(train_loader)

print("Running inference on test set...")
test_df = pd.read_csv(TEST_CSV_PATH)

test_dataset = ImageFeatureDataset(test_df, TEST_IMAGE_PATH, norm)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

test_features, test_ids = compute_features(test_loader)

k = 20  # increased number of neighbours for smoother regression

train_norm_sq = np.sum(train_features**2, axis=1)
test_norm_sq = np.sum(test_features**2, axis=1)[:, None]

dist_sq = test_norm_sq + train_norm_sq - 2.0 * np.dot(test_features, train_features.T)
dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety

nearest_k_idx = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]

neighbor_labels = train_df["diagnosis"].values[nearest_k_idx].astype(np.float32)
neighbor_dist = np.take_along_axis(dist_sq, nearest_k_idx, axis=1)

eps = 1e-6
sigma = np.mean(neighbor_dist) + eps
weights = np.exp(-neighbor_dist / sigma)

weighted_avg = np.sum(weights * neighbor_labels, axis=1) / np.sum(weights, axis=1)

preds = np.rint(weighted_avg).astype(int)
preds = np.clip(preds, 0, 4)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3691507286.py in <cell line: 0>()
      5 TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
      6 
----> 7 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      8 ImageFile.LOAD_TRUNCATED_IMAGES = True
      9 

NameError: name 'torch' is not defined
