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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
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
timm==1.0.19
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

17.02184905599554

# 6. Current score

23.31015

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I remove the `cuml` dependency that crashes due to an insufficient CUDA driver and rely on CPU `sklearn.svm.SVR` for loading/predicting with the already-trained SVR pickles. I fix the Albumentations namespace collision that makes `A` a `functools.partial` by explicitly re-importing albumentations inside `get_data`, ensuring `A.Compose` is valid. I also fix two inference bugs: using the passed `svr` object instead of an undefined `clf` inside `tta`, and ensuring tabular features are `float32` tensors. Finally, I clip predictions to the valid [1, 100] range before writing `submission.csv` to satisfy Kaggle’s submission constraints.'
- What this solution (achieved 42.24644) has done: 'I fix the two blocking runtime issues: the hardcoded `saved-weights` path (it doesn’t exist in this environment) and the `timm` “Unknown model” error (those `_in22k` model names aren’t registered in timm 1.0.19). To preserve the core inference logic, I (1) auto-detect the correct `model_weights_dir` location from the available `/kaggle/input/**` folders and (2) map the unsupported model names to the closest timm-registered equivalents while keeping the same ensemble flow and SVR blending. I also make `learn.load(...)` compatible with fastai by passing the filename stem (fastai expects no extension) and keep the existing prediction clipping and submission formatting unchanged. These changes are execution-unblocking and should substantially improve RMSE versus the current broken/degenerate run (42) by actually loading the intended fold weights and running the intended backbones end-to-end.'
- What this solution (achieved 23.31015) has done: 'I fix the execution-blocking import/order bugs by ensuring `os`, `random`, `torch`, `Dataset`, etc. are defined before first use (your very first line checks `os.path.isdir` before importing `os`). I keep the modeling/inference logic unchanged, but make the weights directory detection robust to the actual Kaggle folder layout and ensure fastai `Learner.load()` is given the correct filename stem. I also add a safe fallback that still produces a valid `submission.csv` (with correct columns and row count) if no matching `.pth`/`.pkl` weights are found in the environment, rather than crashing. These changes are primarily correctness/unblocking; if the pretrained weights are present, it should run the intended ensemble and produce a much better RMSE than any degenerate fallback.'
- What this solution (achieved 23.31015) has done: 'I make two minimal inference-only fixes that are directly tied to RMSE: (1) remove training-only random augmentations from the test transform so TTA actually measures different deterministic views (flip + slight resize/pad/center-crop only), reducing noisy prediction variance; and (2) stop forcing mixed precision at inference (`to_fp16()`), because fp16 can slightly degrade regression accuracy for this ensemble while not being needed for the small test set. Core model architecture, SVR blending, fold ensembling, and submission format remain unchanged. These changes should move your score down (better) toward the 17.02 target without altering the overall approach or adding extra training.'
- What this solution (achieved 23.31015) has done: 'Your current RMSE (23.31, lower is better) is still worse than the target (17.02), so we should cautiously improve inference without changing the ensemble’s core logic. The biggest low-risk gain here is to make TTA actually cover both flip states deterministically instead of relying on random flips with replacement; that reduces prediction variance and usually improves RMSE for this competition. I also make the DataLoader fully deterministic and aligned (no worker RNG drift) so the TTA loop is reproducible and not accidentally “noisy” across runs, while keeping the same models, folds, SVR blending, and clipping. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 23.31015) has done: 'I make two minimal inference-only adjustments that tend to improve RMSE on this competition without changing your ensemble/blending core logic: (1) use the same `DataLoaders` normalization/transform pipeline but ensure the fastai `DataLoaders` used by the `Learner` matches the exact test `DataLoader` (same dataset object) to avoid subtle transform/stat mismatches, and (2) load weights onto the right device explicitly and run inference under `torch.inference_mode()` to reduce any accidental autograd overhead/side effects. These are small, low-risk changes aimed at reducing prediction noise and moving your 23.31 RMSE closer to the 17.02 target. The model architectures, fold loop, TTA (no-flip + flip), SVR blending, and clipping remain unchanged. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 23.31015) has done: 'I make two inference-only fixes that keep your ensemble, SVR blending, and TTA logic the same, but reduce systematic prediction noise so RMSE moves down toward your 17.02 target. First, I set `persistent_workers=False` (and guard it) because Kaggle environments can recycle workers unpredictably across repeated `get_learner()` calls, which can subtly destabilize ordering/augmentation and hurt regression RMSE. Second, I ensure `test_csv` uses a clean 0..N-1 index before the Dataset reads by `iloc`, preventing any accidental misalignment if `test_csv` is modified in-place later (this is low-risk and protects submission row order correctness).'
- What this solution (achieved 23.31015) has done: 'Your current RMSE (23.31, lower is better) is still worse than the target (17.02), so the smallest safe way to improve is to reduce inference-time stochasticity and mismatches rather than changing the ensemble itself. I (1) make TTA truly deterministic by removing any RNG dependence in the DataLoader (no generator/workers) and enforcing a stable center-crop pipeline, and (2) ensure the tabular features are passed as contiguous float32 arrays (SVR + torch concat can be sensitive to dtype/stride), while keeping the exact same models, weights, folds, SVR blending, and clipping. I also add a small guard to always run the model under `torch.inference_mode()` and `eval()` (already mostly done) and keep submission ordering aligned to `test.csv`. These are minimal inference-only adjustments aimed at lowering RMSE without altering the core approach.'
- What this solution (achieved 23.31015) has done: 'I make two inference-only fixes that keep your model/ensemble/SVR blending logic identical but should reduce RMSE noise and move 23.31 closer to the 17.02 target. First, I ensure the tabular features are moved to the GPU as a contiguous float32 tensor (your current `torch.as_tensor(..., device=...)` can preserve non-contiguity and sometimes triggers extra copies/strides that subtly affect throughput and occasionally numerical stability). Second, I align the fastai `DataLoaders` to use the exact same underlying PyTorch `DataLoader` ordering and batching by building `DataLoaders` from the same dataset and `after_batch` normalization is already handled by Albumentations, so we avoid any hidden fastai transforms being applied differently across runs. These are minimal changes that should improve consistency and typically lower RMSE without changing architecture, weights, or blending.'
- What this solution (achieved 23.31015) has done: 'Your current RMSE (23.31, lower is better) is still above the 17.02 target, so the safest way to move toward the target without changing your ensemble architecture is to reduce systematic inference mismatches. I (1) ensure the timm backbone is configured to output features (not classification logits) by setting `num_classes=0`, matching your `Network` design intent and improving the embedding/regression head behavior, and (2) fix the loss/metric scaling mismatch in `petfinder_rmse` by removing the extra `100*` factor so it matches the actual Pawpularity scale used in inference. These are minimal, inference-consistent fixes that keep the same models, folds, SVR blending, and TTA, but should reduce error meaningfully toward your target. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 23.31015) has done: 'Your RMSE (23.31, lower is better) is still worse than the target (17.02), so the safest way to move toward the target without changing your ensemble architecture/training loop is to fix an inference-time mismatch: right now you use the raw embedding tensor `embed` (which includes the 12 tabular features concatenated) as input to the SVR, but the SVR was trained on the image embedding only, so this feature dimension mismatch silently hurts (or can even break) SVR predictions. I minimally change `tta()` to pass only the image embedding portion (`embed[:, :embed_dim]`) into the SVR while keeping the same blending weights and TTA averaging. I also add a tiny shape-guard so if an SVR expects a different feature count, we slice/pad safely rather than crashing, preserving end-to-end submission generation. No model architecture, loss, weights, folds, or transforms are changed beyond this feature alignment.'
- What this solution (achieved 23.31015) has done: 'I make one metric-aligned change that directly affects inference quality without touching your model architecture, folds, SVR blending, or transforms: remove the extra `* 100.0` scaling inside `petfinder_rmse`, which currently mismatches your model’s training objective vs the true Pawpularity scale and can lead to poorly-calibrated regressors/embeddings for the loaded weights. This is a minimal, semantics-preserving fix (your inference already outputs 1–100), and it should reduce RMSE from 23.31 toward your 17.02 target. Everything else (TTA no-flip/flip, SVR feature alignment, clipping, and submission writing) remains the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import random
import pickle
import glob

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from PIL import Image

import timm
from fastai.vision.all import *
from fastai.data.core import *

from sklearn.svm import SVR

import albumentations as A
from albumentations.pytorch import ToTensorV2

if os.path.isdir("../input/saved-weights/pytorch-image-models/pytorch-image-models"):
    sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
def petfinder_rmse(input, target):
    return torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()) * 100.0, target))




## === cell 3
base_dir = "/kaggle/input"


def _find_model_weights_dir(base_dir: str) -> str:
    """
    Find a directory containing the required .pth + .pkl artifacts.
    This unblocks execution across different Kaggle dataset mounting layouts.
    """
    explicit_candidates = [
        os.path.join(base_dir, "saved-weights"),
        os.path.join(base_dir, "petfinder-pawpularity-score", "saved-weights"),
        os.path.join(base_dir, "petfinder-pawpularity-score", "saved_weights"),
        os.path.join(base_dir, "petfinder-pawpularity-score", "saved_weights_full"),
    ]
    for c in explicit_candidates:
        if os.path.isdir(c):
            return c

    best = None
    best_score = -1
    for root, _, files in os.walk(base_dir):
        pth = [f for f in files if f.endswith(".pth")]
        pkl = [f for f in files if f.endswith(".pkl")]
        if pth and pkl:
            score = len(pth) + len(pkl)
            if score > best_score:
                best = root
                best_score = score

    if best is not None:
        return best

    return os.path.join(base_dir, "saved-weights")


def _find_petfinder_data_dir(base_dir: str) -> str:
    """
    Prefer the canonical competition folder if present; otherwise fall back to /kaggle/input.
    """
    candidates = [
        os.path.join(base_dir, "petfinder-pawpularity-score"),
        os.path.join(
            base_dir, "petfinder-pawpularity-score", "petfinder-pawpularity-score"
        ),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")) and os.path.isdir(
            os.path.join(c, "test")
        ):
            return c
    if os.path.exists(os.path.join(base_dir, "test.csv")) and os.path.isdir(
        os.path.join(base_dir, "test")
    ):
        return base_dir
    return os.path.join(base_dir, "petfinder-pawpularity-score")


model_weights_dir = _find_model_weights_dir(base_dir)
data_dir = _find_petfinder_data_dir(base_dir)

test_folder = os.path.join(data_dir, "test")
test_file = os.path.join(data_dir, "test.csv")



## === cell 4
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
embed_dim = 128
num_of_hidden = 2
batch_size = 64
hidden_dimension = [256, 64]
input_shape_224 = (224, 224, 3)
input_shape_384 = (384, 384, 3)
max_size_384 = 480
max_size_224 = 384

models_list = {
    "beit_base_patch16_224_in22k": 128,
    "swin_large_patch4_window12_384_in22k": 64,
    "swin_base_patch4_window7_224_in22k": 128,
    "swin_large_patch4_window7_224_in22k": 64,
}

save_name = model_weights_dir

weights_svr = {
    "beit_base_patch16_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window12_384_in22k": [0.1, 0.9],
    "swin_base_patch4_window7_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window7_224_in22k": [0.4, 0.6],
}

model_weights = {
    "beit_base_patch16_224_in22k": 0.2,
    "swin_large_patch4_window12_384_in22k": 0.4,
    "swin_base_patch4_window7_224_in22k": 0.2,
    "swin_large_patch4_window7_224_in22k": 0.2,
}



## === cell 5
assert os.path.exists(test_file), f"Missing test.csv at: {test_file}"
assert os.path.isdir(test_folder), f"Missing test folder at: {test_folder}"

print("Using data_dir:", data_dir)
print("Using test_file:", test_file)
print("Using test_folder:", test_folder)
print("Using model_weights_dir:", model_weights_dir)
print("Using device:", device)



## === cell 6
test_csv = pd.read_csv(test_file)

test_csv = test_csv.reset_index(drop=True)

test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)

missing_imgs = [
    p for p in test_csv["path_img"].head(50).tolist() if not os.path.exists(p)
]
if missing_imgs:
    raise FileNotFoundError(
        f"Some test images are missing. Example missing path: {missing_imgs[0]}"
    )

test_csv.head()




## === cell 7
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
        self.cat = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path_img"].iloc[idx]
        label_1 = self.df["Pawpularity"].iloc[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = np.ascontiguousarray(
            self.df[self.cat].iloc[idx].values, dtype=np.float32
        )

        return (img, df_data, label_1)




## === cell 8
def get_data(batch_size, max_size, input_shape, hflip_p: float):
    import albumentations as A_local
    from albumentations.pytorch import ToTensorV2 as ToTensorV2_local

    test_transform = A_local.Compose(
        [
            A_local.LongestMaxSize(max_size=max_size, interpolation=1),
            A_local.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            A_local.CenterCrop(height=input_shape[0], width=input_shape[1]),
            A_local.HorizontalFlip(p=hflip_p),
            A_local.Normalize(mean, std_dev),
            ToTensorV2_local(),
        ]
    )

    test_dataset = PetsDataset(test_csv, test_transform)

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    dls = DataLoaders.from_dsets(
        test_dataset, test_dataset, bs=batch_size, shuffle=False
    )
    return dls, testloader




## === cell 9
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
    ):
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise Exception(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)

        base.head = nn.Linear(base.head.in_features, embed_dim)
        nn.init.kaiming_normal_(base.head.weight)
        nn.init.constant_(base.head.bias, 0)

        self.p = 0.5
        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )
        self.network = self.__freeze_layer(base, freeze_layer)
        self.__initialise_weights()

    def __initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __freeze_layer(self, base, freeze_layer):
        cnt = 0
        for child in base.children():
            cnt += 1
            if cnt > freeze_layer:
                break
            for param in child.parameters():
                param.requires_grad = False
        return base

    def __fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        layers.append(nn.Dropout(self.p))
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != (number_of_hidden - 1):
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg, x




## === cell 10
TIMM_NAME_MAP = {
    "beit_base_patch16_224_in22k": "beit_base_patch16_224",
    "swin_large_patch4_window12_384_in22k": "swin_large_patch4_window12_384",
    "swin_base_patch4_window7_224_in22k": "swin_base_patch4_window7_224",
    "swin_large_patch4_window7_224_in22k": "swin_large_patch4_window7_224",
}


def _resolve_timm_name(name: str) -> str:
    resolved = TIMM_NAME_MAP.get(name, name)
    if not timm.is_model(resolved):
        if timm.is_model(name):
            return name
        raise RuntimeError(
            f"Neither model '{name}' nor mapped model '{resolved}' is available in timm."
        )
    return resolved


def get_learner(
    model_name,
    batch_size,
    loss,
    metric,
    max_size,
    input_size,
    save_path,
    hflip_p: float,
):
    dls, testloader = get_data(batch_size, max_size, input_size, hflip_p=hflip_p)
    dls = dls.to(device)

    timm_name = _resolve_timm_name(model_name)

    network = timm.create_model(timm_name, pretrained=False, num_classes=0)

    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    model = model.to(device)

    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    return learn, testloader




## === cell 11
def _svr_align_features(X: np.ndarray, svr) -> np.ndarray:
    """
    Score-improving fix (minimal): the SVR was trained on image embedding features only.
    Our model's returned `embed` includes [image_embed, tabular] concatenated, so we must
    feed only the image_embed part to the SVR to match training and reduce RMSE.
    Also add a small guard to slice/pad if a specific SVR fold expects a different dim.
    """
    X_img = X[:, :embed_dim]

    expected = getattr(svr, "n_features_in_", None)
    if expected is None:
        return X_img

    if X_img.shape[1] == expected:
        return X_img
    if X_img.shape[1] > expected:
        return X_img[:, :expected]

    pad = np.zeros((X_img.shape[0], expected - X_img.shape[1]), dtype=X_img.dtype)
    return np.concatenate([X_img, pad], axis=1)


def tta(learn, svr, svr_weight, testloader_no_flip, testloader_flip):
    tta_outputs = []
    learn.model.eval()

    for testloader in (testloader_no_flip, testloader_flip):
        final_outputs = []
        svr_data = []

        with torch.inference_mode():
            for images, tabular, _ in testloader:
                images = images.to(device, non_blocking=True)

                tabular = torch.from_numpy(
                    np.ascontiguousarray(tabular, dtype=np.float32)
                ).to(device=device, non_blocking=True)

                reg_output, embed = learn.model(images, tabular)
                reg_output = 100 * torch.sigmoid(reg_output)

                output = reg_output.detach().to("cpu").numpy().reshape(-1).tolist()
                final_outputs.extend(output)

                embed_np = embed.detach().cpu().numpy().astype(np.float32, copy=False)
                svr_data.extend(embed_np.tolist())

        final_outputs = np.array(final_outputs, dtype=np.float32)
        svr_data = np.array(svr_data, dtype=np.float32)

        svr_X = _svr_align_features(svr_data, svr)
        svr_preds = svr.predict(svr_X).astype(np.float32)

        blended = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs
        tta_outputs.append(blended)

    tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)
    return tta_outputs_arr




## === cell 12
all_pth = glob.glob(os.path.join(model_weights_dir, "*.pth"))
all_pkl = glob.glob(os.path.join(model_weights_dir, "*.pkl"))
weights_available = (len(all_pth) > 0) and (len(all_pkl) > 0)

print(f"Found {len(all_pth)} .pth and {len(all_pkl)} .pkl in model_weights_dir.")
print("Weights available:", weights_available)

final_predictions = []

if weights_available:
    for model_name, bs in models_list.items():
        m_name = model_name[: model_name.find("_", 5)]

        fold_tta = []
        svr_weights = weights_svr[model_name]
        w = model_weights[model_name]
        m_384 = False
        if model_name.find("384") != -1:
            m_384 = True
            m_name = m_name + "_384"

        print("#################################")
        print("Testing Model: {}".format(m_name))
        print("#################################")

        for fold in range(N_FOLDS):
            print("Fold: {}".format(fold))
            saved_name = m_name + "_fold_{}_full.pth".format(fold)
            svr_name = m_name + "_svr_fold_{}.pkl".format(fold)

            if m_384:
                learn_nf, testloader_nf = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_384,
                    input_shape_384,
                    save_name,
                    hflip_p=0.0,
                )
                _, testloader_f = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_384,
                    input_shape_384,
                    save_name,
                    hflip_p=1.0,
                )
            else:
                learn_nf, testloader_nf = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_224,
                    input_shape_224,
                    save_name,
                    hflip_p=0.0,
                )
                _, testloader_f = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_224,
                    input_shape_224,
                    save_name,
                    hflip_p=1.0,
                )

            weights_path = os.path.join(model_weights_dir, saved_name)
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights file: {weights_path}")

            learn_nf.load(saved_name.replace(".pth", ""), with_opt=False, device=device)

            svr_path = os.path.join(model_weights_dir, svr_name)
            if not os.path.exists(svr_path):
                raise FileNotFoundError(f"Missing SVR pickle: {svr_path}")
            svr_model = pickle.load(open(svr_path, "rb"))

            fold_tta.append(
                tta(learn_nf, svr_model, svr_weights, testloader_nf, testloader_f)
            )

            del learn_nf, svr_model, testloader_nf, testloader_f
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        fold_tta = np.array(fold_tta)
        fold_tta = np.mean(fold_tta, axis=0)
        final_predictions.append(w * fold_tta)
else:
    final_predictions = [
        np.full(shape=(len(test_csv),), fill_value=50.0, dtype=np.float32)
    ]



## === cell 13
final_predictions = np.array(final_predictions)



## === cell 14
final_predictions = np.sum(final_predictions, axis=0)



## === cell 15
final_predictions = np.clip(final_predictions, 1.0, 100.0)
test_csv["Pawpularity"] = final_predictions



## === cell 16
test_csv = test_csv[["Id", "Pawpularity"]]



## === cell 17
test_csv.head()



## === cell 18
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(test_csv["Pawpularity"].describe())
