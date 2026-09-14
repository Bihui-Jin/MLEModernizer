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
model-signing==1.1.1
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
sigstore-models==0.0.5
sklearn-pandas==2.2.0
statsmodels==0.14.5
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

17.30894069590541

# 6. Current score

31.61105

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I (1) remove the failing `cuml` dependency by switching SVR loading/prediction to CPU-safe `sklearn` (the pickled SVR models can be loaded with `pickle` and used without cuML), (2) fix the Albumentations namespace collision causing `A.Compose` to break, and (3) correct a couple of inference-time bugs (wrong variable `clf` vs `svr`, and ensure tabular features are float tensors). Finally, I clip predictions into the required \[1, 100\] range right before writing `submission.csv` to satisfy Kaggle’s validation without changing the core modeling logic.'
- What this solution (achieved 42.24644) has done: 'I fix the import/runtime failure by removing the hard dependency on the missing `models.convnext` module and instead using `timm`’s built-in ConvNeXt models when `conv_next=True`. This keeps the architecture intent (ConvNeXt backbone + tabular concat + same head) while making the code run reliably in the Kaggle environment. I also make `Learner.load()` use the correct `name` relative to `model_dir` (fastai expects a stem, not an absolute `.pth` path), and I keep the rest of the pipeline (TTA, SVR blending, clipping, submission formatting) unchanged to improve RMSE vs the current broken run. These changes are directly tied to execution correctness and should also restore the intended ensemble behavior, moving the score down toward the target.'
- What this solution (achieved 42.24644) has done: 'The crash is from looking for weights in a non-existent `/kaggle/input/saved-weights` dataset; your environment only shows weights under the competition dataset folder. I fix the weight/SVR path resolution to search the actual available locations and raise a clear error if still missing, without changing the model/ensemble logic. I also make the fastai `DataLoaders` compatible with your custom dataset by providing `after_batch` normalization and ensuring `DataLoaders.from_dsets` uses the same `device`, which prevents silent device mismatches during inference. These changes are execution-critical and should restore the intended 5-fold ensemble + SVR blend, which should improve RMSE toward the target versus the current broken run.'
- What this solution (achieved 42.24644) has done: 'I fix the runtime failure by making the weights/SVR file discovery robust to the actual Kaggle input layout: instead of assuming a single `saved-weights/` directory, the code search recursively under `/kaggle/input` for the required `.pth`/`.pkl` filenames. I also add a clear preflight check that lists which required files are missing before starting inference, so the notebook fails fast with actionable information rather than mid-loop. These changes are execution-critical and score-neutral (they don’t alter the model, TTA, blending, or post-processing), and they ensure the pipeline completes and writes a valid `submission.csv`. No training/evaluation semantics are changed.'
- What this solution (achieved 42.24644) has done: 'The run is failing because the script expects external fold `.pth` and `.pkl` SVR files that are not present in your attached Kaggle inputs, so it crashes before producing `submission.csv`. I keep the same model/TTA/SVR-blend core logic when those files exist, but add a minimal “no-weights fallback” path that still runs end-to-end: it build the same ConvNeXt-based network untrained and output a valid submission (score be worse, but it unblocks execution). I also fix one critical metric bug: your `petfinder_rmse` currently compares a sigmoid’d value in `[0,1]` against a target in `[0,100]`, which is inconsistent with how you scale predictions elsewhere; correcting it is score-positive when you do have weights. Finally, I ensure the submission columns and file suffix are correct and always written.'
- What this solution (achieved 32.62858) has done: 'I fix the ConvNeXt forward shape error by making the backbone return the pooled feature vector (not a flattened spatial map) so the replaced head produces the expected `embed_dim` output. I also make the DataLoaders/testloader use deterministic, inference-safe transforms (no random augmentation) so TTA works as intended and improves RMSE toward your target instead of adding harmful noise. Finally, I harden the inference loop so `final_predictions` is always the correct length (and fail early with a clear error if not), ensuring `submission.csv` is always written in the required format.'
- What this solution (achieved 32.62858) has done: 'I make two minimal, score-positive fixes that preserve your ensemble/TTA/SVR-blend logic: (1) ensure TTA actually applies test-time augmentation instead of repeating the exact same deterministic transform, and (2) fix the inference post-processing so it matches how the models were trained (trained with `100*sigmoid` inside the RMSE metric, but inference currently uses `100*sigmoid` which can saturate and worsen RMSE; we keep semantics but use the same scaling consistently by leaving the network output in logit space for SVR/features and only applying `100*sigmoid` at the very end). These are small, localized changes in the data transforms and in `tta()` that typically reduce RMSE substantially from your current 32.6 without changing architecture, training loops, or loss. The rest of the pipeline (weight discovery, fold averaging, SVR blend, clipping, submission format) remains unchanged. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 31.61103) has done: 'I make two minimal changes that usually improve RMSE for this competition without changing your model/ensemble logic: (1) use the correct training-time input normalization for timm backbones (ImageNet mean/std) instead of custom values, and (2) clamp the NN/SVR-blended predictions to the valid \[1,100\] range *inside* TTA before averaging, which reduces the impact of outlier augmentations on RMSE. Everything else (ConvNeXt backbone + tabular concat + same head, fold ensembling, SVR blending, loss/metric, and submission formatting) is preserved. This should move your score down from ~32.6 toward the 17.3 target while keeping changes localized and deterministic.'
- What this solution (achieved 31.61103) has done: 'Your current score (31.61 RMSE) is worse than the target (17.31), so we should make the smallest changes that legitimately improve generalization without changing your architecture, loss, or overall inference logic. The biggest score drag here is likely that you are forcing fp16 inference (`Learner(...).to_fp16()`) even though your saved weights (and SVR) were almost certainly produced in fp32; fp16 can noticeably degrade regression calibration and RMSE. I keep everything else identical but run inference in fp32 (no mixed precision) and also make the data pipeline deterministic for TTA by giving the `DataLoaders` the same transforms as the `DataLoader` (so the Learner/testloader are consistent). These are localized, execution-safe changes that typically reduce RMSE toward your target without altering the model design or ensembling semantics.'
- What this solution (achieved 31.61105) has done: 'Your current RMSE (31.61, lower is better) is still far from the target (17.31), so we should make the smallest legitimate changes that improve calibration/generalization without changing the model architecture, loss, or ensemble design. The most impactful low-risk fix here is that you’re running inference with dropout and batchnorm in training mode because `Network.forward()` calls `self.network(x)` (the backbone) instead of `self._backbone(x)`, which bypasses `model.eval()` for the backbone and makes predictions noisy—especially harmful under TTA. I change that single call so the backbone correctly respects eval mode, keeping the same features and head. I also disable cuDNN benchmarking (it can introduce nondeterministic algorithm choices that can worsen stability slightly) while keeping the rest of your pipeline identical and still writing a valid `submission.csv`.'
- What this solution (achieved 44.52656) has done: 'You’re currently far above the target RMSE (31.61 vs 17.31; lower is better), so we should make a small, legitimate improvement without changing your model/ensemble design. The biggest likely score drag is a preprocessing mismatch: your Albumentations pipeline uses ImageNet mean/std, but timm ConvNeXt models expect their own default normalization (mean=0.5/std=0.5 for many ConvNeXt variants). I switch the normalization to be pulled from the timm model’s `pretrained_cfg` per model (still “normalize then ToTensorV2”, same transforms otherwise), and keep everything else (TTA, folds, SVR blending, clipping, submission formatting) identical. This is a localized change that usually improves calibration and reduces RMSE for timm backbones, moving you toward the target band.'
- What this solution (achieved 27.23051) has done: 'Your current RMSE (44.53, lower is better) is still far from the target (17.31), so we should make the smallest changes that improve correctness/calibration without changing the model, loss, or ensemble design. The biggest likely issue is that ConvNeXt’s forward in timm often returns a feature *map* unless you call `forward_features()` (and global pool), so your head may be receiving the wrong representation depending on timm version/model, hurting predictions badly. I adjust the `Network.forward()` to use `forward_features()` + timm’s pooling/classifier reset path in a version-safe way (still the same backbone + linear embed + tabular concat + same head), and I also make sure we always run the backbone in eval-safe feature mode. Everything else (TTA, folds, SVR blend, clipping, submission format, paths) stays the same.'
- What this solution (achieved 44.52656) has done: 'Your current RMSE (27.23) is still above the target (17.31), so we should make a small, legitimate improvement without changing the model/ensemble design. The biggest low-risk gain here is fixing a train-vs-infer mismatch in the backbone feature extraction: after `reset_classifier(num_classes=0, global_pool="avg")`, many timm models expect `forward_head(feats, pre_logits=True)` to produce the exact pooled feature vector the head was trained on; using `global_pool(feats)` directly can be subtly different across timm versions/models and hurts calibration. I adjust `_extract_backbone_features()` to use `forward_features()` + `forward_head(..., pre_logits=True)` when available (fallback to the current pooling path otherwise), keeping the architecture and weights compatible. Everything else (TTA, SVR blending, clipping, submission format/paths) remains unchanged, and it still write a valid `submission.csv`.'
- What this solution (achieved 44.52656) has done: 'Your current RMSE (44.53; lower is better) is still far above the target (17.31), so we should make the smallest legitimate change that improves prediction correctness without changing the model architecture, loss, or ensembling approach. The most likely remaining score-drag is that your SVR is being fed the *concatenated* (image-embed + 12 tabular features) tensor, while the SVR models were almost certainly trained on image-only embeddings (dimension = `embed_dim`)—this feature mismatch can severely hurt SVR blending. I change the model to also return the image-only embedding (same backbone/head, no architectural change) and feed that to SVR, while keeping the NN prediction path and blending weights identical. Additionally, I ensure tabular tensors are always float32 and keep submission clipping/format unchanged.'
- What this solution (achieved 31.61105) has done: 'Your current RMSE (44.53, lower is better) is far above the target (17.31), so we should make a small, legitimate improvement without changing your architecture, loss, or ensemble/TTA design. The biggest likely score drag that fits “minimal change” is an inference mismatch: your checkpoint models were almost certainly trained with fastai’s `Normalize.from_stats(*imagenet_stats)` (ImageNet mean/std), but your current code dynamically pulls per-model timm normalization (ConvNeXt often uses 0.5/0.5), which can severely degrade predictions when weights were trained under different normalization. I switch normalization back to ImageNet stats for all backbones (keeping the rest of the Albumentations pipeline identical), and I also force `learn.model.eval()` right after loading each fold to ensure no accidental train-mode behavior leaks into inference. These are localized changes that typically reduce RMSE materially while preserving your core logic (same model, same folds, same TTA, same SVR blend, same clipping and submission format).'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import random
import pickle

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from PIL import Image

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import timm
from fastai.vision.all import *
from fastai.data.core import *

from sklearn.svm import SVR  # noqa: F401


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
def petfinder_rmse(input, target):
    pred = 100.0 * torch.sigmoid(input.flatten())
    target = target.flatten().to(pred.dtype)
    return torch.sqrt(F.mse_loss(pred, target))




## === cell 2
base_dir = "/kaggle/input"
model_weights_dir = os.path.join(base_dir, "saved-weights")

test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")

if not os.path.exists(test_file):
    test_folder = os.path.join(base_dir, "test")
    test_file = os.path.join(base_dir, "test.csv")



## === cell 3
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
    "convnext_base": 128,
    "conv_next": 64,
}

save_name = "/kaggle/working/"

weights_svr = {
    "beit_base_patch16_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window12_384_in22k": [0.1, 0.9],
    "swin_base_patch4_window7_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window7_224_in22k": [0.4, 0.6],
    "conv_next": [0.2, 0.8],
    "convnext_base": [0.2, 0.8],
}

model_weights = {
    "beit_base_patch16_224_in22k": 0.0,
    "swin_large_patch4_window12_384_in22k": 0.0,
    "swin_base_patch4_window7_224_in22k": 0.0,
    "swin_large_patch4_window7_224_in22k": 0.0,
    "convnext_base": 0.3,
    "conv_next": 0.7,
}


_FILE_CACHE = {}


def resolve_weight_path(fname: str) -> str:
    if fname in _FILE_CACHE:
        return _FILE_CACHE[fname]

    candidates = [
        os.path.join(base_dir, "saved-weights", fname),
        os.path.join(base_dir, "petfinder-pawpularity-score", "saved-weights", fname),
        os.path.join(base_dir, "petfinder-pawpularity-score", fname),
        os.path.join(base_dir, fname),
    ]
    for p in candidates:
        if os.path.exists(p):
            _FILE_CACHE[fname] = p
            return p

    for root, _, files in os.walk(base_dir):
        if fname in files:
            p = os.path.join(root, fname)
            _FILE_CACHE[fname] = p
            return p

    raise FileNotFoundError(
        f"Could not find required file: {fname}\n"
        f"Tried direct candidates:\n" + "\n".join(candidates) + "\n"
        f"And recursive search under: {base_dir}"
    )


def timm_norm_for_model(model_name: str, conv_next: bool):
    mean, std = imagenet_stats
    return list(mean), list(std)




## === cell 4
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].map(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1.0
test_csv.head()




## === cell 5
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

        df_data = self.df[self.cat].iloc[idx].values.astype(np.float32)
        return (img, df_data, np.float32(label_1))




## === cell 6
def get_data(batch_size, max_size, input_shape, norm_mean, norm_std):
    base_transform = albu.Compose(
        [
            albu.LongestMaxSize(max_size=max_size, interpolation=1),
            albu.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            albu.CenterCrop(height=input_shape[0], width=input_shape[1]),
            albu.Normalize(norm_mean, norm_std),
            ToTensorV2(),
        ]
    )

    tta_transform = albu.Compose(
        [
            albu.LongestMaxSize(max_size=max_size, interpolation=1),
            albu.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            albu.RandomResizedCrop(
                size=(input_shape[0], input_shape[[1]][0] if False else input_shape[1]),
                scale=(0.90, 1.00),
                ratio=(0.95, 1.05),
                p=1.0,
            ),
            albu.HorizontalFlip(p=0.5),
            albu.Normalize(norm_mean, norm_std),
            ToTensorV2(),
        ]
    )

    test_dataset = PetsDataset(test_csv, base_transform)
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    dls = DataLoaders.from_dsets(test_dataset, bs=batch_size, device=device)
    return dls, testloader, tta_transform




## === cell 7
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

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
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)

        self._backbone = base
        if hasattr(self._backbone, "reset_classifier"):
            self._backbone.reset_classifier(num_classes=0, global_pool="avg")

        backbone_dim = getattr(self._backbone, "num_features", None)
        if backbone_dim is None:
            raise AttributeError(
                "Backbone model missing num_features; cannot build head."
            )

        self.embed = nn.Linear(int(backbone_dim), embed_dim)
        nn.init.kaiming_normal_(self.embed.weight)
        nn.init.constant_(self.embed.bias, 0)

        self.p = 0.5
        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )
        self.network = self.__freeze_layer(self._backbone, freeze_layer)
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
        layers = [nn.Dropout(self.p)]
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != (number_of_hidden - 1):
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self._backbone, "forward_features"):
            feats = self._backbone.forward_features(x)

            if hasattr(self._backbone, "forward_head"):
                try:
                    feats2 = self._backbone.forward_head(feats, pre_logits=True)
                    if isinstance(feats2, torch.Tensor) and feats2.ndim == 2:
                        return feats2
                except TypeError:
                    pass

            if isinstance(feats, torch.Tensor) and feats.ndim == 4:
                if (
                    hasattr(self._backbone, "global_pool")
                    and self._backbone.global_pool is not None
                ):
                    feats = self._backbone.global_pool(feats)
                else:
                    feats = feats.mean(dim=(2, 3))
            if isinstance(feats, torch.Tensor) and feats.ndim == 4:
                feats = feats.flatten(1)
            return feats

        feats = self._backbone(x)
        if isinstance(feats, torch.Tensor) and feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        return feats

    def forward(self, x, tab):
        feats = self._extract_backbone_features(x)  # [bs, backbone_dim]
        x1 = self.embed(feats)  # [bs, embed_dim]
        x = torch.cat([x1, tab], dim=1)  # [bs, embed_dim + 12]
        reg = self.regression(x)  # logits
        return reg, x, x1




## === cell 8
def get_learner(
    model_name, batch_size, loss, metric, max_size, input_size, conv_next, save_path
):
    norm_mean, norm_std = timm_norm_for_model(model_name, conv_next)
    dls, testloader, tta_transform = get_data(
        batch_size, max_size, input_size, norm_mean, norm_std
    )
    dls = dls.to(device)

    if conv_next:
        if model_name == "convnext_base":
            backbone_name = "convnext_base"
        else:
            backbone_name = "convnext_large"
        network = timm.create_model(backbone_name, pretrained=False)
        model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    else:
        network = timm.create_model(model_name, pretrained=False)
        model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)

    model = model.to(device)

    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    return learn, testloader, tta_transform




## === cell 9
def tta(testloader, learn, svr, svr_weight, tta_transform=None, tta_steps=4):
    tta_outputs = []
    learn.model.eval()

    ds = testloader.dataset
    base_transform = ds.transform

    for step in range(tta_steps):
        if (tta_transform is not None) and (step > 0):
            ds.transform = tta_transform
        else:
            ds.transform = base_transform

        final_outputs = []
        svr_data = []

        with torch.no_grad():
            for images, tabular, _ in testloader:
                images = images.to(device, non_blocking=True)
                tabular = torch.as_tensor(tabular, device=device, dtype=torch.float32)

                reg_logits, _concat_embed, img_embed = learn.model(images, tabular)

                reg_logits = reg_logits.detach().to("cpu").float().numpy().reshape(-1)
                final_outputs.extend(reg_logits.tolist())
                svr_data.extend(img_embed.detach().float().cpu().numpy())

        final_outputs = np.asarray(final_outputs, dtype=np.float32)
        svr_data = np.asarray(svr_data, dtype=np.float32)

        if svr is None:
            nn_preds = (100.0 / (1.0 + np.exp(-final_outputs))).astype(np.float32)
            nn_preds = np.clip(nn_preds, 1.0, 100.0)
            tta_outputs.append(nn_preds)
        else:
            svr_preds = svr.predict(svr_data).astype(np.float32)
            nn_preds = (100.0 / (1.0 + np.exp(-final_outputs))).astype(np.float32)
            blended = svr_weight[0] * svr_preds + svr_weight[1] * nn_preds
            blended = np.clip(blended, 1.0, 100.0)
            tta_outputs.append(blended)

    ds.transform = base_transform
    tta_outputs_arr = np.mean(np.asarray(tta_outputs), axis=0)
    return tta_outputs_arr




## === cell 10
required_files = []
for model_name, _bs in models_list.items():
    m_name = model_name[: model_name.find("_", 5)]
    m_384 = False
    conv_next = False

    if model_name.find("384") != -1:
        m_384 = True
        m_name = m_name + "_384"

    if model_name.find("conv") != -1:
        conv_next = True
        m_name = model_name
        m_384 = True

    for fold in range(N_FOLDS):
        saved_name = f"{m_name}_fold_{fold}_full.pth"
        svr_name = f"{m_name}_svr_fold_{fold}_full.pkl"
        required_files.extend([saved_name, svr_name])

missing = []
for fn in required_files:
    try:
        _ = resolve_weight_path(fn)
    except FileNotFoundError:
        missing.append(fn)

HAVE_EXTERNAL_WEIGHTS = len(missing) == 0
if HAVE_EXTERNAL_WEIGHTS:
    print(f"All required weight/SVR files found: {len(required_files)} files.")
else:
    print(
        "WARNING: Missing external weight/SVR files; will run fallback inference.\n"
        f"Missing {len(missing)}/{len(required_files)} files, e.g.:\n"
        + "\n".join(missing[:10])
    )



## === cell 11
final_predictions = []

if HAVE_EXTERNAL_WEIGHTS:
    for model_name, bs in models_list.items():
        m_name = model_name[: model_name.find("_", 5)]

        fold_tta = []
        svr_weights = weights_svr[model_name]
        w = model_weights[model_name]

        m_384 = False
        conv_next = False

        if model_name.find("384") != -1:
            m_384 = True
            m_name = m_name + "_384"

        if model_name.find("conv") != -1:
            conv_next = True
            m_name = model_name
            m_384 = True

        print("#################################")
        print(f"Testing Model: {m_name}")
        print("#################################")

        for fold in range(N_FOLDS):
            print(f"Fold: {fold}")

            saved_name = f"{m_name}_fold_{fold}_full.pth"
            svr_name = f"{m_name}_svr_fold_{fold}_full.pkl"

            if m_384:
                learn, testloader, tta_transform = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_384,
                    input_shape_384,
                    conv_next,
                    save_name,
                )
            else:
                learn, testloader, tta_transform = get_learner(
                    model_name,
                    bs,
                    BCEWithLogitsLossFlat(),
                    petfinder_rmse,
                    max_size_224,
                    input_shape_224,
                    conv_next,
                    save_name,
                )

            weights_path = resolve_weight_path(saved_name)

            os.makedirs(save_name, exist_ok=True)
            local_pth = os.path.join(save_name, saved_name)
            if not os.path.exists(local_pth):
                with open(weights_path, "rb") as src, open(local_pth, "wb") as dst:
                    dst.write(src.read())

            learn.load(saved_name.replace(".pth", ""))

            learn.model.eval()

            svr_path = resolve_weight_path(svr_name)
            with open(svr_path, "rb") as f:
                svr_model = pickle.load(f)

            fold_tta.append(
                tta(
                    testloader,
                    learn,
                    svr_model,
                    svr_weights,
                    tta_transform=tta_transform,
                    tta_steps=4,
                )
            )

            del learn, svr_model
            torch.cuda.empty_cache()
            gc.collect()

        fold_tta = np.mean(np.asarray(fold_tta), axis=0)
        final_predictions.append(w * fold_tta)

    final_predictions = np.asarray(final_predictions, dtype=np.float32)
    final_predictions = np.sum(final_predictions, axis=0)

else:
    model_name = "conv_next"
    bs = models_list[model_name]
    conv_next = True
    m_384 = True

    learn, testloader, tta_transform = get_learner(
        model_name,
        bs,
        BCEWithLogitsLossFlat(),
        petfinder_rmse,
        max_size_384,
        input_shape_384,
        conv_next,
        save_name,
    )

    preds = tta(
        testloader,
        learn,
        svr=None,
        svr_weight=(0.0, 1.0),
        tta_transform=tta_transform,
        tta_steps=1,
    )
    final_predictions = preds.astype(np.float32)

    del learn
    torch.cuda.empty_cache()
    gc.collect()

final_predictions = np.asarray(final_predictions, dtype=np.float32).reshape(-1)
if len(final_predictions) != len(test_csv):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(final_predictions)} preds but test has {len(test_csv)} rows."
    )



## === cell 12
final_predictions = np.asarray(final_predictions, dtype=np.float32)
final_predictions = np.clip(final_predictions, 1.0, 100.0)

test_csv["Pawpularity"] = final_predictions
test_csv = test_csv[["Id", "Pawpularity"]]
test_csv.head()



## === cell 13
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(test_csv.describe())
