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

3.12

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5505869260275978

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23101) has done: 'The failure comes from trying to load pretrained ensemble weights from a Kaggle input dataset that is not present in your environment, leaving `models_list` empty and causing downstream `torch.cat()` and `final_predictions` errors. I keep your inference-only pipeline intact, but add a safe fallback: if no weight files are found, we create the same model architecture with ImageNet pretrained weights so predictions can still be generated. I also make image loading robust (`RGB` conversion) and ensure `torch.load(..., map_location=...)` works on CPU/GPU consistently. Finally, I always write a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.1353) has done: 'Your current fallback uses ImageNet-pretrained backbones with a 5-class head that has random weights, so predictions are essentially noise and QWK stays low; we can move toward the target by making the fallback produce *sensible ordinal labels* from pretrained features without changing your overall inference-only flow. I keep the same dataloader, transforms, and `timm.create_model(...)` usage, but in the “no finetuned weights found” case I switch the model to output pretrained logits (1000 classes) and then map those logits to 5 ordered bins using the distribution from `train.csv`. This preserves the core idea (single-pass inference + arg-like discretization) while making outputs non-random and typically much closer to a reasonable kappa baseline. The submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.73661) has done: 'Your current fallback turns ImageNet probabilities into labels using only the max-softmax “confidence”, which is a weak signal for an ordinal medical severity task and tends to produce near-random ordering. To move the score upward toward the 0.55 target with minimal change and without altering the inference-only structure, I keep the same model/dataloader flow but switch the fallback to extract a stable embedding (`num_classes=0`) and use simple cosine similarity to 5 class prototypes computed from the training set. This preserves the same overall pipeline (single forward pass + deterministic postprocess) while making predictions align with DR class structure much better than confidence binning. I also keep the original finetuned-weight path intact and only activate the prototype fallback when finetuned weights are missing.'
- What this solution (achieved 0.0) has done: 'I fix the deterministic-algorithms crash by setting the required `CUBLAS_WORKSPACE_CONFIG` environment variable *before* importing torch, and by falling back to deterministic-safe cosine similarity via `F.linear` (normalized dot product) if needed. I also make sure the prototype tensor is on the correct device/dtype and guard against any remaining edge cases so `final_predictions` is always created. Finally, I keep your inference/prototype fallback logic intact and ensure a valid `submission.csv` is always written with the correct columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your current score is 0.0 because the script crashes in the non-fallback branch: when finetuned weights are found it call the models (which output 5 logits), but `models_list` currently contains a `num_classes=0` backbone in the missing-weights case and then later tries to `argmax` on `all_outputs` only if `fallback_prototypes` is False. To move the score upward toward the 0.5506 target with minimal change, I (1) make the code robust so it never ends up in an inconsistent “no prototypes but also no logits” state, and (2) ensure the finetuned-weight path and fallback-prototype path are mutually consistent and always produce `final_predictions`. I also expand `model_paths` to include more potential local weight filenames if present, but keep the same ensemble/inference logic; if none exist, we stay with the prototype fallback. Finally, I keep submission formatting identical and add a couple of guards so the CSV is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 4
"""
Fix the 0.0-score failure by guaranteeing a consistent inference mode:
- If any finetuned 5-class weights are found and loaded, we do plain logits averaging + argmax.
- Otherwise, we deterministically build class prototypes from a pretrained feature extractor
  and predict by cosine-similarity softmax expected-class rounding (same fallback logic as before).

This keeps the core inference-only design, but prevents the "inconsistent branch" that can crash
and yield no valid submission (hence 0.0 score).
"""

candidate_weight_paths = [
    "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
    "/kaggle/input/aptos_ensamble-models/seresnext101_32x4d.pth",
    "/kaggle/input/aptos2019-blindness-detection/seresnext101_32x4d.pth",
]

model_name = "seresnext101_32x4d"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

found_weight_path = None
for p in candidate_weight_paths:
    if os.path.exists(p):
        found_weight_path = p
        break

models_list = []
loaded_any = False
fallback_prototypes = False

if found_weight_path is not None:
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    state = torch.load(found_weight_path, map_location="cpu")
    model.load_state_dict(state)
    loaded_any = True
    fallback_prototypes = False
    models_list.append(model)
else:
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    loaded_any = False
    fallback_prototypes = True
    models_list.append(model)

for m in models_list:
    m.to(device)
    m.eval()

print(
    f"Using {len(models_list)} model(s). Loaded finetuned weights: {loaded_any}. Fallback prototypes: {fallback_prototypes}"
)

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

prototypes = None
if fallback_prototypes:
    train_dataset = BlindnessDataset(
        train_csv_file, train_root_dir, transform=transform, test=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    sums = None
    counts = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building class prototypes"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = [m(images).unsqueeze(0) for m in models_list]  # [n_models, B, D]
            feats = torch.cat(feats, dim=0)
            feats = torch.mean(feats, dim=0)  # [B, D]

            if sums is None:
                sums = torch.zeros(
                    (5, feats.shape[1]), device=device, dtype=feats.dtype
                )

            for c in range(5):
                mask = labels == c
                if mask.any():
                    sums[c] += feats[mask].sum(dim=0)
                    counts[c] += int(mask.sum().item())

    counts_safe = counts.clamp(min=1).to(device)
    prototypes = sums / counts_safe.unsqueeze(1)
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)

    print("Prototype counts:", counts.cpu().tolist())



## === cell 5
import torch.nn.functional as F

all_outputs = []
all_preds = []

FALLBACK_TEMP = 2.5

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer test"):
        images = images.to(device, non_blocking=True)

        if not fallback_prototypes:
            outputs = [model(images).unsqueeze(0) for model in models_list]
            outputs = torch.cat(outputs, dim=0)  # [n_models, batch, 5]
            averaged_outputs = torch.mean(outputs, dim=0)  # [batch, 5]
            all_outputs.append(averaged_outputs.detach().cpu().numpy())
        else:
            if prototypes is None:
                raise RuntimeError(
                    "Fallback prototypes enabled but prototypes were not built."
                )
            prot = prototypes.to(device=device, dtype=images.dtype)

            feats = [m(images).unsqueeze(0) for m in models_list]  # [n_models, B, D]
            feats = torch.cat(feats, dim=0)
            feats = torch.mean(feats, dim=0)  # [B, D]
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # [B, D]

            sims = F.linear(feats, prot)  # [B, 5]

            probs = torch.softmax(sims / FALLBACK_TEMP, dim=1)  # [B, 5]
            expected = (probs * torch.arange(5, device=device, dtype=probs.dtype)).sum(
                dim=1
            )
            preds = torch.round(expected).to(torch.int64)

            all_preds.append(preds.detach().cpu().numpy())

if not fallback_prototypes:
    all_outputs = np.concatenate(all_outputs, axis=0)
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)
else:
    final_predictions = np.concatenate(all_preds, axis=0).astype(int)
    final_predictions = np.clip(final_predictions, 0, 4)

print("Pred shape:", final_predictions.shape)
print("Pred label counts:", np.bincount(final_predictions, minlength=5).tolist())



## === cell 6
test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(
    final_predictions
), "Mismatch between test ids and predictions length."

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
