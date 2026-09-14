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

0.8300184662704527

# 6. Current score

-0.17491

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08442) has done: 'I fix the missing external model checkpoint issue by falling back to a timm pretrained model when the provided weight file path doesn’t exist, so `models_list` is never empty. I also make image loading robust (RGB conversion) and ensure `torch.load` uses `map_location` for CPU/GPU compatibility. Then I keep the same ensemble-averaging core logic, but make it work for both the “1+ models” case and the fallback single-model case. Finally, I always write a valid `submission.csv` with the required `id_code,diagnosis` columns and correct row alignment.'
- What this solution (achieved -0.3469) has done: 'Your current score is far below the target, and the biggest issue is that the fallback model is a plain ImageNet-pretrained classifier with a 5-class head that’s not trained for DR labels—so predictions are essentially random for this task. To move the score toward the target while keeping your core “timm model(s) + average logits + argmax” logic intact, the minimal high-impact fix is to actually load the intended DR-trained checkpoint(s) by pointing to the correct existing input path. I also make the checkpoint loading robust to common formats (`state_dict`, `model`, `module.` prefixes) without changing the inference semantics. The rest of the pipeline (dataset, transforms, dataloader, ensemble averaging, submission writing) stays the same.'
- What this solution (achieved -0.02557) has done: 'Your score is far below the target, so we should increase performance, not tune toward a lower score. The biggest current issue is that you’re very likely *not loading the intended DR-trained checkpoints* (those `/kaggle/input/.../models/*.pth` paths don’t exist in your provided input tree), so you fall back to ImageNet-pretrained classifiers and get near-random predictions. I make the code *discover and load any `.pth` checkpoints that actually exist under the competition input directory*, and only fall back to ImageNet weights if none are found, keeping your exact “timm model(s) → average logits → argmax” core logic unchanged. I also make checkpoint loading robust to common wrappers/prefixes and enforce safe DataLoader settings in Kaggle to avoid silent worker issues.'
- What this solution (achieved 0.08182) has done: 'Your score is far below the target (gap ≈ -0.855), so we should improve performance without changing the core “timm model(s) → average logits → argmax” inference logic. The most likely cause of the near-random kappa is that no meaningful DR-trained checkpoints are being loaded, so the code falls back (or effectively behaves like) untrained/random heads. I (1) expand checkpoint discovery to include common DR weight extensions (`.pt`, `.bin`) and (2) make loading compatible with more checkpoint formats by handling `ema_state_dict` and by mapping common classifier head key names to timm’s `classifier`/`fc` when needed—still using the same model architectures and argmax over averaged logits. If no compatible checkpoints exist, the behavior remains the same as your current fallback, and the script still produce a valid `submission.csv`.'
- What this solution (achieved 0.00282) has done: 'Your score is far below the target (0.08182 vs 0.83), and the most likely reason is that no DR-trained checkpoints are actually being loaded, so you end up ensembling ImageNet-pretrained models with randomly-initialized 5-class heads. I make the checkpoint discovery/load path robust to this by (1) searching both `/kaggle/input` and `/kaggle/data` trees you have available, (2) allowing a safe fallback where we load pretrained backbones but **replace the head to 5 classes after loading** (so we don’t accidentally discard pretrained weights), and (3) enabling `timm`’s `checkpoint_path` loading when possible (often handles more formats correctly) while still keeping your exact “timm model(s) → average logits → argmax” inference logic. These are minimal, inference-only changes aimed at getting non-random predictions and moving kappa upward toward the target. The submission writing and column schema stay unchanged.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should increase performance with the smallest change that preserves your core “timm model(s) → average logits → argmax” inference logic. The biggest issue is that your “fallback” uses ImageNet-pretrained backbones with a randomly initialized 5-class head, which produces near-random DR labels. I keep the same models and ensembling, but change the fallback to use timm models pretrained on ImageNet and output the original 1000 classes, then map those logits to 5 classes via a fixed, deterministic linear projection (no training) so predictions become less random while preserving the same inference loop structure. I also switch the resize to each model’s native timm config (still just resize/normalize) to reduce preprocessing mismatch without changing the overall pipeline.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with near-random predictions, which in your code happens when no real DR-trained checkpoints are loaded and you fall back to a deterministic random projection. To move the score upward toward the 0.83 target while preserving the exact core logic (“timm model(s) → average logits → argmax”), I (1) ensure checkpoint discovery actually finds *competition-provided* weights by searching only realistic weight locations and filtering out irrelevant tiny/empty files, and (2) load checkpoints in a more compatible way for common training wrappers by also accepting `model_state_dict` and `ema` variants and handling `state_dict` nesting without changing inference semantics. I also switch the preprocessing transform to be derived from the *same model family as the first successfully loaded model* (or keep resnet18 if none), reducing preprocessing mismatch while keeping the same “resize/normalize” style pipeline. The submission writing stays identical and still always produce a valid `submission.csv`.'
- What this solution (achieved -0.14078) has done: 'Your 0.0 score strongly suggests the current run is still effectively “fallback/random” (no real DR-trained checkpoints loaded), so the smallest legitimate way to move toward the 0.83 target is to (1) stop averaging random 5-class heads and instead use a known strong ImageNet backbone as the fallback (single model) and (2) use timm’s native pretrained weights + correct head replacement via `num_classes=5` (keeps architecture/inference loop identical: logits → argmax). I keep your ensemble-averaging and submission-writing logic intact, but make checkpoint discovery prefer the actual competition folder and only include checkpoints that look like real model weights (by size), reducing the chance of loading irrelevant files. Finally, I derive the preprocessing transform from the exact first inference model used (not a separately created model), eliminating a common mismatch that can collapse performance.'
- What this solution (achieved -0.17491) has done: 'Your current score is far below the target, so we should increase performance with the smallest changes that keep your exact “timm model(s) → average logits → argmax” inference core intact. The biggest likely issue is the fallback path: `pretrained=True, num_classes=5` in timm typically initializes a random 5-class head (not ImageNet-pretrained), yielding near-random predictions and negative kappa. I keep the same model list/averaging/argmax, but make the fallback produce meaningful 5-class logits by using a fixed 1000→5 class aggregation from true ImageNet logits (no training, deterministic). I also ensure the inference transform is derived from the *actual inference model* used (1000-class backbone) to avoid preprocessing mismatch that can further degrade predictions.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import timm

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = None



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_dataset = None
test_loader = None



## === cell 4
model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

CKPT_SEARCH_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input",
]


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "ema_state_dict",
            "ema",
            "model",
            "net",
            "weights",
        ):
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, dict):
                    if "state_dict" in v and isinstance(v["state_dict"], dict):
                        return v["state_dict"]
                    return v
    return ckpt


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            out[k] = v
            continue
        for pref in ("module.", "model.", "net.", "encoder."):
            if k.startswith(pref):
                k = k[len(pref) :]
        out[k] = v
    return out


def _remap_head_keys_for_timm(model, state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    try:
        classifier_name = model.get_classifier()
    except Exception:
        classifier_name = None
    if not classifier_name:
        return state_dict

    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("last_linear."):
            nk = classifier_name + nk[len("last_linear") :]
        elif nk.startswith("head."):
            if classifier_name != "head":
                nk = classifier_name + nk[len("head") :]
        remapped[nk] = v
    return remapped


def _find_checkpoint_files(roots):
    exts = (".pth", ".pt", ".bin")
    ckpts = []
    seen = set()
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(exts):
                    continue
                p = os.path.join(dirpath, fn)
                if p in seen:
                    continue
                try:
                    if os.path.getsize(p) < 10 * 1024 * 1024:  # < 10MB
                        continue
                except OSError:
                    continue
                ckpts.append(p)
                seen.add(p)
    return sorted(ckpts)


def _guess_model_key_from_path(path):
    p = path.lower().replace("-", "_")
    for key in model_names.keys():
        if key.lower() in p:
            return key
    if "seresnext50" in p:
        return "seresnext50_32x4d"
    if "seresnext101" in p:
        return "seresnext101_32x4d"
    if "efficientnet" in p and "b5" in p:
        return "efficientnet_b5"
    if "inceptionresnetv2" in p or "inception_resnet_v2" in p:
        return "inception_resnet_v2"
    if "inceptionv4" in p or "inception_v4" in p:
        return "inception_v4"
    if "resnet18" in p:
        return "resnet18"
    return None


discovered_ckpts = _find_checkpoint_files(CKPT_SEARCH_ROOTS)
print(f"Discovered {len(discovered_ckpts)} checkpoint(s) (>=10MB) under search roots.")

model_paths = {}
for p in discovered_ckpts:
    key = _guess_model_key_from_path(p)
    if key is not None and key not in model_paths:
        model_paths[key] = p

print("Resolved checkpoints per architecture:")
for k in sorted(model_names.keys()):
    print(f"  {k}: {model_paths.get(k, None)}")



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_any_ckpt = False
first_loaded_model_name = None


def _create_and_load(model_name, ckpt_path):
    if ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            m = timm.create_model(
                model_name, pretrained=False, num_classes=5, checkpoint_path=ckpt_path
            )
            return m, True
        except Exception as e:
            print(
                f"[warn] timm checkpoint_path load failed for {model_name} from {ckpt_path}: {e}"
            )

    m = timm.create_model(model_name, pretrained=False, num_classes=5)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _strip_known_prefixes(_extract_state_dict(ckpt))
        state = _remap_head_keys_for_timm(m, state)
        missing, unexpected = m.load_state_dict(state, strict=False)
        print(
            f"[warn] manual load {model_name}: missing={len(missing)} unexpected={len(unexpected)} (from {ckpt_path})"
        )
        return m, True
    return m, False


for model_key, model_name in model_names.items():
    ckpt_path = model_paths.get(model_key, None)
    model, loaded = _create_and_load(model_name, ckpt_path)
    if loaded:
        loaded_any_ckpt = True
        if first_loaded_model_name is None:
            first_loaded_model_name = model_name
        models_list.append(model.to(device).eval())


class ImageNetToDR5(torch.nn.Module):
    def __init__(self, backbone_1000: torch.nn.Module):
        super().__init__()
        self.backbone = backbone_1000
        self.register_buffer(
            "bins", torch.arange(1000, dtype=torch.long) % 5, persistent=False
        )

    def forward(self, x):
        logits1000 = self.backbone(x)  # (B, 1000)
        out = logits1000.new_zeros((logits1000.shape[0], 5))
        for c in range(5):
            out[:, c] = logits1000[:, self.bins == c].mean(dim=1)
        return out


fallback_wrapper = None
transform_model_name_for_cfg = None

if len(models_list) == 0:
    print(
        "[warn] No usable DR checkpoints found; falling back to ImageNet-pretrained backbone with deterministic 1000->5 aggregation."
    )
    fallback_name = "efficientnet_b5"
    backbone = (
        timm.create_model(fallback_name, pretrained=True, num_classes=1000)
        .to(device)
        .eval()
    )
    fallback_wrapper = ImageNetToDR5(backbone).to(device).eval()
    models_list = [fallback_wrapper]
    transform_model_name_for_cfg = fallback_name
    first_loaded_model_name = fallback_name
else:
    transform_model_name_for_cfg = first_loaded_model_name

print(
    f"Loaded {len(models_list)} model(s) on {device}. First model for data_cfg: {transform_model_name_for_cfg}"
)

_tmp_for_cfg = timm.create_model(transform_model_name_for_cfg, pretrained=True)
_ref_cfg = timm.data.resolve_model_data_config(_tmp_for_cfg)
transform = timm.data.create_transform(**_ref_cfg, is_training=False)
del _tmp_for_cfg

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)

        if len(models_list) == 1:
            averaged_outputs = models_list[0](images)
        else:
            outputs = [model(images).unsqueeze(0) for model in models_list]
            outputs = torch.cat(outputs, dim=0)
            averaged_outputs = torch.mean(outputs, dim=0)

        all_outputs.append(averaged_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)
print("Predictions shape:", final_predictions.shape)



## === cell 7
test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(
    final_predictions
), "Mismatch between test ids and predictions."

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with", len(submission_df), "rows.")
