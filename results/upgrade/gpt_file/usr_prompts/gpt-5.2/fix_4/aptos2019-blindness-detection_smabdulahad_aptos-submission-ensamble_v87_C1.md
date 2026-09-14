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

0.8949546090873821

# 6. Current score

0.02995

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0763) has done: 'I fix the runtime by making the model weight loading robust to missing Kaggle input files: if the ensemble checkpoints aren’t available, the code fall back to a single pretrained timm model so inference can still run end-to-end. I also fix the ensemble weighting logic so weights are computed only for actually loaded models, preventing the empty `torch.cat()` crash. To preserve your core inference semantics, the prediction still uses softmax probabilities and argmax over 5 classes, just with a safe fallback path. Finally, I ensure the submission CSV is always written with the required columns and correct row alignment to `test.csv`.'
- What this solution (achieved 0.0748) has done: 'Your low score is almost certainly because the fallback path uses an ImageNet-pretrained model with a randomly initialized 5-class head, so predictions are essentially noise. To move the score toward the target with minimal core-logic change, I keep your exact inference pipeline (resize/normalize, softmax, weighted averaging, argmax) but make the fallback choose a timm model that already has pretrained weights for this exact competition (`hf_hub` APTOS fine-tuned weights), which is still “pretrained weights + num_classes=5” and doesn’t change architecture or evaluation semantics. I also add a tiny robustness tweak to handle common checkpoint formats (`state_dict` key) without changing behavior when your original ensemble files exist. This should lift QWK substantially toward your target while keeping runtime and code structure nearly identical.'
- What this solution (achieved 0.02995) has done: 'Your score is far below the target, and the most likely cause is that the current fallback model isn’t actually loading APTOS-finetuned weights (HF hub access is typically unavailable in Kaggle offline), so the 5-class head remains randomly initialized and predictions are near-random. To move the score toward the target with minimal change and without altering your inference semantics (resize/normalize → model → softmax → weighted average → argmax), I make the fallback load a locally-available APTOS-finetuned checkpoint from the Kaggle dataset if present, and only then fall back to the HF-hub attempt. I also make checkpoint loading slightly more robust to common key prefixes (`module.`, `model.`), which can otherwise silently prevent using the good weights. This should substantially increase QWK toward your target while keeping the architecture and prediction pipeline the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




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

test_df = pd.read_csv(test_csv_file)
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 2 if torch.cuda.is_available() else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
model_paths = {
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}


def _extract_state_dict(maybe_ckpt):
    if isinstance(maybe_ckpt, dict):
        if "state_dict" in maybe_ckpt and isinstance(maybe_ckpt["state_dict"], dict):
            return maybe_ckpt["state_dict"]
        if "model" in maybe_ckpt and isinstance(maybe_ckpt["model"], dict):
            return maybe_ckpt["model"]
    return maybe_ckpt


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def _try_load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = _clean_state_dict_keys(_extract_state_dict(ckpt))
    model.load_state_dict(state, strict=True)
    return model


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    model = _try_load_weights(model, path)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    local_fallback_candidates = [
        "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
        "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
        "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
        "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
        "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
    ]

    loaded = False
    for p in local_fallback_candidates:
        if os.path.exists(p):
            base = os.path.basename(p).lower()
            if "b1" in base:
                mk, mn = "efficientnet_b1", "efficientnet_b1"
            elif "b2" in base:
                mk, mn = "efficientnet_b2", "efficientnet_b2"
            elif "b3" in base:
                mk, mn = "efficientnet_b3", "efficientnet_b3"
            elif "seresnext101" in base:
                mk, mn = "seresnext101_32x4d", "seresnext101_32x4d"
            else:
                continue

            model = timm.create_model(mn, pretrained=False, num_classes=5)
            try:
                model = _try_load_weights(model, p)
                model.to(device).eval()
                loaded_model_keys = [mk]
                models_list = [model]
                loaded = True
                print("Loaded local fallback checkpoint:", p)
                break
            except Exception as e:
                print("Failed loading local fallback checkpoint:", p, "Error:", repr(e))

    if not loaded:
        fallback_key = "tf_efficientnet_b3_ns"
        try:
            model = timm.create_model(
                fallback_key,
                pretrained=True,
                num_classes=5,
                pretrained_cfg_overlay={
                    "hf_hub_id": "timm/tf_efficientnet_b3_ns.aptos2019"
                },
            )
            loaded_model_keys = [fallback_key]
            models_list = [model.to(device).eval()]
        except Exception as e:
            print(
                "APTOS finetuned fallback unavailable, using ImageNet fallback. Error:",
                repr(e),
            )
            fallback_key = "efficientnet_b0"
            model = timm.create_model(fallback_key, pretrained=True, num_classes=5)
            loaded_model_keys = [fallback_key]
            models_list = [model.to(device).eval()]

print("Loaded models:", loaded_model_keys)



## === cell 5
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
    "tf_efficientnet_b3_ns": 0.91,
}

available_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = float(sum(available_scores.values()))
weights = {
    k: (v / total_score if total_score > 0 else 1.0 / len(available_scores))
    for k, v in available_scores.items()
}

print("Ensemble weights:", weights)



## === cell 6
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device, non_blocking=True)

        probs_list = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)
            probs_list.append(weights[model_key] * probs)

        weighted_outputs = torch.stack(probs_list, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)

assert len(final_predictions) == len(test_df), (len(final_predictions), len(test_df))



## === cell 7
submission_df = pd.DataFrame(
    {
        "id_code": test_df["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape=", submission_df.shape)
print(submission_df.head())
