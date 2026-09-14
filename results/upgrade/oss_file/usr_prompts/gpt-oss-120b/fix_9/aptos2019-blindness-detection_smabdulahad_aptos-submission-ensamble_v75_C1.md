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

0.8926602111463993

# 6. Current score

-0.04658

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05708) has done: 'Implemented safe model loading with fallback to pretrained weights when checkpoint files are missing, preventing the FileNotFoundError and ensuring the model list is populated. Added error handling and logging, so the ensemble still runs even without the original checkpoints. This resolves the empty tensor list error during inference and allows the script to create a valid `submission.csv` file.'
- What this solution (achieved -0.10793) has done: 'I adjust the ensemble so that only models for which the true checkpoint files are found are used, and recompute the ensemble weights based solely on those successfully‑loaded models (using the provided validation scores). This removes pretrained‑only models that hurt performance and lets the best available trained model dominate, moving the score much closer to the target. The changes are limited to model loading, weight calculation, and the inference loop, preserving the original architecture and data handling.'
- What this solution (achieved -0.04861) has done: 'Implemented a stricter model‑loading strategy: only models with actual checkpoint files are kept in the ensemble (missing checkpoints are now skipped instead of falling back to generic ImageNet weights). If no checkpoints are found, the script falls back to a single strong pretrained EfficientNet‑b3 model so inference can still run. Ensemble weights are recomputed solely from the successfully‑loaded models, ensuring that weak pretrained‑only models no longer dilute performance. These adjustments keep the original architecture and training logic intact while aiming to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.36114) has done: 'I make the model‑loading step tolerant to wrong or missing checkpoint paths by searching the checkpoint directory for any file that matches the model key (ignoring case and spelling variations). This lets the intended fine‑tuned models be loaded when they exist, which should raise the validation‑derived ensemble score and thus improve the Kaggle metric.  
I also replace the hard argmax prediction with a probability‑weighted expected class (rounded) which aligns better with the quadratic weighted kappa metric for ordinal labels.'
- What this solution (achieved -0.0039) has done: 'I modify the model‑loading loop so every architecture is instantiated with ImageNet‑pretrained weights and, when a checkpoint exists, its fine‑tuned state is loaded (flag True). This guarantees that the ensemble always contains all models rather than falling back to a single network, which should raise the quadratic weighted kappa. I also adjust the weight calculation to use **all** loaded models (both checkpoint‑loaded and pretrained‑only) based on the provided validation scores, normalising them to sum to 1. These minimal changes keep the original architecture and inference logic intact while improving the expected score.'
- What this solution (achieved 0.07385) has done: 'I adjust the ensemble weighting to give importance only to models that successfully loaded a checkpoint (ignoring pure ImageNet‑pretrained models) and fall back to uniform weights when none are available. Then I change the final prediction from the rounded expected class to the class with highest probability (argmax), which better matches the ordinal Quadratic Weighted Kappa metric. These minimal tweaks keep the core architecture unchanged while improving the quality of the predictions, moving the score upward toward the target.'
- What this solution (achieved 0.06592) has done: 'I adjust the model‑loading logic so that only fine‑tuned checkpoints are kept; if none are available the script falls back to a single strong pretrained EfficientNet‑b3 model. I also replace the arg‑max prediction with an expected‑value (probability‑weighted) prediction, which better matches the ordinal quadratic weighted kappa metric. These minimal changes keep the core architecture intact while aiming to lift the score toward the target.'
- What this solution (achieved -0.04658) has done: 'I correct the typo in the model checkpoint paths (\"efficentNet\" → \"efficientnet\") so that the pretrained fine‑tuned weights can be found and loaded. This enables the ensemble to use the validated model scores for weighting, which should raise the quadratic weighted kappa toward the target value. All other logic remains unchanged.'

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
import warnings




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
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)




## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18.pth",
    "efficientnet_b0": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b0.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b3.pth",
    "efficientnet_b4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b4.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficientnet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = []


def find_alternative_path(expected_path, model_key):
    """Search the directory of expected_path for any .pth file that contains the model_key."""
    dir_path = os.path.dirname(expected_path)
    if not os.path.isdir(dir_path):
        return None
    candidates = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".pth") and model_key.lower() in f.lower()
    ]
    return candidates[0] if candidates else None


for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=True, num_classes=5)
    load_path = path if os.path.exists(path) else find_alternative_path(path, model_key)
    checkpoint_loaded = False
    if load_path and os.path.exists(load_path):
        try:
            state = torch.load(load_path, map_location=device)
            model.load_state_dict(state)
            checkpoint_loaded = True
        except Exception as e:
            warnings.warn(
                f"Error loading checkpoint for '{model_key}' from '{load_path}': {e}. "
                "Using ImageNet‑pretrained weights instead."
            )
    else:
        warnings.warn(
            f"Checkpoint for '{model_key}' not found at '{path}'. Using ImageNet‑pretrained weights."
        )
    model.to(device)
    model.eval()
    loaded_models.append((model_key, model, checkpoint_loaded))

if not any(loaded for _, _, loaded in loaded_models):
    fallback_key = "efficientnet_b3"
    fallback_name = model_names[fallback_key]
    warnings.warn(
        "No checkpointed models found; using only pretrained EfficientNet‑b3 for inference."
    )
    fallback_model = timm.create_model(fallback_name, pretrained=True, num_classes=5)
    fallback_model.to(device)
    fallback_model.eval()
    loaded_models = [(fallback_key, fallback_model, False)]




## === cell 5
validation_scores = {
    "resnet18": 0.879,
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
}




## === cell 6
checkpointed_keys = [k for k, _, loaded in loaded_models if loaded]
if checkpointed_keys:
    score_list = [validation_scores.get(k, 0.0) for k in checkpointed_keys]
    total_score = sum(score_list)
    weights = {
        k: validation_scores.get(k, 0.0) / total_score for k in checkpointed_keys
    }
else:
    uniform = 1.0 / len(loaded_models)
    weights = {k: uniform for k, _, _ in loaded_models}




## === cell 7
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        model_outputs = []
        for model_key, model, _ in loaded_models:
            if model_key not in weights:
                continue  # safety check
            logits = model(images)  # (B, 5)
            probs = nn.functional.softmax(logits, dim=1)  # (B, 5)
            weighted = weights[model_key] * probs  # scale by ensemble weight
            model_outputs.append(weighted.unsqueeze(0))  # (1, B, 5)
        if not model_outputs:
            raise RuntimeError("No model outputs available for inference.")
        stacked = torch.cat(model_outputs, dim=0)  # (M, B, 5)
        weighted_sum = torch.sum(stacked, dim=0)  # (B, 5)
        all_outputs.extend(weighted_sum.cpu().numpy())

all_outputs = np.array(all_outputs)  # shape (num_test, 5)

final_predictions = np.rint(np.dot(all_outputs, np.arange(5))).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)




## === cell 8
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
