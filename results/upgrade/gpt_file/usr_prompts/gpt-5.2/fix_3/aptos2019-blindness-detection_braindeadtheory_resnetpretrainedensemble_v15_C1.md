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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8911934928977304

# 6. Current score

-0.05088

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08916) has done: 'The runtime failure comes from trying to load pretrained weights from a non-existent `../input/resnet/` dataset; I make the script robust by falling back to ImageNet weights (same ResNet architectures) when those `.pt` files aren’t available. I also fix path handling to use Kaggle’s absolute `/kaggle/input/...` locations (so it runs consistently), and replace the notebook-only tqdm import with standard `tqdm` to avoid import/runtime issues. Finally, I keep the ensemble logic intact (average logits then argmax) but make it work even if only one model successfully loads, ensuring a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -0.05088) has done: 'Your current very low kappa is consistent with a label/target mismatch: you’re using ImageNet-pretrained backbones but random-initialized 5-class heads (because the custom `.pt` weights aren’t present), so predictions are essentially arbitrary. To move the score toward your target with minimal change and without altering the model/training core, I (1) make the script *actually use the provided trained weights* by searching common `/kaggle/input/**` locations for the expected `.pt` files, (2) only fall back to ImageNet when those files truly aren’t available, and (3) ensure inference is numerically stable and correct by running under `inference_mode()` and softmax-averaging probabilities (keeps the same ensemble semantics: average then argmax, but improves calibration for kappa). The output submission format and paths remain unchanged, and the rest of the pipeline (dataset, transforms, models, inference loop) stays intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob

base_in = "/kaggle/input"
if os.path.exists(base_in):
    for dirname, _, filenames in os.walk(base_in):
        for filename in filenames[:5]:
            print(os.path.join(dirname, filename))
        break



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm



## === cell 2
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file).reset_index(drop=True)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                TRAIN_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self.eye_frame.loc[idx, "diagnosis"])
        else:
            img_name = os.path.join(
                TEST_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self.eye_frame.loc[idx, "id_code"]




## === cell 3
test_dataset = APTOSDataset(csv_file=TEST_CSV, filetype="test", transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 4
def _try_load_state_dict(model, weight_path, device):
    if weight_path is None or (not os.path.exists(weight_path)):
        return False
    state = torch.load(weight_path, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=False)
    return True


def build_resnet(model_name, num_classes=5, weights_fallback=True):
    if model_name == "resnet152":
        if weights_fallback:
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            model = torchvision.models.resnet152(weights=weights)
        else:
            model = torchvision.models.resnet152(weights=None)
    elif model_name == "resnet101":
        if weights_fallback:
            weights = torchvision.models.ResNet101_Weights.DEFAULT
            model = torchvision.models.resnet101(weights=weights)
        else:
            model = torchvision.models.resnet101(weights=None)
    else:
        raise ValueError("Unsupported model_name")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def find_weight_file(filename, preferred_dir="/kaggle/input/resnet"):
    cand = os.path.join(preferred_dir, filename)
    if os.path.exists(cand):
        return cand
    matches = glob.glob(os.path.join("/kaggle/input", "**", filename), recursive=True)
    return (
        matches[0] if matches else cand
    )  # keep deterministic fallback path for logging


paths = {
    "model0": find_weight_file("FinalResnet152_0.pt"),
    "model1": find_weight_file("FinalResnet02.pt"),
    "model2": find_weight_file("FinalResnet01.pt"),
    "model3": find_weight_file("FinalResnet00.pt"),
    "model4": find_weight_file("FinalResnet0.pt"),
}

models_list = []

m0 = build_resnet("resnet152", num_classes=5, weights_fallback=True)
loaded0 = _try_load_state_dict(m0, paths["model0"], device)
print("model0 weights loaded from file:", loaded0, "| path:", paths["model0"])
models_list.append(m0.to(device))

for key in ["model1", "model2", "model3", "model4"]:
    m = build_resnet("resnet101", num_classes=5, weights_fallback=True)
    loaded = _try_load_state_dict(m, paths[key], device)
    print(f"{key} weights loaded from file:", loaded, "| path:", paths[key])
    models_list.append(m.to(device))

if not any(
    [loaded0]
    + [os.path.exists(paths[k]) for k in ["model1", "model2", "model3", "model4"]]
):
    print(
        "WARNING: No custom .pt weight files found under /kaggle/input; ensemble will use ImageNet backbones with random heads."
    )




## === cell 5
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict(train)"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.detach().cpu())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100.0
    else:
        predictions = []
        img_ids = []
        out = []
        for inputs, img_id in tqdm(data_loader, desc="Predict(test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu())
            img_ids.extend(list(img_id))
            out.extend(outputs.detach().cpu())
        predictions = [int(pred.item()) for pred in predictions]
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        return final_predictions, out, img_ids




## === cell 6
with torch.inference_mode():
    for m in models_list:
        m.eval()

    probs_list = []
    ids_ref = None
    for mi, m in enumerate(models_list):
        print(f"Computing Test Predictions for model {mi}")
        _, out_i, ids_i = compute_predictions(m, "test", test_loader, device)
        out_i = torch.stack(out_i, dim=0)  # [N, 5]
        prob_i = torch.softmax(out_i, dim=1)
        probs_list.append(prob_i)
        if ids_ref is None:
            ids_ref = ids_i
        else:
            if ids_i != ids_ref:
                raise RuntimeError(
                    "Test id order mismatch between models; cannot ensemble safely."
                )



## === cell 7
prob_avg = torch.stack(probs_list, dim=0).mean(dim=0)  # [N, 5]
predictions = torch.argmax(prob_avg, dim=1).cpu().numpy().astype(int)

final_predictions = pd.DataFrame(
    {"id_code": np.array(ids_ref), "diagnosis": predictions}
)
final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
