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

0.8650788169580392

# 6. Current score

-0.01626

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.0427) has done: 'I add robust handling for missing model files: when a weight file isn’t found the code fall back to a pretrained version of the architecture instead of crashing. I also recompute ensemble weights only for the models that are actually loaded, and guard the inference loop so that it produces a default “all‑zeros” prediction if no models are available. Finally, I ensure the submission CSV is always written.'
- What this solution (achieved 0.07911) has done: 'I keep the overall pipeline and model loading unchanged, but replace the naïve `argmax` post‑processing with an expected‑value rounding approach. By computing the weighted average of class indices from the soft‑max probabilities and rounding to the nearest integer (clipped to 0‑4), we obtain smoother predictions that typically improve quadratic weighted kappa compared to a hard argmax, moving the score toward the target.'
- What this solution (achieved -0.07485) has done: 'I correct the model weight file path typo so the pretrained‑fine‑tuned weights can be loaded (instead of falling back to an untrained ImageNet model). This give the ensemble meaningful predictions and move the quadratic weighted kappa much closer to the target score while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.58346) has done: 'I load the training labels to compute the empirical class distribution and use that distribution instead of a uniform guess when no fine‑tuned model is available. This simple prior typically yields a higher quadratic weighted kappa than the current uniform fallback, moving the score toward the target while keeping the core model‑inference logic unchanged.'
- What this solution (achieved 0.0) has done: 'I make the model‑loading step discover all available fine‑tuned *.pth* files under the input directory instead of hard‑coding a single path. This allows the ensemble to include any additional pretrained models that actually exist, which improves the weighted prediction and moves the quadratic weighted kappa closer to the target. The rest of the pipeline, including transforms, weighting by validation scores, expected‑value rounding and CSV output, stays unchanged.'
- What this solution (achieved -0.02515) has done: 'I keep the overall pipeline and model handling unchanged, but replace the simple rounding of the expected‑value prediction with a distribution‑matching assignment: the predictions are reordered so that their overall class frequencies follow the empirical label distribution from the training set. This small post‑processing tweak preserves the core logic while usually raising quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'We replace the distribution‑matching post‑processing with a straightforward expected‑value rounding (clipped to the valid class range). This keeps the model‑inference pipeline unchanged while providing predictions that align better with the quadratic weighted kappa metric, moving the score upward toward the target.'
- What this solution (achieved -0.00265) has done: 'I add a fallback that loads each listed architecture with ImageNet‑pretrained weights when a fine‑tuned *.pth* file is not found, ensuring the ensemble always contains models (instead of reverting to a uniform prior). This gives the pipeline real predictions and moves the quadratic weighted kappa toward the target. The rest of the logic, weighting, and CSV output remain unchanged.'
- What this solution (achieved 0.0) has done: 'I added a global device definition in the model‑loading cell so the inference loop always has a valid `device` variable, even when no checkpoint files are found. This eliminates the NameError that halted execution, allowing the ensemble (or fallback priors) to run and produce `final_predictions`. With the device fixed, the submission CSV is correctly written.'
- What this solution (achieved 0.00962) has done: 'I make the model‑loading step fall back to ImageNet‑pretrained weights whenever a fine‑tuned *.pth* file is missing or cannot be loaded. This keeps the original architecture, weighting and post‑processing unchanged but replaces the uniform‑prior fallback with reasonable visual features, which should raise the quadratic weighted kappa toward the target. The only modification is in the loading loop (cell 4); all other cells remain the same.'
- What this solution (achieved 0.18065) has done: 'I add a deterministic random seed, blend the model soft‑max outputs slightly with the class‑distribution prior to give a more sensible base prediction, and replace the simple rounding of the expected‑value with a distribution‑matching step that forces the overall predicted class frequencies to follow the empirical label distribution from the training set. These small changes keep the original architecture and ensemble logic intact while providing a more calibrated prediction that should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.01626) has done: 'I simplify the post‑processing: instead of forcing the predictions to match the training class distribution, I keep the blended model‑+‑prior probabilities, compute the expected class value for each image and round it to the nearest integer (clipped to 0‑4). This small change preserves the ensemble logic while providing smoother, more calibrated predictions, which should raise the quadratic weighted kappa toward the target.'

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

torch.manual_seed(42)
np.random.seed(42)

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
if os.path.exists(train_csv_file):
    train_df = pd.read_csv(train_csv_file)
    class_counts = train_df["diagnosis"].value_counts().sort_index()
    class_probs = (class_counts / class_counts.sum()).values.astype(float)  # shape (5,)
else:
    class_probs = np.full(5, 1 / 5, dtype=float)




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
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)




## === cell 4
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_paths = glob.glob("/kaggle/input/**/*.pth", recursive=True)

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

model_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in checkpoint_paths}

models_list = []
loaded_model_keys = []

for model_key, timm_name in model_names.items():
    if model_key in model_paths:
        path = model_paths[model_key]
        try:
            model = timm.create_model(timm_name, pretrained=False, num_classes=5)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded fine‑tuned weights for {model_key} from {path}")
        except Exception as e:
            warnings.warn(
                f"Failed to load fine‑tuned weights for {model_key} ({e}). Using ImageNet pretrained."
            )
            model = timm.create_model(timm_name, pretrained=True, num_classes=5)
    else:
        model = timm.create_model(timm_name, pretrained=True, num_classes=5)
        print(f"No checkpoint for {model_key}; using ImageNet pretrained model.")

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if not models_list:
    print(
        "No models could be instantiated; predictions will rely on class‑distribution priors."
    )




## === cell 5
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}




## === cell 6
if loaded_model_keys:
    total_score = sum(
        validation_scores[k] for k in loaded_model_keys if k in validation_scores
    )
    weights = {
        k: validation_scores[k] / total_score
        for k in loaded_model_keys
        if k in validation_scores
    }
else:
    weights = {}




## === cell 7
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        if models_list:
            outputs = [
                weights[model_key]
                * nn.functional.softmax(model(images), dim=1).unsqueeze(0)
                for model_key, model in zip(loaded_model_keys, models_list)
                if model_key in weights
            ]
            if outputs:
                outputs = torch.cat(outputs)  # (n_models, batch, 5)
                weighted_outputs = torch.sum(outputs, dim=0)  # (batch, 5)

                blended = 0.8 * weighted_outputs.cpu().numpy() + 0.2 * class_probs
                all_outputs.extend(blended)
            else:
                batch_size = images.size(0)
                priors = np.tile(class_probs, (batch_size, 1))
                all_outputs.extend(priors)
        else:
            batch_size = images.size(0)
            priors = np.tile(class_probs, (batch_size, 1))
            all_outputs.extend(priors)

all_outputs = np.array(all_outputs)

class_indices = np.arange(5)
expected_vals = np.dot(all_outputs, class_indices)

final_predictions = np.rint(expected_vals).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)




## === cell 8
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
