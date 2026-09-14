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

0.9193205713721068

# 6. Current score

-0.17485

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I import the missing torchvision model definitions and adjust the builder functions to use the standard `densenet121` and `resnet50` models with a 5‑class output layer. This resolves the `NameError`s, allowing the models to be instantiated, predictions to be generated, and a non‑empty submission CSV to be written.'
- What this solution (achieved 0.0) has done: 'I read the official test.csv to obtain the exact list of id_code values, build a dictionary of predictions for each model (so missing images default to 0), and combine the three model outputs in that order.  
I also normalize the Densenet inputs (as was already done for the ResNet models) and replace the hand‑crafted thresholds with a simple rounding to the nearest integer 0‑4, which aligns with the competition’s rating scale.  
These minimal adjustments ensure a complete, correctly ordered submission.csv and should move the quadratic weighted kappa score away from 0 toward the target.'
- What this solution (achieved 0.0) has done: 'I enable ImageNet pretrained weights by calling the model builders with `pretrain=True` (instead of forcing random initialization when custom checkpoints are missing) and I give the stronger densenet model a higher weight in the ensemble (0.5 for densenet, 0.25 each for the two ResNet variants). These minimal tweaks keep the original architecture and training logic intact while giving the predictions a better calibrated starting point, which should move the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.06253) has done: 'I adjust the prediction routine to use the class with the highest average probability (arg‑max) instead of the expectation‑based rating, which aligns better with the discrete 0‑4 scoring required by the competition. This small change keeps the model architecture and training untouched while producing more realistic integer predictions, helping the submission move away from a zero score toward the target.'
- What this solution (achieved -0.17485) has done: 'I adjust the inference so that each model returns a probability vector instead of a single class index, then combine the models by averaging the weighted probabilities and taking an arg‑max to obtain the final integer rating. This keeps the same architectures and loading logic, but uses richer information for ensembling, which should raise the quadratic weighted kappa toward the target while leaving the rest of the pipeline untouched.'

# 9. Code solution

## === cell 0
import os
import re
import math
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True




## === cell 1
__all__ = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth",
    "densenet169": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth",
    "densenet201": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth",
    "densenet161": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth",
    "inceptionv3": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ["inceptionv3"]:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}
for model_name in __all__:
    pretrained_settings[model_name] = {
        "imagenet": {
            "url": model_urls[model_name],
            "input_space": "RGB",
            "input_size": input_sizes[model_name],
            "input_range": [0, 1],
            "mean": means[model_name],
            "std": stds[model_name],
            "num_classes": 1000,
        }
    }


def update_state_dict(state_dict):
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def load_pretrained(model, num_classes, settings):
    assert (
        num_classes == settings["num_classes"]
    ), "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    state_dict = torch.hub.load_state_dict_from_url(settings["url"])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
    return model




## === cell 2
def get_densenet121_gem(pretrain=False):
    """Return a Densenet121 model with a 5‑class linear head."""
    model = models.densenet121(pretrained=pretrain)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 5)
    return model


def get_se_resnet50_gem(pretrain=False):
    """Return a ResNet50 model (used as a surrogate for SE‑ResNet50) with a 5‑class head."""
    model = models.resnet50(pretrained=pretrain)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"

TEST_IMAGE_PATH = os.path.join(BASE_PATH, "test_images")
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))
print(f"Found {len(test_images)} test images on disk.")

test_csv_path = os.path.join(BASE_PATH, "test.csv")
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).tolist()
print(f"Loaded {len(test_ids)} ids from test.csv.")




## === cell 4
def make_predictions(model, test_images, transform, size=256, device=device):
    """
    Run model on each image and return a dict {id_code: prob_tensor (5,)}.
    Using both original and horizontally flipped image and averaging their softmax probabilities.
    """
    model.eval()
    preds = {}
    for im_path in test_images:
        id_code = os.path.splitext(os.path.basename(im_path))[0]
        with Image.open(im_path) as img:
            img = img.convert("RGB")
            img = img.resize((size, size), resample=Image.BILINEAR)
        img_tensor = transform(img).to(device)

        with torch.no_grad():
            out1 = model(img_tensor.unsqueeze(0))
            out2 = model(torch.flip(img_tensor.unsqueeze(0), dims=(3,)))
        probs1 = torch.softmax(out1.squeeze(0), dim=0)
        probs2 = torch.softmax(out2.squeeze(0), dim=0)
        avg_probs = (probs1 + probs2) / 2.0  # shape (5,)
        preds[id_code] = avg_probs.cpu()  # store on CPU for later aggregation
    return preds




## === cell 5
def load_model_safe(builder_fn, weight_path):
    """Instantiate model via builder_fn (with ImageNet pre‑training), load custom weights if present, and set to eval."""
    model = builder_fn(pretrain=True)
    model = model.to(device)
    if os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Failed to load weights from {weight_path}: {e}")
    else:
        print(
            f"Weight file not found at {weight_path}; using ImageNet pretrained weights."
        )
    model.eval()
    return model




## === cell 6
norm_common = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

densenet_weight_path = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
model_densenet = load_model_safe(get_densenet121_gem, densenet_weight_path)
predictions_densenet = make_predictions(
    model_densenet, test_images, norm_common, size=224
)

seresnet_weight_path = "/kaggle/input/seresnet50testpseudo/model10.pth"
model_seresnet = load_model_safe(get_se_resnet50_gem, seresnet_weight_path)
predictions_seresnet = make_predictions(
    model_seresnet, test_images, norm_common, size=256
)

seresnet512_weight_path = "/kaggle/input/seresnet50-512/model30_512.pth"
model_seresnet_512 = load_model_safe(get_se_resnet50_gem, seresnet512_weight_path)
predictions_seresnet_512 = make_predictions(
    model_seresnet_512, test_images, norm_common, size=512
)




## === cell 7
final_predictions = []
weights = {
    "densenet": 0.5,
    "seresnet": 0.25,
    "seresnet512": 0.25,
}
for id_code in test_ids:
    weighted_probs = torch.zeros(5)
    total_w = 0.0
    if id_code in predictions_densenet:
        weighted_probs += weights["densenet"] * predictions_densenet[id_code]
        total_w += weights["densenet"]
    if id_code in predictions_seresnet:
        weighted_probs += weights["seresnet"] * predictions_seresnet[id_code]
        total_w += weights["seresnet"]
    if id_code in predictions_seresnet_512:
        weighted_probs += weights["seresnet512"] * predictions_seresnet_512[id_code]
        total_w += weights["seresnet512"]
    if total_w > 0:
        weighted_probs /= total_w
    pred_class = int(torch.argmax(weighted_probs, dim=0).item())
    final_predictions.append([id_code, pred_class])




## === cell 8
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} (rows: {len(submission)})")
