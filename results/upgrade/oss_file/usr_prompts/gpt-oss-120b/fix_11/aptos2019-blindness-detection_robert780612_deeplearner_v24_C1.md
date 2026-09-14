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

0.9180467915491412

# 6. Current score

0.00485

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the model‑loading steps robust by catching missing weight files and falling back to a simple baseline prediction (value 0) for every test image. This ensures that `predictions_densenet` is always defined, fixes the NameError in cell 8, and guarantees a non‑empty submission CSV. No core model logic is altered, only error handling and a safe fallback are added.'
- What this solution (achieved 0.0) has done: 'I wrap the model‑creation calls in try/except blocks so that a failed download of ImageNet weights does not abort the notebook, and I add a safe fallback that generates zero‑valued predictions when the model cannot be built. This guarantees that `predictions_densenet` is always defined, fixes the NameError in the ensemble step, and ensures the final DataFrame is non‑empty, allowing a valid `submission.csv` to be written.'
- What this solution (achieved 0.0) has done: 'I add a small preprocessing step that reads the training labels and computes the overall mean diagnosis. This mean is then used as a sensible constant fallback value whenever the DenseNet model cannot be loaded, replacing the original zero‑prediction fallback. The rest of the pipeline—including model definitions, prediction logic, ensembling, and submission creation—remains unchanged, ensuring a valid CSV is produced and moving the score away from the all‑zero baseline toward the target.'
- What this solution (achieved 0.14259) has done: 'I add a deterministic brightness‑based fallback that maps each test image to the diagnosis class whose training‑image average brightness is closest. This replaces the simple mean‑constant fallback, giving more varied and informative predictions when the pretrained models cannot be loaded, which should move the Quadratic Weighted Kappa score closer to the target. The rest of the pipeline and model logic remain unchanged.'
- What this solution (achieved 0.14259) has done: 'I add a fallback for the SEResNet model so that when its weights cannot be loaded the script still produces predictions (using the same brightness‑based heuristic). This allows the ensemble step to always average two sets of predictions instead of falling back to a single baseline, giving a modest but consistent improvement toward the target score while keeping the core model logic unchanged.'
- What this solution (achieved 0.14259) has done: 'I fix the SEResNet fallback by using a regular ResNet‑50 (which is available) instead of the undefined `se_resnet50`, and I apply proper ImageNet normalization to the DenseNet inputs. These small fixes let both models run with real pretrained weights rather than falling back to the brightness heuristic, which should move the Quadratic Weighted Kappa score much closer to the target.'
- What this solution (achieved 0.0) has done: 'I add a very light linear‑regression based fallback that learns a relationship between image brightness and the DR grade from the training set, and use it instead of the nearest‑class‑brightness heuristic. This keeps the overall pipeline unchanged while giving the model a more informative signal, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.00485) has done: 'I add a simple quadratic regression fallback (using the training image brightness‑to‑diagnosis relationship) to replace the current constant/linear fallback, because it gives a richer mapping and can improve the quadratic weighted‑kappa score while keeping the overall pipeline unchanged. The new fallback is used when the pretrained models cannot be loaded, so the submission contain more informative predictions and move the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV I/O
import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from PIL import Image, ImageFile
from torchvision import transforms
import types
import re



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
    "vgg19",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth",
    "densenet169": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth",
    "densenet201": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth",
    "densenet161": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth",
    "inceptionv3": "https://download.pytorch.org/models/inception_v_3_google-1a9a5a14.pth",
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
...




## === cell 2
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
    """
    Use a regular ResNet‑50 (which is available) in place of the undefined SE‑ResNet‑50.
    This keeps the original architecture intent while ensuring the model can be built.
    """
    if pretrain == "imagenet":
        model = resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))



## === cell 4
train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_path)



## === cell 5
train_image_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
brightness_sums = {c: 0.0 for c in range(5)}
brightness_counts = {c: 0 for c in range(5)}

train_brightness_vals = []
train_labels_vals = []

for _, row in train_df.iterrows():
    img_path = os.path.join(train_image_dir, f"{row['id_code']}.png")
    try:
        img = Image.open(img_path).convert("L")  # convert to grayscale
        arr = np.asarray(img, dtype=np.float32)
        mean_brightness = arr.mean()
        label = int(row["diagnosis"])
        brightness_sums[label] += mean_brightness
        brightness_counts[label] += 1

        train_brightness_vals.append(mean_brightness)
        train_labels_vals.append(label)
    except Exception:
        continue  # skip missing or corrupted images

class_mean_brightness = {}
overall_mean = train_df["diagnosis"].mean()
for c in range(5):
    if brightness_counts[c] > 0:
        class_mean_brightness[c] = brightness_sums[c] / brightness_counts[c]
    else:
        class_mean_brightness[c] = overall_mean  # safety fallback

if len(train_brightness_vals) >= 2:
    a_lin, b_lin = np.polyfit(train_brightness_vals, train_labels_vals, 1)
else:
    a_lin, b_lin = 0.0, overall_mean

if len(train_brightness_vals) >= 3:
    quad_coeffs = np.polyfit(
        train_brightness_vals, train_labels_vals, 2
    )  # coeffs: ax^2 + bx + c
else:
    quad_coeffs = None


def brightness_based_fallback(image_path):
    """Return the diagnosis class whose average brightness is closest to the image brightness."""
    try:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        img_brightness = arr.mean()
    except Exception:
        return int(round(overall_mean))
    best_class = min(
        class_mean_brightness.items(), key=lambda kv: abs(kv[1] - img_brightness)
    )[0]
    return int(best_class)


def regression_fallback(image_path):
    """
    Predict diagnosis using a quadratic regression of brightness → label.
    Falls back to the linear model if the quadratic coefficients are unavailable.
    """
    try:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        img_brightness = arr.mean()
    except Exception:
        pred = overall_mean
    else:
        if quad_coeffs is not None:
            pred = np.polyval(quad_coeffs, img_brightness)
        else:
            pred = a_lin * img_brightness + b_lin
    pred = np.clip(pred, 0, 4)
    return int(round(pred))




## === cell 6
def make_predictions(model, test_images, transforms, size=256, device=device):
    predictions = []
    for im_path in test_images:
        try:
            image = Image.open(im_path).convert("RGB")
        except Exception:
            continue  # skip unreadable files
        image = image.resize((size, size), resample=Image.BILINEAR)
        image = transforms(image).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))
        final_prediction = (output.item() + output_flip.item()) / 2.0
        predictions.append(
            (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
        )
    return predictions




## === cell 7
dense_net_path = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
predictions_densenet = None
try:
    model_dense = get_densenet121_gem(pretrain="imagenet")
    try:
        state_dict_dense = torch.load(dense_net_path, map_location=device)
        model_dense.load_state_dict(state_dict_dense)
    except Exception as e:
        print(
            f"Warning: fine‑tuned DenseNet weights not loaded ({e}); using ImageNet weights."
        )
    model_dense.to(device)
    norm_dense = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_densenet = make_predictions(
        model_dense, test_images, norm_dense, size=224
    )
except Exception as e:
    print(f"Warning: DenseNet model creation failed ({e}); using regression fallback.")
    predictions_densenet = [
        (os.path.splitext(os.path.basename(p))[0], regression_fallback(p))
        for p in test_images
    ]



## === cell 8
predictions_seresnet = None
try:
    seresnet_path = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
    model_seres = get_se_resnet50_gem(pretrain="imagenet")
    state_dict_seres = torch.load(seresnet_path, map_location=device)
    model_seres.load_state_dict(state_dict_seres)
    model_seres.to(device)
    norm_seres = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet = make_predictions(
        model_seres, test_images, norm_seres, size=256
    )
except Exception as e:
    print(f"Warning: SEResNet model creation failed ({e}); using regression fallback.")
    predictions_seresnet = [
        (os.path.splitext(os.path.basename(p))[0], regression_fallback(p))
        for p in test_images
    ]



## === cell 9
final_predictions = []
if predictions_seresnet is not None:
    for (id1, pred1), (id2, pred2) in zip(predictions_densenet, predictions_seresnet):
        if id1 != id2:
            raise ValueError(f"Mismatched IDs: {id1} vs {id2}")
        final_predictions.append([id1, (pred1 + pred2) / 2.0])
else:
    final_predictions = [[id_code, pred] for id_code, pred in predictions_densenet]



## === cell 10
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].astype(float)
submission["diagnosis"] = submission["diagnosis"].clip(0, 4).round().astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
