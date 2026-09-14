# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from torchvision.models import densenet121, resnet50  # added import for actual models
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
    Build a ResNet‑50 (available in torchvision) and replace its pooling with GeM.
    Adjust the final fully‑connected layer to output a single regression value.
    """
    if pretrain == "imagenet":
        model = resnet50(pretrained=True)
    else:
        model = resnet50(pretrained=False)

    model.avgpool = GeM()

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_densenet121_gem(pretrain):
    """
    Build a DenseNet‑121 and replace its classifier with a single scalar output.
    The original pooling is retained (it works well with ImageNet weights).
    """
    if pretrain == "imagenet":
        model = densenet121(pretrained=True)
    else:
        model = densenet121(pretrained=False)

    model.classifier = nn.Linear(model.classifier.in_features, 1)
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
def make_predictions(model, image_paths, transforms, size=256, device=device):
    predictions = []
    model.eval()
    for im_path in image_paths:
        try:
            image = Image.open(im_path).convert("RGB")
        except Exception:
            continue  # skip unreadable files
        image = image.resize((size, size), resample=Image.BILINEAR)
        image = transforms(image).to(device)
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
    train_image_paths = [
        os.path.join(train_image_dir, f"{row['id_code']}.png")
        for _, row in train_df.iterrows()
    ]
    train_preds_dense = [
        pred
        for _, pred in make_predictions(
            model_dense, train_image_paths, norm_dense, size=224
        )
    ]
    train_labels = train_df["diagnosis"].tolist()
    if len(train_preds_dense) >= 3:
        coeffs_dense = np.polyfit(train_preds_dense, train_labels, 2)

        def calibrate_dense(x):
            return np.clip(np.polyval(coeffs_dense, x), 0, 4)

    else:
        a_dense, b_dense = (
            np.polyfit(train_preds_dense, train_labels, 1)
            if len(train_preds_dense) > 0
            else (1.0, 0.0)
        )

        def calibrate_dense(x):
            return np.clip(a_dense * x + b_dense, 0, 4)

    raw_preds_dense = make_predictions(model_dense, test_images, norm_dense, size=224)
    predictions_densenet = [
        (img_id, calibrate_dense(pred)) for img_id, pred in raw_preds_dense
    ]
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
    train_preds_seres = [
        pred
        for _, pred in make_predictions(
            model_seres, train_image_paths, norm_seres, size=256
        )
    ]
    if len(train_preds_seres) >= 3:
        coeffs_seres = np.polyfit(train_preds_seres, train_labels, 2)

        def calibrate_seres(x):
            return np.clip(np.polyval(coeffs_seres, x), 0, 4)

    else:
        a_seres, b_seres = (
            np.polyfit(train_preds_seres, train_labels, 1)
            if len(train_preds_seres) > 0
            else (1.0, 0.0)
        )

        def calibrate_seres(x):
            return np.clip(a_seres * x + b_seres, 0, 4)

    raw_preds_seres = make_predictions(model_seres, test_images, norm_seres, size=256)
    predictions_seresnet = [
        (img_id, calibrate_seres(pred)) for img_id, pred in raw_preds_seres
    ]
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
