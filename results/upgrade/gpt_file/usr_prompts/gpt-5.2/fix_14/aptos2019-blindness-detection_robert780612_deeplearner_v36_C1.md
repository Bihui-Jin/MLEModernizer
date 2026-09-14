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

0.9160158399570948

# 6. Current score

0.52026

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.02106) has done: 'I fix the pipeline so it no longer hard-fails when the external checkpoint datasets are absent (common in a fresh Kaggle environment), while keeping the existing ensemble/inference logic intact when checkpoints do exist. Specifically, I (1) detect available checkpoints under `/kaggle/input` more robustly, (2) add a safe fallback model (torchvision ResNet18 pretrained on ImageNet) to always generate predictions end-to-end, and (3) fix the id/image alignment by iterating over `test_df["id_code"]` and constructing image paths directly (instead of relying on `glob` order). This makes the submission deterministic, valid, and runnable; if your original checkpoints are present, they still be used as before and should yield the intended score behavior.'
- What this solution (achieved -0.00793) has done: 'Most of the timeout is from per-image inference done twice (original + flip) in a Python loop with repeated PIL decode/resize/ToTensor and no DataLoader batching, plus (in fallback) 5-fold training + per-image validation inference. The core logic is preserved, but we move image loading/resize/transform into a multi-worker Dataset/DataLoader, batch the model forward passes (still doing TTA flip), and enable pinned-memory + non_blocking GPU transfers to reduce overhead. We also make the kappa/threshold search vectorized (same semantics), and set deterministic seeds/flags so results remain stable. These changes reduce Python overhead and increase GPU utilization without changing the model architectures, loss, thresholds, or evaluation logic.'
- What this solution (achieved -0.00793) has done: 'Your current negative QWK strongly suggests the predictions are on the wrong scale for the fixed thresholds (0.75/1.5/2.5/3.6), especially in fallback mode where the ResNet18 head is not calibrated and can output values far outside 0–4. To move the score toward the target with minimal core-logic change, I keep the exact same model/inference/TTA/threshold-search logic, but (1) learn thresholds on OOF predictions for *any* mode (not just fallback) and (2) ensure the test predictions are on the same scale as the OOF predictions by fitting a simple linear calibration (a,b) on OOF (least-squares) and applying it to test preds before thresholding. This is a small post-processing fix aligned with the QWK metric and should substantially improve from -0.00793 without changing architectures, losses, or training loops. The script still writes a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target (gap ≈ -0.924), so we need a real improvement rather than runtime-only tweaks. The main issue is that you’re calibrating/optimizing thresholds using OOF predictions from a *different model* (freshly trained ResNet18) than the one producing test predictions (your checkpoint ensemble or fallback), which breaks the scale alignment and can yield near-random/negative QWK. With minimal core-logic change, I keep your existing ensemble + TTA inference intact, but compute OOF predictions using the *same inference pipeline* (same models/checkpoints, same resizing, same normalization) and then learn a single linear calibration + thresholds from those aligned OOF predictions. This makes the post-processing consistent with what you actually submit, and should move QWK sharply upward toward your target without changing architectures, losses, or the inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission-format/typing issue rather than model quality, because the pipeline already calibrates + thresholds but can still silently produce non-integer/object labels if the per-batch `ids` come out as tuples (DataLoader collation) and don’t match `test_ids` exactly. I make ID handling deterministic by returning the raw `id_code` from the dataset (not parsing from path) and forcing `ids` to plain Python strings during prediction collection, which fixes the common “all predictions missing -> fallback/zeros/NaNs” failure mode without changing the model logic. I also ensure the final `diagnosis` column is strict int in [0..4] and that the submission row order exactly matches `test.csv`. These are minimal changes aimed at turning the current 0.0 into a meaningful QWK so it moves toward your 0.916 target.'
- What this solution (achieved 0.52026) has done: 'The timeout is dominated by redundant image decoding/resizing work: you run full test inference, then you re-run inference over the entire training set again just to get OOF predictions for threshold calibration (and the fallback training path is even more expensive if checkpoints are missing). The key speedup is to cache decoded+resized tensors on disk (per image size) and reuse them across all inference passes and across models, which is exactly equivalent to re-decoding/resizing each time. Additionally, we remove an unnecessary full-image existence scan for train images (which does 3k+ filesystem stats) and replace it with a fast spot-check, and we switch the inference DataLoader to a deterministic, faster collate with pinned memory and larger prefetch while keeping seeds/determinism intact. Core model logic, TTA (flip), calibration, threshold search, and evaluation semantics remain unchanged.'
- What this solution (achieved 0.52026) has done: 'Your current 0.52026 is far below the 0.916 target (higher-is-better), so we need a real but still minimal change that improves QWK without changing the model/training core. The biggest issue is that your “OOF” predictions are not actually out-of-fold: `compute_oof_predictions_fast()` predicts once using models that have (directly or indirectly) seen all training data, then copies those same predictions into every fold, which makes threshold/calibration overfit and generalize poorly to test. I replace that with a true OOF computation that, for each fold, trains the fallback ResNet18 (same architecture/loss/epochs/TTA as you already use) on the fold’s training split and predicts only the fold’s validation split; this produces properly-aligned OOF predictions for calibration/thresholds. When external checkpoints exist, the pipeline remains unchanged (we keep your existing inference/ensemble), but we *not* run the expensive/incorrect OOF routine; instead we keep the original fixed thresholds in that case to avoid a bad calibration step.'
- What this solution (achieved 0.52026) has done: 'Your 0.52026 score is far below the 0.916 target, and the biggest quality blocker in your current pipeline is that when external checkpoints exist you *skip* any calibration/threshold optimization and fall back to fixed thresholds that may not match your ensemble’s raw output scale. I keep the exact same models, TTA (flip), ensembling, and threshold-search logic, but compute out-of-fold (OOF) predictions on the training set using the *same predictor pipeline you use for test*, then fit the same linear calibration (a,b) and optimize thresholds on those aligned OOF predictions. To stay within runtime, this OOF is done with a small 5-fold loop (predict only each validation fold) and reuses your existing tensor caching + batched DataLoader inference, so it’s much cheaper than retraining and directly targets QWK. Finally, I apply the learned calibration + thresholds to the test ensemble predictions and write a valid `submission.csv` in the required order.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import re

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
    state_dict = model_zoo.load_url(settings["url"])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
    return model


def modify_alexnet(model):
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def alexnet(num_classes=1000, pretrained="imagenet"):
    model = models.alexnet(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["alexnet"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model


def modify_densenets(model):
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet121(num_classes=1000, pretrained="imagenet"):
    model = models.densenet121(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    model = models.densenet169(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    model = models.densenet201(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    model = models.densenet161(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained="imagenet"):
    model = models.inception_v3(weights=None, aux_logits=True)
    if pretrained is not None:
        settings = pretrained_settings["inceptionv3"][pretrained]
        model = load_pretrained(model, num_classes, settings)

    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        x = self.Conv2d_1a_3x3(input)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def logits(self, features):
        x = F.avg_pool2d(features, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        if self.training and self.aux_logits:
            aux = self._out_aux
            self._out_aux = None
            return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def modify_resnets(model):
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes=1000, pretrained="imagenet"):
    model = models.resnet18(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    model = models.resnet34(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    model = models.resnet50(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    model = models.resnet101(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    model = models.resnet152(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet152"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def modify_squeezenets(model):
    model.dropout = model.classifier[0]
    model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    del model.classifier

    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def squeezenet1_0(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_0(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_1(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_1"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def modify_vggs(model):
    model._features = model.features
    del model.features
    model.linear0 = model.classifier[0]
    model.relu0 = model.classifier[1]
    model.dropout0 = model.classifier[2]
    model.linear1 = model.classifier[3]
    model.relu1 = model.classifier[4]
    model.dropout1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def vgg11(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg19_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model




## === cell 2
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo

__all__ = [
    "SENet",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
]

pretrained_settings = {
    "senet154": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet50": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet101": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet152": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnext50_32x4d": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnext101_32x4d": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
}


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2,
            planes * 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        base_width=4,
    ):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        groups,
        reduction,
        dropout_p=0.2,
        inplanes=128,
        input_3x3=True,
        downsample_kernel_size=3,
        downsample_padding=1,
        num_classes=1000,
    ):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ("conv1", nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(64)),
                ("relu2", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ("bn3", nn.BatchNorm2d(inplanes)),
                ("relu3", nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv2d(
                        3, inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                ),
                ("bn1", nn.BatchNorm2d(inplanes)),
                ("relu1", nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(("pool", nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        groups,
        reduction,
        stride=1,
        downsample_kernel_size=1,
        downsample_padding=0,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=downsample_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert (
        num_classes == settings["num_classes"]
    ), "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    model.load_state_dict(model_zoo.load_url(settings["url"]))
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]


def senet154(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEBottleneck,
        [3, 8, 36, 3],
        groups=64,
        reduction=16,
        dropout_p=0.2,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["senet154"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet50"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 4, 23, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet101"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 8, 36, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet152"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnext50_32x4d"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnext101_32x4d"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model




## === cell 3
import sys

sys.path.append("/kaggle/working/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
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


class Resnet18OrdinalHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=w)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)




## === cell 4
import os
from glob import glob

import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["id_code"].astype(str).tolist()

test_images = [os.path.join(TEST_IMAGE_PATH, f"{_id}.png") for _id in test_ids]
missing_imgs = [p for p in test_images if not os.path.exists(p)]
if missing_imgs:
    raise FileNotFoundError(
        f"Missing {len(missing_imgs)} test images (e.g. {missing_imgs[0]})."
    )



## === cell 5
from sklearn.model_selection import StratifiedKFold


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    yt = y_true[mask]
    yp = y_pred[mask]
    O = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes).astype(
        np.float64
    )
    O = O.reshape(n_classes, n_classes)

    act_hist = np.bincount(y_true.clip(0, n_classes - 1), minlength=n_classes).astype(
        np.float64
    )
    pred_hist = np.bincount(y_pred.clip(0, n_classes - 1), minlength=n_classes).astype(
        np.float64
    )
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    i = np.arange(n_classes, dtype=np.float64)
    W = ((i[:, None] - i[None, :]) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def apply_thresholds(x, thresholds):
    t = np.asarray(thresholds, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.int64)
    out[x >= t[0]] = 1
    out[x >= t[1]] = 2
    out[x >= t[2]] = 3
    out[x >= t[3]] = 4
    return out


def optimize_thresholds(
    y_true, x_pred, init_thresholds=(0.75, 1.5, 2.5, 3.6), n_iter=4
):
    y_true = np.asarray(y_true, dtype=int)
    x_pred = np.asarray(x_pred, dtype=np.float64)

    t = np.array(init_thresholds, dtype=np.float64)

    lo = np.percentile(x_pred, 1)
    hi = np.percentile(x_pred, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(x_pred.min()), float(x_pred.max() + 1e-6)
    span = hi - lo + 1e-6

    best_t = t.copy()
    best_score = quadratic_weighted_kappa(y_true, apply_thresholds(x_pred, best_t))

    for it in range(n_iter):
        step = span / (25 * (2**it))
        for k in range(4):
            left = lo if k == 0 else best_t[k - 1] + 1e-6
            right = hi if k == 3 else best_t[k + 1] - 1e-6
            if left >= right:
                continue
            candidates = np.arange(
                max(left, best_t[k] - 5 * step),
                min(right, best_t[k] + 5 * step) + 0.5 * step,
                step,
            )
            for cand in candidates:
                t_try = best_t.copy()
                t_try[k] = cand
                score = quadratic_weighted_kappa(
                    y_true, apply_thresholds(x_pred, t_try)
                )
                if score > best_score:
                    best_score = score
                    best_t = t_try
    return best_t, best_score


def fit_linear_calibration(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 1e-12:
        a = 1.0
    else:
        a = ((x - x_mean) * (y - y_mean)).sum() / denom
    b = y_mean - a * x_mean
    return float(a), float(b)




## === cell 6
from torch.utils.data import Dataset, DataLoader

import io
import hashlib
from torch.utils.data import get_worker_info

_TENSOR_CACHE_DIR = "/kaggle/working/img_tensor_cache_v1"
os.makedirs(_TENSOR_CACHE_DIR, exist_ok=True)

_WORKER_BYTES_CACHE = {}  # worker-local: path -> bytes


def _get_img_bytes_worker_local(path: str) -> bytes:
    wi = get_worker_info()
    cache = _WORKER_BYTES_CACHE if wi is None else _WORKER_BYTES_CACHE
    b = cache.get(path)
    if b is None:
        with open(path, "rb") as f:
            b = f.read()
        cache[path] = b
    return b


def _load_resized_rgb_from_bytes(path, size: int):
    b = _get_img_bytes_worker_local(path)
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img = img.resize((int(size), int(size)), resample=Image.BILINEAR)
    return img


def _cache_key(path: str, size: int) -> str:
    h = hashlib.md5((str(path) + "|" + str(int(size))).encode("utf-8")).hexdigest()
    return os.path.join(_TENSOR_CACHE_DIR, f"{h}.pt")


def _get_resized_tensor_cached(path: str, size: int) -> torch.Tensor:
    ck = _cache_key(path, size)
    try:
        t = torch.load(ck, map_location="cpu", weights_only=True)
        if isinstance(t, torch.Tensor) and t.ndim == 3:
            return t
    except Exception:
        pass
    img = _load_resized_rgb_from_bytes(path, size)
    t = transforms.ToTensor()(img)  # float32 CHW in [0,1]
    tmp = ck + ".tmp"
    torch.save(t, tmp)
    os.replace(tmp, ck)
    return t


class _InferDataset(Dataset):
    def __init__(self, image_paths, ids, tfm, size):
        self.image_paths = image_paths
        self.ids = [str(i) for i in ids]
        self.tfm = tfm
        self.size = int(size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        x = _get_resized_tensor_cached(p, self.size)
        x = self.tfm(x) if self.tfm is not None else x
        _id = self.ids[idx]
        return _id, x


def _default_collate_infer(batch):
    ids, xs = zip(*batch)
    return list(ids), torch.stack(xs, dim=0)


def make_predictions(
    model, test_images, ids, transforms, size=256, device=device, batch_size=32
):
    ds = _InferDataset(test_images, ids, transforms, size)
    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 8 else 2
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        collate_fn=_default_collate_infer,
    )
    model.eval()
    predictions = []
    with torch.inference_mode():
        for ids_b, xb in loader:
            xb = xb.to(device, non_blocking=True)
            xb_flip = torch.flip(xb, dims=(3,))
            x2 = torch.cat([xb, xb_flip], dim=0)
            out2 = model(x2).squeeze(1)
            out = out2[: xb.shape[0]]
            out_flip = out2[xb.shape[0] :]
            final = ((out + out_flip) * 0.5).detach().cpu().numpy()
            predictions.extend([(str(i), float(p)) for i, p in zip(ids_b, final)])
    return predictions


def _load_state_dict_flexible(model, ckpt_path, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    return model


def _find_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


def _find_any_checkpoint_by_patterns(base="/kaggle/input", patterns=(".pth", ".pt")):
    found = []
    for pat in patterns:
        found.extend(glob(os.path.join(base, "**", f"*{pat}"), recursive=True))
    return sorted(set(found))


def build_predictors(
    dense_path, seres_pseudo_path, seres_512_path, device, fallback_ckpt_path=None
):
    predictors = []

    if dense_path is not None:
        model_dense = get_densenet121_gem(pretrain=False).to(device)
        _load_state_dict_flexible(model_dense, dense_path, map_location=device)
        tfm_dense = transforms.Compose([])  # ToTensor already applied by cache

        def _pred_dense(image_paths, ids):
            return make_predictions(
                model_dense,
                image_paths,
                ids,
                tfm_dense,
                size=224,
                device=device,
                batch_size=32,
            )

        predictors.append(_pred_dense)

    if seres_pseudo_path is not None:
        model_seres_pseudo = get_se_resnet50_gem(pretrain=False).to(device)
        _load_state_dict_flexible(
            model_seres_pseudo, seres_pseudo_path, map_location=device
        )
        tfm_seres = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def _pred_seres_pseudo(image_paths, ids):
            return make_predictions(
                model_seres_pseudo,
                image_paths,
                ids,
                tfm_seres,
                size=256,
                device=device,
                batch_size=16,
            )

        predictors.append(_pred_seres_pseudo)

    if seres_512_path is not None:
        model_seres_512 = get_se_resnet50_gem(pretrain=False).to(device)
        _load_state_dict_flexible(model_seres_512, seres_512_path, map_location=device)
        tfm_seres512 = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def _pred_seres_512(image_paths, ids):
            return make_predictions(
                model_seres_512,
                image_paths,
                ids,
                tfm_seres512,
                size=512,
                device=device,
                batch_size=4,
            )

        predictors.append(_pred_seres_512)

    if len(predictors) == 0:
        model_fb = Resnet18OrdinalHead(pretrained=True).to(device)
        if fallback_ckpt_path is not None and os.path.exists(fallback_ckpt_path):
            _load_state_dict_flexible(model_fb, fallback_ckpt_path, map_location=device)
        tfm_fb = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def _pred_fallback(image_paths, ids):
            return make_predictions(
                model_fb,
                image_paths,
                ids,
                tfm_fb,
                size=224,
                device=device,
                batch_size=64,
            )

        predictors.append(_pred_fallback)

    return predictors


def ensemble_predict_from_predictors(predictors, image_paths, ids_in_order):
    from collections import defaultdict

    predictions_list = []
    for fn in predictors:
        predictions_list.append(fn(image_paths, ids_in_order))

    preds_by_id = defaultdict(list)
    for preds in predictions_list:
        for _id, p in preds:
            preds_by_id[str(_id)].append(p)

    final_predictions = []
    missing_in_pred = 0
    for _id in ids_in_order:
        _id = str(_id)
        if _id not in preds_by_id:
            missing_in_pred += 1
            continue
        final_predictions.append([_id, float(np.mean(preds_by_id[_id]))])

    if missing_in_pred > 0:
        raise RuntimeError(
            f"{missing_in_pred} ids missing predictions; cannot create a valid submission."
        )
    return final_predictions


def compute_true_oof_predictions_fallback_resnet18(
    train_image_paths, y, n_splits=5, seed=42
):
    y = np.asarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=np.float64)

    tfm = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class _TrainDatasetLocal(Dataset):
        def __init__(self, image_paths, labels, tfm, size):
            self.image_paths = image_paths
            self.labels = labels.astype(np.float32)
            self.tfm = tfm
            self.size = int(size)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            p = self.image_paths[idx]
            x = _get_resized_tensor_cached(p, self.size)
            x = self.tfm(x) if self.tfm is not None else x
            yy = self.labels[idx]
            return x, torch.tensor([yy], dtype=torch.float32)

    def _default_collate_train(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _train_one_fold(train_paths, train_y, val_paths, fold_seed):
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

        model = Resnet18OrdinalHead(pretrained=True).to(device)
        train_ds = _TrainDatasetLocal(train_paths, train_y, tfm=tfm, size=224)
        val_ds = _TrainDatasetLocal(
            val_paths, np.zeros(len(val_paths), dtype=np.float32), tfm=tfm, size=224
        )

        num_workers = 4 if os.cpu_count() and os.cpu_count() >= 8 else 2
        train_loader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=_default_collate_train,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=_default_collate_train,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        epochs = 2  # keep identical to your existing fallback training

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        val_pred = []
        with torch.inference_mode():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                xb_flip = torch.flip(xb, dims=(3,))
                x2 = torch.cat([xb, xb_flip], dim=0)
                out2 = model(x2).squeeze(1)
                out = out2[: xb.shape[0]]
                out_flip = out2[xb.shape[0] :]
                pred = ((out + out_flip) * 0.5).detach().cpu().numpy()
                val_pred.append(pred)
        return np.concatenate(val_pred, axis=0).astype(np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        tr_paths = [train_image_paths[i] for i in tr_idx]
        tr_y = y[tr_idx]
        va_paths = [train_image_paths[i] for i in va_idx]
        va_pred = _train_one_fold(tr_paths, tr_y, va_paths, fold_seed=seed + fold)
        oof_pred[va_idx] = va_pred

    return oof_pred


def compute_oof_predictions_from_predictors(
    predictors, image_paths, ids, y, n_splits=5, seed=42
):
    y = np.asarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(ids), dtype=np.float64)

    ids = [str(i) for i in ids]
    image_paths = list(image_paths)

    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        va_ids = [ids[i] for i in va_idx]
        va_paths = [image_paths[i] for i in va_idx]

        fold_preds = ensemble_predict_from_predictors(
            predictors=predictors, image_paths=va_paths, ids_in_order=va_ids
        )
        pred_map = {str(_id): float(p) for _id, p in fold_preds}

        missing = [i for i, _id in zip(va_idx, va_ids) if _id not in pred_map]
        if missing:
            raise RuntimeError(
                f"OOF fold {fold}: missing {len(missing)} predictions; cannot calibrate."
            )

        for i, _id in zip(va_idx, va_ids):
            oof_pred[i] = pred_map[str(_id)]

    return oof_pred




## === cell 7
DENSE_CANDIDATES = [
    "/kaggle/input/densenet121/model_densenet121_bs64_30.pth",
    "/kaggle/input/densenet121/model.pth",
]
SERES_PSEUDO_CANDIDATES = [
    "/kaggle/input/seresnet50testpseudo/model10.pth",
]
SERES_512_CANDIDATES = [
    "/kaggle/input/seresnet50-512/model30_512.pth",
]

dense_path = _find_first_existing(DENSE_CANDIDATES)
seres_pseudo_path = _find_first_existing(SERES_PSEUDO_CANDIDATES)
seres_512_path = _find_first_existing(SERES_512_CANDIDATES)

available = {
    "densenet121": dense_path,
    "seresnet50_pseudo": seres_pseudo_path,
    "seresnet50_512": seres_512_path,
}
print("Checkpoint availability:", available)

if all(v is None for v in available.values()):
    any_ckpts = _find_any_checkpoint_by_patterns("/kaggle/input")
    print(
        f"No expected checkpoints found. Total .pth/.pt files under /kaggle/input: {len(any_ckpts)}"
    )



## === cell 8
import time
from torch.utils.data import Dataset, DataLoader

FALLBACK_CKPT_PATH = "/kaggle/working/fallback_resnet18_foldavg.pth"


class _TrainDataset(Dataset):
    def __init__(self, image_paths, labels, tfm, size):
        self.image_paths = image_paths
        self.labels = labels.astype(np.float32)
        self.tfm = tfm
        self.size = int(size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        x = _get_resized_tensor_cached(p, self.size)
        x = self.tfm(x) if self.tfm is not None else x
        y = self.labels[idx]
        return x, torch.tensor([y], dtype=torch.float32)


def _default_collate_train(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _train_one_fold_resnet18(train_paths, train_y, val_paths, val_y, fold_seed=42):
    torch.manual_seed(fold_seed)
    torch.cuda.manual_seed_all(fold_seed)

    model = Resnet18OrdinalHead(pretrained=True).to(device)

    tfm = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = _TrainDataset(train_paths, train_y, tfm=tfm, size=224)
    val_ds = _TrainDataset(val_paths, val_y, tfm=tfm, size=224)

    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 8 else 2
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_default_collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_default_collate_train,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    val_pred = []
    with torch.inference_mode():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            xb_flip = torch.flip(xb, dims=(3,))
            x2 = torch.cat([xb, xb_flip], dim=0)
            out2 = model(x2).squeeze(1)
            out = out2[: xb.shape[0]]
            out_flip = out2[xb.shape[0] :]
            pred = ((out + out_flip) * 0.5).detach().cpu().numpy()
            val_pred.append(pred)
    val_pred = np.concatenate(val_pred, axis=0).astype(np.float64)
    return model.state_dict(), val_pred


def _maybe_train_fallback_if_needed():
    if not all(v is None for v in available.values()):
        return None  # external ckpts exist, do nothing
    if os.path.exists(FALLBACK_CKPT_PATH):
        return FALLBACK_CKPT_PATH  # already trained in this run environment

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_ids_local = train_df["id_code"].astype(str).tolist()
    y_local = train_df["diagnosis"].astype(int).values
    train_images_local = [
        os.path.join(TRAIN_IMAGE_PATH, f"{_id}.png") for _id in train_ids_local
    ]

    skf_local = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_pred_local = np.zeros(len(train_ids_local), dtype=np.float64)

    state_dicts = []
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(
        skf_local.split(np.zeros(len(y_local)), y_local), start=1
    ):
        print(
            f"Training fallback fold {fold}/5: train {len(tr_idx)}, val {len(va_idx)}"
        )
        tr_paths = [train_images_local[i] for i in tr_idx]
        tr_y = y_local[tr_idx]
        va_paths = [train_images_local[i] for i in va_idx]
        va_y = y_local[va_idx]
        sd, va_pred = _train_one_fold_resnet18(
            tr_paths, tr_y, va_paths, va_y, fold_seed=SEED + fold
        )
        state_dicts.append(sd)
        oof_pred_local[va_idx] = va_pred
        print(f"  fold {fold} done in {time.time()-t0:.1f}s")

    avg_state = {}
    for k in state_dicts[0].keys():
        avg_state[k] = sum(sd[k].float() for sd in state_dicts) / float(
            len(state_dicts)
        )
    torch.save(avg_state, FALLBACK_CKPT_PATH)
    print("Saved trained fallback checkpoint to:", FALLBACK_CKPT_PATH)

    np.save("/kaggle/working/fallback_oof_pred.npy", oof_pred_local)
    np.save("/kaggle/working/fallback_y.npy", y_local.astype(np.int64))
    return FALLBACK_CKPT_PATH


fallback_ckpt = _maybe_train_fallback_if_needed()
predictors = build_predictors(
    dense_path,
    seres_pseudo_path,
    seres_512_path,
    device=device,
    fallback_ckpt_path=fallback_ckpt,
)



## === cell 9
final_predictions = ensemble_predict_from_predictors(
    predictors=predictors,
    image_paths=test_images,
    ids_in_order=test_ids,
)
print("Ensembled predictions:", len(final_predictions), "rows")



## === cell 10
learned_thresholds = None
calib_a, calib_b = 1.0, 0.0

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["id_code"].astype(str).tolist()
y = train_df["diagnosis"].astype(int).values

train_images = [os.path.join(TRAIN_IMAGE_PATH, f"{_id}.png") for _id in train_ids]

for p in (train_images[0], train_images[len(train_images) // 2], train_images[-1]):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing train image: {p}")

if all(v is None for v in available.values()):
    if os.path.exists("/kaggle/working/fallback_oof_pred.npy"):
        oof_pred = np.load("/kaggle/working/fallback_oof_pred.npy").astype(np.float64)
        y_for_cal = np.load("/kaggle/working/fallback_y.npy").astype(np.int64)
        assert len(oof_pred) == len(y_for_cal) == len(train_ids)
    else:
        t0 = time.time()
        oof_pred = compute_true_oof_predictions_fallback_resnet18(
            train_image_paths=train_images, y=y, n_splits=5, seed=SEED
        )
        y_for_cal = y.astype(np.int64)
        print(f"Computed TRUE OOF fallback predictions in {time.time()-t0:.1f}s")

    calib_a, calib_b = fit_linear_calibration(oof_pred, y_for_cal.astype(np.float64))
    oof_cal = calib_a * oof_pred + calib_b

    learned_thresholds, best_oof_kappa = optimize_thresholds(
        y_for_cal, oof_cal, init_thresholds=(0.75, 1.5, 2.5, 3.6), n_iter=5
    )
    print("Calibration (a,b):", calib_a, calib_b)
    print("Learned thresholds (OOF-calibrated):", learned_thresholds.tolist())
    print("Best OOF QWK (calibrated):", float(best_oof_kappa))
else:
    t0 = time.time()
    oof_pred = compute_oof_predictions_from_predictors(
        predictors=predictors,
        image_paths=train_images,
        ids=train_ids,
        y=y,
        n_splits=5,
        seed=SEED,
    )
    print(f"Computed predictor-aligned OOF predictions in {time.time()-t0:.1f}s")

    calib_a, calib_b = fit_linear_calibration(oof_pred, y.astype(np.float64))
    oof_cal = calib_a * oof_pred + calib_b
    learned_thresholds, best_oof_kappa = optimize_thresholds(
        y, oof_cal, init_thresholds=(0.75, 1.5, 2.5, 3.6), n_iter=5
    )
    print("Calibration (a,b):", calib_a, calib_b)
    print("Learned thresholds (OOF-calibrated):", learned_thresholds.tolist())
    print("Best OOF QWK (calibrated):", float(best_oof_kappa))



## === cell 11
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["id_code"] = submission["id_code"].astype(str)

submission["diagnosis"] = (calib_a * submission["diagnosis"].values + calib_b).astype(
    np.float64
)

if learned_thresholds is not None:
    submission["diagnosis"] = apply_thresholds(
        submission["diagnosis"].values, learned_thresholds
    ).astype(np.int64)
else:
    submission.loc[submission.diagnosis < 0.75, "diagnosis"] = 0
    submission.loc[
        (0.75 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
    ] = 1
    submission.loc[
        (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
    ] = 2
    submission.loc[
        (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.6), "diagnosis"
    ] = 3
    submission.loc[3.6 <= submission.diagnosis, "diagnosis"] = 4
    submission["diagnosis"] = submission["diagnosis"].astype(np.int64)

submission["diagnosis"] = submission["diagnosis"].clip(0, 4).astype(np.int64)
submission = submission.set_index("id_code").reindex(test_ids).reset_index()
if submission["diagnosis"].isna().any():
    raise RuntimeError(
        "Submission has NaNs after reindexing; predictions are not aligned."
    )

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
