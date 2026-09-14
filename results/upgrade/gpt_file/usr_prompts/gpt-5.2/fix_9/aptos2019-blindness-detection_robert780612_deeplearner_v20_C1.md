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

0.917679475610932

# 6. Current score

0.02926

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the execution blockers that prevent any submission from being generated: (1) remove the invalid `from __future__` imports placed mid-file, and (2) make the device selection CPU-safe because this Kaggle runtime has no NVIDIA driver. I also make model weight loading robust to CPU/GPU checkpoints, add `torch.no_grad()` during inference to reduce memory/overhead, and ensure test image ordering is deterministic so merges align. Finally, I add a safe fallback so if a provided model path isn’t present, the code still produces a valid `submission.csv` in the required format (score may be low in that fallback case, but it run end-to-end).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a “valid file but essentially random/constant predictions” outcome caused by not loading the intended checkpoints (both model paths are likely missing), plus a small preprocessing mismatch for the DenseNet branch (no ImageNet normalization). I keep the same models, ensembling, and regression-to-ordinal thresholds, but (1) make checkpoint discovery robust by searching common `/kaggle/input/**` locations if the hardcoded paths don’t exist, and (2) apply the correct normalization for DenseNet at inference to match typical training. I also ensure we always write a submission aligned exactly to `sample_submission.csv` ordering (already mostly done) and keep everything CPU-safe and deterministic. These are minimal changes that should move the score up from 0.0 toward your target if the checkpoints are present anywhere in the dataset inputs.'
- What this solution (achieved 0.01899) has done: 'I fix the execution blocker by removing the hard failure when checkpoints aren’t found and instead produce a valid submission using a deterministic, non-constant heuristic based on image brightness/contrast (so it’s more likely to score above 0.0 while staying within the same “regression → thresholds” evaluation semantics). I also make checkpoint discovery broader (searching `/kaggle/input`, `/kaggle/data`, and `/kaggle/working`) and allow slightly more flexible state_dict loading (handling common key-prefix mismatches) so that if the intended weights exist anywhere, the original model ensemble path run unchanged. Finally, I keep ordering aligned to `sample_submission.csv` and always write `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.01899) has done: 'Your current score is far below the target, so the fastest way to move toward 0.917 is to ensure you are actually using the intended trained checkpoints rather than falling back to the brightness heuristic. I keep the same two-model ensemble and the same regression→fixed-threshold binning, but make checkpoint discovery much more reliable by (1) scanning for any `.pth/.pt` under the dataset roots and selecting the best match by filename tokens, and (2) trying multiple candidate checkpoints until one loads successfully. Finally, I add a small sanity-print of the predicted continuous distribution so it’s obvious if the pipeline still fell back or outputs near-constant values (which correlates with very low kappa).'
- What this solution (achieved 0.02926) has done: 'Your score gap is large and consistent with the script still falling back to the brightness heuristic (or producing poorly calibrated ordinal bins), so the most direct way to move toward the target is to (1) make checkpoint discovery/loading more reliable and observable, and (2) keep your exact model+ensemble logic but tune only the *post-processing thresholds* using a small validation split with quadratic weighted kappa. This keeps the architecture, inference procedure (TTA flip + averaging), and regression→ordinal semantics identical, but replaces fixed cutpoints with data-driven cutpoints that usually produce a big kappa jump. I also ensure test ordering is strictly aligned to `sample_submission.csv` from the start (so merges can’t silently misalign), and I print whether we actually loaded weights or not.'
- What this solution (achieved 0.02926) has done: 'Your current score (0.02926) is far below the target (0.91768), so we should increase performance; the most likely culprit is that checkpoint discovery is still not finding/using the intended trained weights, leaving you with weak predictions (often the heuristic fallback). I make checkpoint selection more reliable by preferring candidates whose filenames contain model-specific tokens (e.g., “densenet”, “seresnet”, “se_resnet”, “fine_tune”, “model30”) and by verifying load success on each candidate, while keeping the same models and inference logic. I also fix a subtle but important validation bug: when `used_mode` is `densenet_only` or `seresnet_only`, the code currently tries to load the *other* missing checkpoint during threshold tuning, which can silently crash or degrade the pipeline; we only tune thresholds using actually-available models. Finally, I keep submission alignment strictly to `sample_submission.csv` ordering and keep the same regression→thresholding semantics, just with safer threshold tuning.'
- What this solution (achieved 0.02926) has done: 'Your current gap to the target is very large, so we should increase performance; the most likely reason for ~0.03 QWK is still that (a) no real checkpoints are being loaded (falling back to the brightness heuristic), and/or (b) the val-threshold tuning is being misled by using the wrong image preprocessing for the training images (APTOS images often have black borders and need the same center-crop/scale behavior as in training). I keep your exact two-model inference/ensemble and the same “regression → thresholds → 0-4 labels” semantics, but make checkpoint discovery more aggressive/specific and add a deterministic “best candidate selection” that validates state_dict keys before using a checkpoint. I also minimally improve the image preprocessing consistency by adding a simple center-crop after resizing (still 224/256 as you already do) for both test and validation predictions, which often stabilizes QWK a lot on this competition without changing the model. Finally, I ensure threshold optimization always runs on predictions generated with the *same* preprocessing as test (per model) and that ordering is strictly aligned everywhere.'

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
    model = models.alexnet(pretrained=False)
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
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained="imagenet"):
    model = models.inception_v3(pretrained=False)
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
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    model = models.resnet152(pretrained=False)
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
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_1(pretrained=False)
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
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19_bn(pretrained=False)
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

        layers = []
        layers.append(
            block(self.inplanes, planes, groups, reduction, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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




## === cell 4
import os
from glob import glob

import torch
import pandas as pd
from PIL import Image, ImageStat
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"

test_csv_path = "/kaggle/input/aptos2019-blindness-detection/test.csv"
train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
sample_path = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"

if os.path.exists(sample_path):
    _sample_df = pd.read_csv(sample_path)
    test_images = [
        os.path.join(TEST_IMAGE_PATH, f"{i}.png")
        for i in _sample_df["id_code"].tolist()
    ]
elif os.path.exists(test_csv_path):
    _test_df = pd.read_csv(test_csv_path)
    test_images = [
        os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in _test_df["id_code"].tolist()
    ]
else:
    test_images = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*.png")))


def find_checkpoint(preferred_path, filename_hint=None, search_roots=None):
    if search_roots is None:
        search_roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]

    exts = (".pth", ".pt", ".bin")

    candidates = []
    if preferred_path and os.path.exists(preferred_path):
        candidates.append(preferred_path)

    if filename_hint is None and preferred_path:
        filename_hint = os.path.basename(preferred_path)

    all_ckpts = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for ext in exts:
            all_ckpts.extend(glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    all_ckpts = sorted(set(all_ckpts))

    if filename_hint:
        hint_base = os.path.basename(filename_hint).lower()
        hint_stem = os.path.splitext(hint_base)[0]

        exact = [p for p in all_ckpts if os.path.basename(p).lower() == hint_base]
        stem_match = [p for p in all_ckpts if hint_stem in os.path.basename(p).lower()]
        tokens = [t for t in re.split(r"[^a-z0-9]+", hint_stem) if t]
        token_match = []
        for p in all_ckpts:
            bn = os.path.basename(p).lower()
            if tokens and all(t in bn for t in tokens[: max(1, min(4, len(tokens)))]):
                token_match.append(p)

        dir_match = []
        for p in all_ckpts:
            pl = p.lower()
            if any(
                s in pl
                for s in (
                    "/weights",
                    "/weight",
                    "/checkpoint",
                    "/checkpoints",
                    "/model",
                    "/models",
                )
            ):
                dir_match.append(p)

        for group in (exact, stem_match, token_match, dir_match, all_ckpts):
            for p in group:
                if p not in candidates:
                    candidates.append(p)
    else:
        for p in all_ckpts:
            if p not in candidates:
                candidates.append(p)

    return candidates


def rank_checkpoint_candidates(paths, prefer_tokens):
    prefer_tokens = [t.lower() for t in (prefer_tokens or [])]
    scored = []
    for p in paths:
        bn = os.path.basename(p).lower()
        score = 0
        for t in prefer_tokens:
            if t and (t in bn):
                score += 10
        score -= min(len(p), 400) / 400.0
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def safe_load_state_dict(model, ckpt_path, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)} from {ckpt_path}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    return model


def _center_crop_pil(img, out_size):
    w, h = img.size
    if w == out_size and h == out_size:
        return img
    left = max(0, (w - out_size) // 2)
    top = max(0, (h - out_size) // 2)
    right = left + out_size
    bottom = top + out_size
    return img.crop((left, top, right, bottom))


def make_predictions(model, test_images, transforms_fn, size=256, device=device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            image = Image.open(im_path).convert("RGB")
            image = image.resize((size, size), resample=Image.BILINEAR)
            image = _center_crop_pil(image, size)

            image = transforms_fn(image).to(device)
            output = model(image.unsqueeze(0))
            output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))

            final_prediction = (output.item() + output_flip.item()) / 2.0
            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions


def heuristic_predictions(test_images):
    preds = []
    for im_path in test_images:
        img = Image.open(im_path).convert("RGB")
        img_small = img.resize((256, 256), resample=Image.BILINEAR).convert("L")
        stat = ImageStat.Stat(img_small)
        mean = float(stat.mean[0]) / 255.0
        std = float(stat.stddev[0]) / 255.0

        score = 4.0 * np.clip(
            0.85 * (1.0 - mean) + 0.15 * np.clip(0.25 - std, 0, 0.25) / 0.25, 0.0, 1.0
        )

        preds.append((os.path.splitext(os.path.basename(im_path))[0], float(score)))
    return preds


def try_load_any_checkpoint(model, candidate_paths, map_location):
    last_err = None
    model_keys = set(model.state_dict().keys())

    def _looks_compatible(path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            if (
                isinstance(ckpt, dict)
                and "state_dict" in ckpt
                and isinstance(ckpt["state_dict"], dict)
            ):
                state = ckpt["state_dict"]
            elif (
                isinstance(ckpt, dict)
                and "model_state_dict" in ckpt
                and isinstance(ckpt["model_state_dict"], dict)
            ):
                state = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict):
                state = ckpt
            else:
                return False
            if any(k.startswith("module.") for k in state.keys()):
                keys = {k.replace("module.", "", 1) for k in state.keys()}
            elif any(k.startswith("model.") for k in state.keys()):
                keys = {k.replace("model.", "", 1) for k in state.keys()}
            else:
                keys = set(state.keys())
            overlap = len(model_keys.intersection(keys))
            return overlap >= max(50, int(0.25 * len(model_keys)))
        except Exception:
            return False

    filtered = [p for p in candidate_paths if p and os.path.exists(p)]
    compat = [p for p in filtered if _looks_compatible(p)]
    ordered = compat + [p for p in filtered if p not in compat]

    for p in ordered:
        try:
            safe_load_state_dict(model, p, map_location=map_location)
            return p
        except Exception as e:
            last_err = e
            continue
    return None


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / (E.sum() + 1e-12) * (O.sum() + 1e-12)

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum() + 1e-12
    return 1.0 - num / den


def apply_thresholds(cont_preds, thr):
    thr = list(thr)
    x = np.asarray(cont_preds, dtype=np.float64)
    out = np.zeros_like(x, dtype=int)
    out[x >= thr[0]] = 1
    out[x >= thr[1]] = 2
    out[x >= thr[2]] = 3
    out[x >= thr[3]] = 4
    return out


def optimize_thresholds_for_qwk(
    y_true, cont_preds, init_thr=(0.7, 1.5, 2.5, 3.5), n_steps=18, step0=0.25
):
    y_true = np.asarray(y_true, dtype=int)
    cont_preds = np.asarray(cont_preds, dtype=np.float64)
    thr = np.array(init_thr, dtype=np.float64)

    best_thr = thr.copy()
    best_score = quadratic_weighted_kappa(
        y_true, apply_thresholds(cont_preds, best_thr)
    )

    step = float(step0)
    for _ in range(n_steps):
        improved = False
        for i in range(4):
            for direction in (-1.0, 1.0):
                cand = best_thr.copy()
                cand[i] += direction * step

                cand = np.clip(cand, -1.0, 5.0)
                cand = np.sort(cand)

                score = quadratic_weighted_kappa(
                    y_true, apply_thresholds(cont_preds, cand)
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best_thr = cand
                    improved = True
        if not improved:
            step *= 0.6
    return tuple(float(x) for x in best_thr), float(best_score)




## === cell 5
MODEL_PATH = "../input/densenet121/model_densenet121_bs64_30.pth"
densenet_candidates = find_checkpoint(
    MODEL_PATH, filename_hint="model_densenet121_bs64_30.pth"
)

densenet_candidates = rank_checkpoint_candidates(
    densenet_candidates,
    prefer_tokens=["densenet", "dense", "121", "gem", "bs64", "model", "30"],
)

model = get_densenet121_gem(pretrain=False).to(device)

predictions_densenet = None
resolved_densenet = try_load_any_checkpoint(
    model, densenet_candidates, map_location=device
)

if resolved_densenet is not None and os.path.exists(resolved_densenet):
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_densenet = make_predictions(
        model, test_images, norm, size=224, device=device
    )



## === cell 6
MODEL_PATH = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
seresnet_candidates = find_checkpoint(
    MODEL_PATH, filename_hint="fine_tune_256_model30.pth"
)

seresnet_candidates = rank_checkpoint_candidates(
    seresnet_candidates,
    prefer_tokens=[
        "seresnet",
        "se_resnet",
        "se-resnet",
        "resnet",
        "50",
        "gem",
        "fine",
        "tune",
        "256",
        "model",
        "30",
    ],
)

model = get_se_resnet50_gem(pretrain=False).to(device)

predictions_seresnet = None
resolved_seresnet = try_load_any_checkpoint(
    model, seresnet_candidates, map_location=device
)

if resolved_seresnet is not None and os.path.exists(resolved_seresnet):
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet = make_predictions(
        model, test_images, norm, size=256, device=device
    )



## === cell 7
if predictions_densenet is not None and predictions_seresnet is not None:
    final_predictions = []
    for x, y in zip(predictions_densenet, predictions_seresnet):
        assert x[0] == y[0]
        final_predictions.append([x[0], (x[1] + y[1]) / 2.0])
    used_mode = "ensemble"
elif predictions_densenet is not None:
    final_predictions = [[k, v] for (k, v) in predictions_densenet]
    used_mode = "densenet_only"
elif predictions_seresnet is not None:
    final_predictions = [[k, v] for (k, v) in predictions_seresnet]
    used_mode = "seresnet_only"
else:
    final_predictions = [[k, v] for (k, v) in heuristic_predictions(test_images)]
    used_mode = "heuristic_fallback"

submission_cont = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
_cont = submission_cont["diagnosis"].astype(float).values
print("Mode:", used_mode)
print("DenseNet ckpt:", resolved_densenet)
print("SEResNet ckpt:", resolved_seresnet)
print(
    "Continuous preds summary:",
    {
        "min": float(np.min(_cont)),
        "p10": float(np.quantile(_cont, 0.1)),
        "median": float(np.median(_cont)),
        "p90": float(np.quantile(_cont, 0.9)),
        "max": float(np.max(_cont)),
        "std": float(np.std(_cont)),
    },
)

thr = (0.7, 1.5, 2.5, 3.5)
thr_source = "fixed_default"

if os.path.exists(train_csv_path) and os.path.exists(TRAIN_IMAGE_PATH):
    train_df = pd.read_csv(train_csv_path)
    train_paths = [
        os.path.join(TRAIN_IMAGE_PATH, f"{i}.png") for i in train_df["id_code"].tolist()
    ]
    y = train_df["diagnosis"].astype(int).values

    rng = np.random.RandomState(123)
    idx = np.arange(len(train_paths))
    rng.shuffle(idx)
    val_n = min(600, len(idx))
    val_idx = idx[:val_n]

    val_paths = [train_paths[i] for i in val_idx]
    y_val = y[val_idx]

    if used_mode != "heuristic_fallback":
        val_preds_list = []

        if (
            used_mode in ("ensemble", "densenet_only")
            and resolved_densenet is not None
            and os.path.exists(resolved_densenet)
        ):
            m = get_densenet121_gem(pretrain=False).to(device)
            _ = safe_load_state_dict(m, resolved_densenet, map_location=device)
            m.eval()
            norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            vp = make_predictions(m, val_paths, norm, size=224, device=device)
            val_preds_list.append(np.array([p[1] for p in vp], dtype=np.float64))

        if (
            used_mode in ("ensemble", "seresnet_only")
            and resolved_seresnet is not None
            and os.path.exists(resolved_seresnet)
        ):
            m = get_se_resnet50_gem(pretrain=False).to(device)
            _ = safe_load_state_dict(m, resolved_seresnet, map_location=device)
            m.eval()
            norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            vp = make_predictions(m, val_paths, norm, size=256, device=device)
            val_preds_list.append(np.array([p[1] for p in vp], dtype=np.float64))

        if len(val_preds_list) > 0:
            val_cont = np.mean(np.stack(val_preds_list, axis=0), axis=0)
        else:
            vp = heuristic_predictions(val_paths)
            val_cont = np.array([p[1] for p in vp], dtype=np.float64)
    else:
        vp = heuristic_predictions(val_paths)
        val_cont = np.array([p[1] for p in vp], dtype=np.float64)

    thr, best_qwk = optimize_thresholds_for_qwk(y_val, val_cont, init_thr=thr)
    thr_source = f"optimized_on_val(n={len(y_val)})"
    print("Optimized thresholds:", thr, "val_qwk:", best_qwk)

pred_labels = apply_thresholds(submission_cont["diagnosis"].astype(float).values, thr)
submission = pd.DataFrame(
    {"id_code": submission_cont["id_code"].values, "diagnosis": pred_labels.astype(int)}
)

if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["id_code"]].merge(submission, on="id_code", how="left")
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

print("Threshold source:", thr_source)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
