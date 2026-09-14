# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8980419219614995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the internet dependency that’s causing the SSL/certificate crash by preventing any pretrained-weight downloads and instead requiring a local checkpoint; if no checkpoint is found, the script still run and emit a valid submission (with a clear warning), rather than erroring out before writing the CSV. I also fix the downstream `NameError` by guaranteeing `final_predictions` is always defined (even in fallback). Finally, I make the checkpoint path discovery robust by searching common Kaggle input locations for the referenced `.pth` files, without changing the model architecture or inference logic. This is primarily a correctness/stability patch to produce a submission end-to-end; score depend on whether the checkpoints are available locally.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the warning path: the model is running with random weights because no checkpoints are being found/loaded, so predictions are effectively noise. To move the score toward the 0.898 target with minimal changes, I (1) make checkpoint discovery actually find common Kaggle dataset layouts by searching for `.pth` files by basename and also by broad glob patterns, and (2) add the intended second model (DenseNet121) as an ensemble only when its checkpoint is found—this preserves your inference approach (TTA flip + average) and post-processing while materially improving predictions if weights exist. If neither checkpoint exists, it still produce a valid `submission.csv` exactly as before (so no regression in “runs end-to-end”). I not change your model architectures, transforms, test-time augmentation logic, or threshold mapping semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly indicates you’re still falling back to random weights (no checkpoints found/loaded), so the smallest improvement toward the 0.898 target is to make checkpoint discovery actually find the `.pth` files that exist under typical Kaggle dataset layouts. I keep your model/inference/post-processing identical, but (1) broaden `_resolve_checkpoint_path` to also glob for likely basenames (including `model30*.pth`, `*512*.pth`, etc.) and prefer matches that contain “densenet/seresnet” in the path, and (2) make checkpoint loading tolerant to common wrapper keys (`model`, `net`, `encoder`, `state_dict`) while still using `strict=True` once the inner dict is extracted. This should load the intended weights when present (moving score up toward target) while still producing a valid `submission.csv` if nothing is found. No changes are made to thresholds, TTA, architectures, or the core prediction loop.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates you’re still almost certainly running with random weights (no real checkpoints loaded), so the smallest change that can move you toward the 0.898 target is to make checkpoint discovery and loading actually succeed on common Kaggle `.pth` formats/paths. I keep your exact models (SE-ResNet50+GeM and optional DenseNet121+GeM), TTA flip averaging, and threshold mapping unchanged, but broaden checkpoint resolution to also search `/kaggle/input/**` for likely basenames and prefer matches by architecture tokens. I also make checkpoint loading robust to common key prefixes (e.g., `model.`, `net.`, `module.`) and “wrapped” dicts while still loading strictly once the correct sub-dict is extracted. If no checkpoints exist, it still produce a valid `submission.csv` exactly as before (but the score remain poor).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the script is still running with random weights (no checkpoints found/loaded), so the minimal way to move toward the 0.898 target is to (1) make checkpoint discovery search for real `.pth` files under `/kaggle/input` robustly (including common archive-extracted locations) and (2) make checkpoint loading handle common wrapper formats and key mismatches (especially `last_linear.*` vs `fc.*`, `classifier.*`, and `avg_pool.*`). These changes keep your exact model architectures, inference loop (TTA flip + average), and thresholding identical, but greatly increase the chance the intended weights actually load. If no checkpoints exist, it still produce a valid `submission.csv` (score remain poor, but it run end-to-end).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running inference from randomly initialized weights because no real checkpoints are actually being found/loaded, so the most direct way to move toward the 0.898 target is to (1) make checkpoint discovery more reliable in Kaggle’s filesystem (including searching for common `.pth` basenames and also looking inside common dataset folders), and (2) make checkpoint loading robust to more “wrapped” formats and key name variants while still preserving the exact same model definitions and inference logic. I not change your architectures (SE-ResNet50+GeM and optional DenseNet121+GeM), your TTA (horizontal flip average), or your threshold mapping; the only functional changes are around reliably locating/loading weights so the same inference pipeline can actually use the intended trained models. If no checkpoints exist, the code still run end-to-end and emit a valid `submission.csv` (as now), but the goal is to greatly reduce the chance you silently fall back to random weights.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because you are still running with randomly initialized weights (no real checkpoints are being loaded), so the smallest change to move toward the 0.898 target is to (1) ensure the code never attempts any internet downloads for “pretrained” URLs, and (2) make local checkpoint resolution/loading actually succeed for common Kaggle `.pth` formats (nested keys, `state_dict` wrappers, and `module.` prefixes). I keep your exact model definitions (SE-ResNet50+GeM and optional DenseNet121+GeM), your inference loop (single-image + horizontal-flip TTA average), and your threshold mapping unchanged. The only functional changes are around reliably finding `.pth` files under `/kaggle/input/**` and robustly extracting the correct tensor dict to load strictly. This should materially increase the chance you use the intended trained weights (raising QWK toward target) while still always producing a valid `submission.csv`.'

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
import os

_DISABLE_NET_PRETRAIN = True
if _DISABLE_NET_PRETRAIN:

    def _no_inet_load_url(*args, **kwargs):
        raise RuntimeError(
            "Internet pretrained weight download disabled. Provide local checkpoint instead."
        )

    model_zoo.load_url = _no_inet_load_url  # type: ignore[attr-defined]

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

if _DISABLE_NET_PRETRAIN:

    def _no_inet_load_url2(*args, **kwargs):
        raise RuntimeError(
            "Internet pretrained weight download disabled. Provide local checkpoint instead."
        )

    model_zoo.load_url = _no_inet_load_url2  # type: ignore[attr-defined]

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
        model = se_resnet50(num_classes=1000, pretrained=None)
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained=None)
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 4
import os
from glob import glob

import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "test.csv")):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection data root in expected paths."
    )

TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

test_df = pd.read_csv(TEST_CSV_PATH)
test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in test_df["id_code"].tolist()
]

missing = [p for p in test_images if not os.path.exists(p)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (showing up to 3): {missing[:3]}"
    )

print("Device:", device)
print("Num test images:", len(test_images))




## === cell 5
def make_predictions(
    model, test_images, transforms, size=256, device=torch.device("cpu")
):
    predictions = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            image = Image.open(im_path).convert("RGB")
            image = image.resize((size, size), resample=Image.BILINEAR)

            image_t = transforms(image).to(device)
            out = model(image_t.unsqueeze(0))
            out_flip = model(torch.flip(image_t.unsqueeze(0), dims=(3,)))

            final_prediction = (out.item() + out_flip.item()) / 2.0
            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions


def _extract_state_dict(state):
    if not isinstance(state, dict):
        return state
    for k in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "encoder",
        "weights",
        "params",
        "ema",
        "student",
        "teacher",
    ]:
        if k in state and isinstance(state[k], dict):
            return state[k]
    return state


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = (
        "module.",
        "model.",
        "net.",
        "network.",
        "encoder.",
        "backbone.",
    )
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_state[nk] = v
    return new_state


def _remap_common_head_names(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("fc."):
            nk = "last_linear." + nk[len("fc.") :]
        elif nk.startswith("classifier."):
            nk = "last_linear." + nk[len("classifier.") :]
        remapped[nk] = v
    return remapped


def _extract_largest_tensor_state_dict(state):
    if not isinstance(state, dict):
        return state
    best = None
    best_n = -1
    for _, v in state.items():
        if isinstance(v, dict) and len(v) > 0:
            n = 0
            for vv in v.values():
                if torch.is_tensor(vv):
                    n += 1
            if n > best_n:
                best_n = n
                best = v
    return best if best is not None and best_n > 0 else state


def _maybe_remove_prefix_from_keys(state_dict, prefix: str):
    if not isinstance(state_dict, dict):
        return state_dict
    if not prefix:
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _load_checkpoint_if_exists(model, model_path, device):
    if model_path is None:
        return False
    if not os.path.exists(model_path):
        return False

    state = torch.load(model_path, map_location=device)

    state = _extract_state_dict(state)
    state = _extract_largest_tensor_state_dict(state)

    state = _strip_known_prefixes(state)
    state = _remap_common_head_names(state)

    state = _maybe_remove_prefix_from_keys(state, "module.")
    state = _maybe_remove_prefix_from_keys(state, "model.")
    state = _maybe_remove_prefix_from_keys(state, "net.")

    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if (len(missing) > 50) or (len(unexpected) > 2000):
            return False
        return True


def _resolve_checkpoint_path(path_hint: str):
    if path_hint is None:
        return None
    if os.path.exists(path_hint) and os.path.isfile(path_hint):
        return path_hint

    fname = os.path.basename(path_hint)
    stem, _ = os.path.splitext(fname)
    hint_l = path_hint.lower()

    roots = ["/kaggle/input", "/kaggle/working", "/kaggle/data"]
    candidates = []

    for base in roots:
        if os.path.exists(base):
            candidates.extend(glob(os.path.join(base, "**", fname), recursive=True))

    parent_token = None
    try:
        parent_token = os.path.basename(os.path.dirname(path_hint)).lower()
        if parent_token in ("input", "kaggle", ""):
            parent_token = None
    except Exception:
        parent_token = None

    if len(candidates) == 0:
        patterns = []
        if "densenet" in hint_l or "dense" in hint_l:
            patterns = [
                "**/*densenet*121*.pth",
                "**/*dense*121*.pth",
                "**/*densenet*.pth",
                "**/*dense*.pth",
                "**/model*dense*.pth",
                "**/model*densenet*.pth",
                "**/*bs64*30*.pth",
                "**/*densenet121*.pth",
            ]
        elif "seresnet" in hint_l or "se_resnet" in hint_l or "senet" in hint_l:
            patterns = [
                "**/*seresnet*50*.pth",
                "**/*se_resnet*50*.pth",
                "**/*senet*.pth",
                "**/*resnet*50*.pth",
                "**/*fine_tune*model*.pth",
                "**/fine_tune_*model*.pth",
                "**/model30*.pth",
                "**/*512*.pth",
                "**/*256*.pth",
                "**/*se*resnet*50*.pth",
            ]
        else:
            patterns = ["**/*.pth"]

        for base in roots:
            if os.path.exists(base):
                for pat in patterns:
                    candidates.extend(glob(os.path.join(base, pat), recursive=True))

    candidates = [p for p in set(candidates) if os.path.isfile(p)]
    if len(candidates) == 0:
        return None

    def score_path(p):
        pl = p.lower()
        s = 0

        if pl.startswith("/kaggle/input/"):
            s += 5

        if parent_token is not None and parent_token in pl:
            s += 12

        if "densenet" in hint_l or "dense" in hint_l:
            s += 25 if ("densenet" in pl or "dense" in pl) else 0
            s += 10 if "121" in pl else 0
            s += 5 if "bs64" in pl else 0
            s += 3 if "30" in pl else 0

        if ("seresnet" in hint_l) or ("se_resnet" in hint_l) or ("senet" in hint_l):
            s += 25 if ("seresnet" in pl or "se_resnet" in pl or "senet" in pl) else 0
            s += 10 if "50" in pl else 0
            s += 8 if "fine_tune" in pl else 0
            s += 6 if "512" in pl else 0
            s += 4 if "256" in pl else 0
            s += 3 if "model30" in pl else 0

        if stem and stem.lower() in pl:
            s += 20

        return (-s, len(p), p)

    candidates.sort(key=score_path)
    return candidates[0]


def _pred_list_to_df(pred_list, colname):
    return pd.DataFrame(pred_list, columns=["id_code", colname])


def _ensemble_avg(pred_a, pred_b, w_a=0.5):
    df_a = _pred_list_to_df(pred_a, "a")
    df_b = _pred_list_to_df(pred_b, "b")
    m = df_a.merge(df_b, on="id_code", how="inner")
    if len(m) != len(df_a) or len(m) != len(df_b):
        raise RuntimeError("Ensemble merge mismatch; check id_code alignment.")
    m["diagnosis_raw"] = w_a * m["a"].astype(float) + (1.0 - w_a) * m["b"].astype(float)
    return list(zip(m["id_code"].tolist(), m["diagnosis_raw"].tolist()))




## === cell 6
MODEL_PATH_DENSENET = "../input/densenet121/model_densenet121_bs64_30.pth"
MODEL_PATH_SERESNET_256 = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
MODEL_PATH_SERESNET_512 = "../input/seresnet50-512/model30_512.pth"

MODEL_PATH_DENSENET = _resolve_checkpoint_path(MODEL_PATH_DENSENET)
MODEL_PATH_SERESNET_256 = _resolve_checkpoint_path(MODEL_PATH_SERESNET_256)
MODEL_PATH_SERESNET_512 = _resolve_checkpoint_path(MODEL_PATH_SERESNET_512)

print("Resolved checkpoints:")
print("  Densenet121:", MODEL_PATH_DENSENET)
print("  SE-ResNet50 256:", MODEL_PATH_SERESNET_256)
print("  SE-ResNet50 512:", MODEL_PATH_SERESNET_512)

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model_seresnet = get_se_resnet50_gem(pretrain=False)
loaded_seresnet = False
use_size_seresnet = 256

if MODEL_PATH_SERESNET_512 is not None:
    loaded_seresnet = _load_checkpoint_if_exists(
        model_seresnet, MODEL_PATH_SERESNET_512, device=device
    )
    if loaded_seresnet:
        use_size_seresnet = 512

if (not loaded_seresnet) and (MODEL_PATH_SERESNET_256 is not None):
    loaded_seresnet = _load_checkpoint_if_exists(
        model_seresnet, MODEL_PATH_SERESNET_256, device=device
    )
    if loaded_seresnet:
        use_size_seresnet = 256

model_seresnet.to(device)
model_seresnet.eval()

predictions_seresnet = None
if loaded_seresnet:
    predictions_seresnet = make_predictions(
        model_seresnet, test_images, norm, size=use_size_seresnet, device=device
    )
else:
    print("WARNING: No SE-ResNet checkpoint found/loaded locally.")

predictions_densenet = None
loaded_densenet = False
if MODEL_PATH_DENSENET is not None:
    model_densenet = get_densenet121_gem(pretrain=False)
    loaded_densenet = _load_checkpoint_if_exists(
        model_densenet, MODEL_PATH_DENSENET, device=device
    )
    if loaded_densenet:
        model_densenet.to(device)
        model_densenet.eval()
        predictions_densenet = make_predictions(
            model_densenet, test_images, norm, size=256, device=device
        )
    else:
        print("WARNING: DenseNet checkpoint path resolved but could not be loaded.")

if (not loaded_seresnet) and (not loaded_densenet):
    raise RuntimeError(
        "No model checkpoints were loaded. Refusing to generate random-weight submission (would score ~0.0). "
        "Please attach the dataset containing the .pth weights or update MODEL_PATH_* hints."
    )

if loaded_seresnet and loaded_densenet and (predictions_densenet is not None):
    final_predictions = _ensemble_avg(
        predictions_seresnet, predictions_densenet, w_a=0.5
    )
    print("Using ensemble: SE-ResNet + DenseNet (avg).")
elif loaded_seresnet:
    final_predictions = predictions_seresnet
    print("Using single model: SE-ResNet only.")
else:
    final_predictions = predictions_densenet
    print("Using single model: DenseNet only.")

print("Predictions generated:", len(final_predictions))
print("Example:", final_predictions[0])



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_54/437746013.py in <cell line: 0>()
     67 # This does not change model/inference semantics; it just ensures we only submit when real weights are loaded.
     68 if (not loaded_seresnet) and (not loaded_densenet):
---> 69     raise RuntimeError(
     70         "No model checkpoints were loaded. Refusing to generate random-weight submission (would score ~0.0). "
     71         "Please attach the dataset containing the .pth weights or update MODEL_PATH_* hints."

RuntimeError: No model checkpoints were loaded. Refusing to generate random-weight submission (would score ~0.0). Please attach the dataset containing the .pth weights or update MODEL_PATH_* hints.

## === cell 7
pred_df = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis_raw"])
submission = test_df.merge(pred_df, on="id_code", how="left")

if submission["diagnosis_raw"].isna().any():
    raise RuntimeError(
        "Some predictions are missing after merge; check id_code alignment."
    )

submission["diagnosis"] = submission["diagnosis_raw"].astype(float)

submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
submission.loc[
    (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
] = 3
submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4

submission["diagnosis"] = submission["diagnosis"].astype(int)
submission = submission[["id_code", "diagnosis"]]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3786272433.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis_raw"])
      2 submission = test_df.merge(pred_df, on="id_code", how="left")
      3 
      4 if submission["diagnosis_raw"].isna().any():
      5     raise RuntimeError(

NameError: name 'final_predictions' is not defined
