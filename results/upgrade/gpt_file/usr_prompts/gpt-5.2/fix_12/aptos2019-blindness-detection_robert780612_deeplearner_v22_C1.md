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

0.9184521305234512

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime by making the device selection CPU-safe (use CUDA only if available) and wrapping inference in `torch.no_grad()` so it runs efficiently end-to-end in Kaggle without a GPU driver. I also make the model weight loading robust by checking alternative common `/kaggle/input/...` paths and using `map_location=device`, so missing/CPU environments don’t crash. Next, I ensure predictions are aligned and complete by sorting by `id_code` and merging on `id_code` instead of relying on `zip`, which can silently misalign. Finally, I always generate a valid `submission.csv` with the required columns, falling back to `sample_submission.csv` shape if any model file is unavailable so you still get a submit-ready CSV.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a submission/prediction alignment problem rather than model quality, because the script can silently drop rows by merging predictions with `how="inner"` (keeping only ids predicted by *all* available models) and then filling missing ids with 0, which heavily hurts QWK. I keep your exact model/inference logic, but change the ensemble merge to a `how="outer"` merge and then re-align to `test_df` so every test id gets a prediction even if one model is missing an image or fails on a subset. I also add a minimal, metric-relevant improvement: compute thresholds on the training labels’ empirical class distribution by mapping the regression output quantiles to classes (still ordinal discretization, no new training), which typically boosts QWK vs fixed cutoffs while keeping semantics the same. Finally, I ensure the submission is exactly 367 rows, in test order, with integer `diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing essentially constant/invalid predictions (often caused here by missing model weights, wrong DATA_ROOT, or silently failing loads), so the smallest score-moving change is to guarantee real predictions even when external weight files aren’t present. I keep your exact architectures and inference flow, but (1) make DATA_ROOT resolve to an actually-existing input folder, (2) optionally use torchvision ImageNet weights as a fallback when your custom `.pth` files aren’t available (so outputs vary per image instead of all-zeros), and (3) fix the DenseNet transform to include the same normalization as the other models (a metric-relevant calibration fix, not a logic rewrite). The submission is still aligned to `test.csv` order and always written as `submission.csv` with 367 rows and integer `diagnosis`.'
- What this solution (achieved -0.10804) has done: 'I fix the immediate runtime errors by removing the broken custom ImageNet-pretrain download paths (which cause `KeyError` and SSL download failures) and instead using torchvision’s built-in ImageNet weights only as a fallback when your custom `.pth` files aren’t found. I also correct the `pretrain` flag logic (it currently passes `False` instead of `None`/`"imagenet"`) so model construction is deterministic and doesn’t accidentally trigger unwanted pretrained-loading code. Finally, I keep your existing inference/ensemble/quantile-binning logic intact, but ensure every test id gets a prediction and a valid `submission.csv` is always written.'
- What this solution (achieved 0.23784) has done: 'Your current negative QWK is most consistent with predictions being effectively random with respect to the ordinal labels, which can happen here because you’re using ImageNet backbones with a randomly-initialized 1-unit regression head when the custom .pth weights are missing/not found. To move the score toward the target with minimal semantic change, I keep your exact models and inference flow, but make weight-path resolution search more robust in the actual `/kaggle/input` tree so the intended fine-tuned checkpoints are much more likely to be loaded. I also make state-dict loading tolerant to common key-prefix differences (`module.`, `model.`) and allow `strict=False` as a fallback only when shapes match, preventing silent all-random heads due to tiny naming mismatches. Finally, I keep your quantile-binning and submission alignment, but add a safe check to ensure we predict for every `id_code` in `test.csv` even if image listing misses any files.'
- What this solution (achieved -0.08661) has done: 'Your current score (0.23784) is far below the target (0.91845), so the most likely blocker is that the intended fine-tuned checkpoints are still not being loaded and you’re effectively predicting with ImageNet backbones + random 1-unit heads. I keep your exact model architectures and inference loop, but make checkpoint discovery more robust (search `/kaggle/input` for multiple common filename variants and also allow `.pt/.bin`), and make state-dict loading handle more real-world wrappers (nested `model_state_dict`, `ema_state_dict`, etc.) so the fine-tuned heads actually load. I also add a strict sanity check that warns if a head stays randomly initialized (by verifying the loaded keys include the final layer), because that directly correlates with low QWK. Finally, I preserve your quantile-binning and submission alignment, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so the smallest likely blocker is that you’re still effectively using ImageNet backbones with a randomly initialized 1-unit head (or partially loaded checkpoints), which makes ordinal predictions near-random and can yield negative QWK. I keep your exact architectures and inference loop, but make checkpoint loading stricter for the regression head: only accept a checkpoint as “loaded” if it actually contains (and loads) `last_linear.*`; otherwise we fall back to ImageNet init + a deterministic “no-skill” mapping rather than random-head noise. Then, instead of quantile-binning random regression outputs, I use a minimal metric-relevant fallback: if no fine-tuned head was loaded for any model, predict the training-set majority class for all test images (this typically beats random/negative QWK on QWK). If at least one fine-tuned head is loaded, your current mean-ensemble + quantile-binning flow remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from a submission validity/alignment issue (e.g., missing some test ids because `test_images` filters out non-existing paths), which can severely break QWK even if the model is reasonable. I make the smallest change that guarantees we predict for all 367 `id_code`s by not dropping missing image paths and instead emitting a safe fallback prediction for any unreadable/missing file, keeping your exact model/inference and ensemble logic intact. I also add a strict final check that the submission matches `test.csv` ids and order (and force it to), because even a small misalignment can collapse the score. These changes should move you upward toward the target without altering the core modeling approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “no real model signal” (e.g., all/fallback predictions) caused by not finding or not correctly loading the fine-tuned checkpoints, so the smallest score-moving change is to (1) locate the intended `.pth` files robustly under both `/kaggle/input` and `/kaggle/data` and (2) correctly adapt common checkpoint formats where the head is stored under `last_linear.*` but your instantiated model uses a different attribute name (e.g., `classifier.*` / `fc.*`). I keep your exact architectures and inference flow, but add a minimal “head remap” during loading so DenseNet and SE-ResNet checkpoints from typical training code actually populate `model.last_linear.*` and are accepted as “head loaded”. With that, your ensemble + quantile-binning stays unchanged, but should stop collapsing to majority-class/random outputs and move the score upward toward the target. The script still always writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is far below the target, so the smallest likely score-moving fix is to ensure you’re actually using fine-tuned checkpoints (and not accidentally falling back to ImageNet + random 1-unit heads), because that can collapse QWK toward ~0. I keep your exact models and inference flow, but make checkpoint discovery/load more robust by (1) finding any `.pth/.pt/.bin` under `/kaggle/input` and `/kaggle/data` via filename *patterns* (not exact names only), and (2) accepting common checkpoint containers while still requiring the regression head to be present. Finally, I make prediction generation deterministic and submission-aligned (still in `test.csv` order) to avoid accidental id mismatches that can also yield near-zero QWK.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is far below the target, so the smallest likely score-moving fix is to stop any hidden misalignment/NaN behavior in the final discretization and to ensure all three models always contribute valid numeric predictions per `id_code`. I keep your exact models and inference loop, but make prediction generation deterministic and complete by (1) forcing float32 outputs, (2) filling per-model missing predictions before averaging, and (3) applying quantile binning only on non-NaN predictions with a safe fallback to majority class per-row if needed. I also add a minimal, metric-relevant improvement that preserves semantics: fit optimal 4 thresholds on the *training predictions* (same model outputs) via a small coordinate-descent to directly maximize QWK, then apply those thresholds to test predictions (no extra training, same outputs). The script still always write a valid `submission.csv` aligned to `test.csv` order with 367 rows.'

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

import os
from glob import glob

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


from torchvision.models import DenseNet121_Weights, ResNet50_Weights


def get_se_resnet50_gem(pretrain):
    model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    if pretrain == "imagenet":
        tv = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        state = tv.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape
        }
        model.load_state_dict(compatible, strict=False)
    return model


def get_densenet121_gem(pretrain):
    model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    if pretrain == "imagenet":
        tv = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        state = tv.state_dict()
        compatible = {
            k: v
            for k, v in state.items()
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape
        }
        model.load_state_dict(compatible, strict=False)
    return model




## === cell 4
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for _r in _CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(_r, "train.csv")) and os.path.exists(
        os.path.join(_r, "test.csv")
    ):
        DATA_ROOT = _r
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"

TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, "train_images")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(TEST_CSV_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in test_df["id_code"].tolist()
]



## === cell 5
from sklearn.metrics import cohen_kappa_score


def make_predictions(
    model, test_images, transforms_fn, size=256, device=torch.device("cpu")
):
    rows = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            img_id = os.path.splitext(os.path.basename(im_path))[0]

            try:
                if not os.path.exists(im_path):
                    raise FileNotFoundError(im_path)

                image = Image.open(im_path).convert("RGB")
                image = image.resize((size, size), resample=Image.BILINEAR)

                image_t = transforms_fn(image).to(device)
                out = model(image_t.unsqueeze(0))
                out_flip = model(torch.flip(image_t.unsqueeze(0), dims=(3,)))

                final_prediction = (out.float().item() + out_flip.float().item()) / 2.0
                if not np.isfinite(final_prediction):
                    final_prediction = 0.0
            except Exception:
                final_prediction = 0.0

            rows.append((img_id, np.float32(final_prediction)))

    return pd.DataFrame(rows, columns=["id_code", "pred"])


def _resolve_model_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_in_kaggle_trees(filename: str):
    for root in ("/kaggle/input", "/kaggle/data"):
        hits = glob(os.path.join(root, "**", filename), recursive=True)
        if len(hits) > 0:
            return hits[0]
    return None


def _find_by_patterns_in_kaggle_trees(patterns):
    hits_all = []
    for root in ("/kaggle/input", "/kaggle/data"):
        for pat in patterns:
            hits = glob(os.path.join(root, "**", pat), recursive=True)
            hits_all.extend(hits)
    hits_all = [h for h in hits_all if os.path.isfile(h)]
    hits_all = sorted(set(hits_all), key=lambda p: (len(p), p))
    return hits_all[0] if len(hits_all) else None


def _find_any_in_kaggle_trees(filenames):
    for fn in filenames:
        hit = _find_in_kaggle_trees(fn)
        if hit is not None:
            return hit
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "ema_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def _remap_head_keys_to_last_linear(cleaned: dict):
    remapped = dict(cleaned)

    def _try_remap(prefix_from: str, prefix_to: str):
        nonlocal remapped
        keys = list(remapped.keys())
        for k in keys:
            if k.startswith(prefix_from):
                nk = prefix_to + k[len(prefix_from) :]
                if nk not in remapped:
                    remapped[nk] = remapped[k]
        return

    _try_remap("fc.", "last_linear.")
    _try_remap("classifier.", "last_linear.")
    _try_remap("head.", "last_linear.")
    _try_remap("regressor.", "last_linear.")
    return remapped


def _safe_load_state_dict(model, model_path, device, expect_head_keys=None):
    if model_path is None:
        return False
    state = torch.load(model_path, map_location=device)
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        return False

    cleaned = {}
    for k, v in state.items():
        nk = k
        for pref in ("model.", "module.", "net.", "network."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    cleaned = _remap_head_keys_to_last_linear(cleaned)

    loaded_ok = False
    try:
        model.load_state_dict(cleaned, strict=True)
        loaded_ok = True
    except Exception:
        msd = model.state_dict()
        compatible = {
            k: v for k, v in cleaned.items() if k in msd and v.shape == msd[k].shape
        }
        if len(compatible) == 0:
            return False
        model.load_state_dict(compatible, strict=False)
        loaded_ok = True

    if loaded_ok and expect_head_keys is not None:
        msd_keys = set(model.state_dict().keys())
        cleaned_keys = set(cleaned.keys())
        present = [
            hk for hk in expect_head_keys if (hk in msd_keys and hk in cleaned_keys)
        ]
        if len(present) == 0:
            print(
                f"WARNING: Checkpoint {os.path.basename(model_path)} did not contain expected head keys "
                f"{[hk for hk in expect_head_keys if hk in msd_keys]}; treating as NOT loaded."
            )
            return False
    return loaded_ok


def _quantile_thresholds_from_train_labels(train_labels: pd.Series):
    counts = train_labels.value_counts().sort_index()
    probs = (counts / counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0.0).values
    cuts = np.cumsum(probs)[:-1]
    cuts = np.clip(cuts, 1e-6, 1 - 1e-6)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = min(1 - 1e-6, cuts[i - 1] + 1e-6)
    return cuts


def _apply_thresholds(preds: np.ndarray, thr: np.ndarray):
    thr = np.asarray(thr, dtype=float)
    thr = np.sort(thr)
    return np.digitize(preds, thr, right=False).astype(int)


def _apply_quantile_binning(preds: np.ndarray, cuts: np.ndarray):
    preds = np.asarray(preds, dtype=float)
    preds = np.where(np.isfinite(preds), preds, np.nan)
    finite = preds[np.isfinite(preds)]
    if finite.size == 0:
        return np.zeros(preds.shape[0], dtype=int)
    qvals = np.quantile(finite, cuts)
    for i in range(1, len(qvals)):
        if qvals[i] <= qvals[i - 1]:
            qvals[i] = qvals[i - 1] + 1e-6
    filled = np.where(np.isfinite(preds), preds, np.nanmean(finite))
    return np.digitize(filled, qvals, right=True).astype(int)


def _fit_qwk_thresholds(
    y_true: np.ndarray, y_pred_cont: np.ndarray, init_thr=None, n_iter=10
):
    y_true = np.asarray(y_true, dtype=int)
    y_pred_cont = np.asarray(y_pred_cont, dtype=float)
    y_pred_cont = np.where(np.isfinite(y_pred_cont), y_pred_cont, np.nan)
    m = np.isfinite(y_pred_cont)
    if m.sum() < 50:
        return None

    y_true = y_true[m]
    y_pred_cont = y_pred_cont[m]

    if init_thr is None:
        cuts = _quantile_thresholds_from_train_labels(pd.Series(y_true))
        init_thr = np.quantile(y_pred_cont, cuts)

    thr = np.sort(np.asarray(init_thr, dtype=float))
    if thr.shape[0] != 4:
        return None

    def score(thr_):
        y_hat = _apply_thresholds(y_pred_cont, thr_)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best = score(thr)
    step = float(np.std(y_pred_cont) + 1e-6) * 0.25
    for _ in range(n_iter):
        improved = False
        for i in range(4):
            for direction in (-1.0, 1.0):
                cand = thr.copy()
                cand[i] = cand[i] + direction * step
                cand = np.sort(cand)
                sc = score(cand)
                if sc > best:
                    thr, best = cand, sc
                    improved = True
        step *= 0.5
        if not improved:
            continue
    return thr




## === cell 6
densenet_filenames = [
    "model_densenet121_bs64_30.pth",
    "model_densenet121_bs64_30.pt",
    "model_densenet121_bs64_30.bin",
    "densenet121_bs64_30.pth",
    "densenet121.pth",
]
densenet_patterns = [
    "*densenet121*bs64*30*.pth",
    "*densenet121*bs64*30*.pt",
    "*densenet121*bs64*30*.bin",
    "*densenet121*.pth",
    "*densenet121*.pt",
    "*densenet121*.bin",
]
densenet_candidates = [
    "../input/densenet121/model_densenet121_bs64_30.pth",
    "/kaggle/input/densenet121/model_densenet121_bs64_30.pth",
    _find_any_in_kaggle_trees(densenet_filenames),
    _find_by_patterns_in_kaggle_trees(densenet_patterns),
]
MODEL_PATH = _resolve_model_path(densenet_candidates)
print("DenseNet checkpoint:", MODEL_PATH)

predictions_densenet = None
densenet_head_loaded = False
model = get_densenet121_gem(pretrain=("imagenet" if MODEL_PATH is None else None))
model.to(device)
if MODEL_PATH is not None:
    densenet_head_loaded = _safe_load_state_dict(
        model,
        MODEL_PATH,
        device=device,
        expect_head_keys=["last_linear.weight", "last_linear.bias"],
    )
print("DenseNet head loaded:", densenet_head_loaded)

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_densenet = make_predictions(
    model, test_images, norm, size=224, device=device
)



## === cell 7
seres256_filenames = [
    "fine_tune_256_model30.pth",
    "fine_tune_256_model30.pt",
    "fine_tune_256_model30.bin",
    "model30_256.pth",
    "seresnet50_256.pth",
]
seres256_patterns = [
    "*fine*tune*256*model30*.pth",
    "*fine*tune*256*model30*.pt",
    "*fine*tune*256*model30*.bin",
    "*model30*256*.pth",
    "*seres*256*.pth",
    "*se_resnet*256*.pth",
    "*se-resnet*256*.pth",
    "*seresnet50*256*.pth",
]
seres256_candidates = [
    "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth",
    "../input/seresnet50pretrain/fine_tune_256_model30.pth",
    _find_any_in_kaggle_trees(seres256_filenames),
    _find_by_patterns_in_kaggle_trees(seres256_patterns),
]
MODEL_PATH = _resolve_model_path(seres256_candidates)
print("SE-ResNet50-256 checkpoint:", MODEL_PATH)

predictions_seresnet = None
seres256_head_loaded = False
model = get_se_resnet50_gem(pretrain=("imagenet" if MODEL_PATH is None else None))
model.to(device)
if MODEL_PATH is not None:
    seres256_head_loaded = _safe_load_state_dict(
        model,
        MODEL_PATH,
        device=device,
        expect_head_keys=["last_linear.weight", "last_linear.bias"],
    )
print("SE-ResNet50-256 head loaded:", seres256_head_loaded)

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_seresnet = make_predictions(
    model, test_images, norm, size=256, device=device
)



## === cell 8
seres512_filenames = [
    "model30_512.pth",
    "model30_512.pt",
    "model30_512.bin",
    "fine_tune_512_model30.pth",
    "seresnet50_512.pth",
]
seres512_patterns = [
    "*fine*tune*512*model30*.pth",
    "*fine*tune*512*model30*.pt",
    "*fine*tune*512*model30*.bin",
    "*model30*512*.pth",
    "*seres*512*.pth",
    "*se_resnet*512*.pth",
    "*se-resnet*512*.pth",
    "*seresnet50*512*.pth",
]
seres512_candidates = [
    "../input/seresnet50-512/model30_512.pth",
    "/kaggle/input/seresnet50-512/model30_512.pth",
    _find_any_in_kaggle_trees(seres512_filenames),
    _find_by_patterns_in_kaggle_trees(seres512_patterns),
]
MODEL_PATH = _resolve_model_path(seres512_candidates)
print("SE-ResNet50-512 checkpoint:", MODEL_PATH)

predictions_seresnet_512 = None
seres512_head_loaded = False
model = get_se_resnet50_gem(pretrain=("imagenet" if MODEL_PATH is None else None))
model.to(device)
if MODEL_PATH is not None:
    seres512_head_loaded = _safe_load_state_dict(
        model,
        MODEL_PATH,
        device=device,
        expect_head_keys=["last_linear.weight", "last_linear.bias"],
    )
print("SE-ResNet50-512 head loaded:", seres512_head_loaded)

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_seresnet_512 = make_predictions(
    model, test_images, norm, size=512, device=device
)



## === cell 9
pred_dfs = []
if predictions_densenet is not None:
    pred_dfs.append(predictions_densenet.rename(columns={"pred": "pred_densenet"}))
if predictions_seresnet is not None:
    pred_dfs.append(predictions_seresnet.rename(columns={"pred": "pred_seresnet_256"}))
if predictions_seresnet_512 is not None:
    pred_dfs.append(
        predictions_seresnet_512.rename(columns={"pred": "pred_seresnet_512"})
    )

any_head_loaded = bool(
    densenet_head_loaded or seres256_head_loaded or seres512_head_loaded
)
majority_class = int(train_df["diagnosis"].value_counts().idxmax())

if len(pred_dfs) == 0:
    submission = pd.read_csv(SAMPLE_SUB_PATH)[["id_code", "diagnosis"]].copy()
    submission["diagnosis"] = majority_class
else:
    merged = pred_dfs[0]
    for df in pred_dfs[1:]:
        merged = merged.merge(df, on="id_code", how="outer")

    submission = test_df[["id_code"]].merge(merged, on="id_code", how="left")

    if not any_head_loaded:
        submission["diagnosis"] = majority_class
        submission = submission[["id_code", "diagnosis"]]
    else:
        pred_cols = [c for c in submission.columns if c.startswith("pred_")]

        for c in pred_cols:
            submission[c] = submission[c].astype(float)
            submission[c] = submission[c].fillna(submission[c].mean())

        submission["pred_mean"] = submission[pred_cols].mean(axis=1).astype(float)

        train_pred_mean = None
        thresholds = None
        try:
            train_images = [
                os.path.join(TRAIN_IMAGE_PATH, f"{i}.png")
                for i in train_df["id_code"].tolist()
            ]

            train_pred_parts = []
            if densenet_head_loaded:
                model_dn = get_densenet121_gem(pretrain=None).to(device)
                _safe_load_state_dict(
                    model_dn,
                    _resolve_model_path(densenet_candidates),
                    device=device,
                    expect_head_keys=["last_linear.weight", "last_linear.bias"],
                )
                train_pred_parts.append(
                    make_predictions(
                        model_dn, train_images, norm, size=224, device=device
                    ).rename(columns={"pred": "pred_densenet"})
                )
            if seres256_head_loaded:
                model_s256 = get_se_resnet50_gem(pretrain=None).to(device)
                _safe_load_state_dict(
                    model_s256,
                    _resolve_model_path(seres256_candidates),
                    device=device,
                    expect_head_keys=["last_linear.weight", "last_linear.bias"],
                )
                train_pred_parts.append(
                    make_predictions(
                        model_s256, train_images, norm, size=256, device=device
                    ).rename(columns={"pred": "pred_seresnet_256"})
                )
            if seres512_head_loaded:
                model_s512 = get_se_resnet50_gem(pretrain=None).to(device)
                _safe_load_state_dict(
                    model_s512,
                    _resolve_model_path(seres512_candidates),
                    device=device,
                    expect_head_keys=["last_linear.weight", "last_linear.bias"],
                )
                train_pred_parts.append(
                    make_predictions(
                        model_s512, train_images, norm, size=512, device=device
                    ).rename(columns={"pred": "pred_seresnet_512"})
                )

            if len(train_pred_parts) > 0:
                trm = train_pred_parts[0]
                for df in train_pred_parts[1:]:
                    trm = trm.merge(df, on="id_code", how="outer")
                trm = train_df[["id_code", "diagnosis"]].merge(
                    trm, on="id_code", how="left"
                )

                tr_pred_cols = [c for c in trm.columns if c.startswith("pred_")]
                for c in tr_pred_cols:
                    trm[c] = trm[c].astype(float)
                    trm[c] = trm[c].fillna(trm[c].mean())

                train_pred_mean = trm[tr_pred_cols].mean(axis=1).values.astype(float)
                init_thr = np.quantile(
                    train_pred_mean,
                    _quantile_thresholds_from_train_labels(train_df["diagnosis"]),
                )
                thresholds = _fit_qwk_thresholds(
                    train_df["diagnosis"].values,
                    train_pred_mean,
                    init_thr=init_thr,
                    n_iter=10,
                )
        except Exception as e:
            thresholds = None

        if thresholds is not None and np.all(np.isfinite(thresholds)):
            submission["diagnosis"] = _apply_thresholds(
                submission["pred_mean"].values.astype(float), thresholds
            )
        else:
            cuts = _quantile_thresholds_from_train_labels(train_df["diagnosis"])
            submission["diagnosis"] = _apply_quantile_binning(
                submission["pred_mean"].values.astype(float), cuts
            )

        submission = submission[["id_code", "diagnosis"]]
    submission["diagnosis"] = submission["diagnosis"].clip(0, 4).astype(int)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
submission["diagnosis"] = (
    submission["diagnosis"].fillna(majority_class).clip(0, 4).astype(int)
)

assert (
    submission.shape[0] == test_df.shape[0]
), "Submission row count mismatch vs test.csv"
assert (
    submission["id_code"].tolist() == test_df["id_code"].tolist()
), "Submission id_code order mismatch"

submission.to_csv("submission.csv", index=False)
print("Any fine-tuned head loaded:", any_head_loaded)
print("Majority class:", majority_class)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
