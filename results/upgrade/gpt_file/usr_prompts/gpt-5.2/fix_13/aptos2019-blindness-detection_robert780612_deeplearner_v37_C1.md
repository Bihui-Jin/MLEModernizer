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

0.9211016047484492

# 6. Current score

0.16148

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end in the Kaggle environment and always writes a valid `submission.csv`. The current failure is caused by a missing external checkpoint dataset; instead of crashing, I fall back to using an ImageNet-pretrained SE-ResNet50-GeM model (same architecture/inference loop) to generate continuous predictions. I also make the pretrained-weight loading offline-safe (no internet) by using torchvision’s built-in weights when available, and ensure deterministic, correctly ordered predictions merged back onto `test_df`. These changes are strictly to remove runtime errors and produce a valid submission; score likely be below the target without the original checkpoint, but it at least yield a submission.'
- What this solution (achieved 0.00381) has done: 'The current 0.0 score is most consistent with the submission being effectively random (no real DR-trained checkpoint loaded) and/or miscalibrated discretization for QWK, even though it writes a valid CSV. To move the score upward toward the 0.921 target with minimal core-logic change, I keep your exact model/inference loop and add a tiny, training-free post-processing step: optimize the 4 rounding thresholds on the training set using out-of-fold predictions from the same model (no label leakage to test, and no training). This aligns continuous outputs to the ordinal labels specifically for QWK and typically yields a large jump versus fixed thresholds when the raw regressor scale is off. I also make sure predictions are computed in the exact `test_df` order and add a small batched inference path (same semantics) to stay within time.'
- What this solution (achieved 0.1237) has done: 'Your current 0.00381 score indicates the model outputs are not on the same scale/order as the DR labels, so the threshold optimizer is searching in the wrong region. I keep your exact model + inference loop and only change the threshold-fitting step to first *calibrate* the raw predictions with an optimal affine transform (a·pred+b) using cross-validation, then fit thresholds on the calibrated predictions. This is training-free (no backprop), uses only train labels, and directly targets QWK alignment, which should move the score substantially upward toward the 0.921 target without changing the core modeling. I also ensure thresholds are learned around the calibrated label scale (near 0–4) and apply the same calibration to test predictions before discretization.'
- What this solution (achieved 0.27582) has done: 'Your current score is far below the target (gap ≈ -0.7974), so we should increase performance with minimal, metric-aligned changes while keeping your model/inference loop intact. The biggest issue is that your “OOF calibration” is not actually OOF (you fit a→y then apply to val, which leaks within folds), and the threshold search is centered around 0–4 even if your raw outputs live on a very different scale. I (1) replace the calibration with a true OOF isotonic regression (monotonic, training-free, strong for ordinal/QWK) and then refit thresholds on those calibrated OOF predictions, and (2) make the threshold search initialize from data-driven quantiles of the calibrated OOF predictions (so it searches in the right region). Everything else (model, TTA, image resize/normalize, inference batching, submission format/path) stays the same.'
- What this solution (achieved 0.0) has done: 'The score gap is large (0.27582 vs target 0.9211), so we need a meaningful lift without changing your core model/inference logic. The most likely bottleneck is that the isotonic + threshold search is still misaligned with QWK because it calibrates on raw outputs directly and searches thresholds on noisy OOF without stabilizing across folds. I keep the same model, image preprocessing, TTA, and prediction loop, but (1) make the OOF calibration truly fold-wise for both isotonic and thresholds (thresholds learned on calibrated OOF, not raw OOF), and (2) replace the coarse coordinate-search with an exact, fast 1D dynamic-programming optimal binning on calibrated OOF predictions to maximize QWK surrogate agreement (stable for ordinal mapping), then evaluate QWK and write `submission.csv`. This remains training-free post-processing using only train labels and is designed to move QWK upward substantially while staying within the 600s budget.'
- What this solution (achieved 0.18306) has done: 'The current 0.0 score is most consistent with the submission containing a single class (or otherwise badly discretized predictions) due to your threshold DP optimizing a majority-vote objective that does not correlate with QWK. I keep your exact model, preprocessing, and inference loop unchanged, and only replace the threshold-fitting step with a small coordinate-descent search that directly maximizes QWK on the calibrated OOF predictions. To prevent the degenerate “all same label” submission, I also add a safe fallback to quantile-based thresholds if the optimizer produces non-finite or non-increasing cutoffs. This is metric-aligned post-processing (no training/backprop, no leakage to test) and should move the score upward toward the target while remaining within the time budget and still writing a valid `submission.csv`.'
- What this solution (achieved 0.1463) has done: 'Your current score (0.18306) is far below the target (0.9211), so we should increase performance with minimal, metric-aligned changes while keeping your exact model/inference loop intact. The biggest likely issue is that you’re fitting isotonic on raw single-model predictions that aren’t DR-trained (fallback ImageNet mapping), so the calibration+thresholds are unstable and can collapse; we stabilize the post-processing without changing the model by (1) using repeated stratified CV to get a smoother OOF calibrated prediction signal, and (2) fitting thresholds on the averaged OOF-calibrated predictions with a slightly stronger, safer QWK search that prevents degenerate class collapse. We also add a tiny “distribution guardrail” that nudges thresholds toward training label priors only if the optimizer yields an overly collapsed prediction distribution (keeps semantics; still purely post-processing on train labels). Everything still runs end-to-end and writes `submission.csv` with the correct columns/order.'
- What this solution (achieved 0.27263) has done: 'You’re far below the target (0.1463 vs 0.9211), so we should cautiously increase QWK with the smallest changes that don’t touch your model/inference core. The biggest current issue is that your isotonic calibration is fit on the full training set, which can distort generalization; we switch to a fold-ensemble isotonic calibration (average predictions from per-fold isotonic models on test), while keeping the same OOF calibration for threshold tuning. Next, we make the threshold search use a better starting point by anchoring initial cutoffs from the calibrated OOF distribution matched to label priors (less collapse than generic 20/40/60/80 quantiles). These are training-free post-processing changes only, and the script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.08638) has done: 'We’re far below the target (0.2726 vs 0.9211), so we should increase QWK using minimal, metric-aligned post-processing without touching your model/inference core. The main improvement is to fit thresholds more robustly by optimizing them on *OOF-calibrated* predictions using a small Nelder–Mead search directly on QWK (more reliable than the current coarse coordinate grid), while keeping your isotonic CV ensemble exactly as-is. To reduce overfitting and improve generalization, we also blend the optimized thresholds with label-prior quantile thresholds (a mild regularization that usually improves public LB stability). Everything still runs end-to-end, preserves the architecture/inference loop, and writes a valid `submission.csv` in the correct order/format.'
- What this solution (achieved 0.17147) has done: 'Your score is far below the target (0.08638 vs 0.9211), so we need a meaningful boost while keeping your model/inference core unchanged. The biggest current correctness issue is that the isotonic “test ensemble” is trained partly on folds where the same sample was used for fitting, and you then also use that same set of models to calibrate train/test, which can distort calibration and thresholds. I change calibration to a leakage-safe scheme: compute true OOF calibrated predictions for train, fit thresholds on those, and for test use the average of per-fold isotonic models (trained on each fold’s train split) without ever predicting a sample with a calibrator trained on itself. I also add a small, deterministic grid-refinement around the learned thresholds (still post-processing only) to improve QWK alignment without altering the network or inference loop.'
- What this solution (achieved 0.16148) has done: 'Your current gap to the target is large (0.17147 vs 0.9211), so we should increase score with the smallest changes that directly improve QWK while keeping your model/inference unchanged. The biggest likely issue is that isotonic calibration is being asked to learn a 5-class mapping from weak ImageNet-ish regression outputs using only the raw scalar, and that instability then propagates into thresholds. I keep the same per-fold isotonic + Nelder–Mead threshold optimization, but add a minimal, leakage-safe “affine pre-calibration” (fit a,b on each fold’s train split) before isotonic; this stabilizes isotonic and makes thresholds more meaningful for QWK. I also add a tiny amount of monotonic regularization by clipping calibrated predictions to [0,4] before threshold search, which matches the label domain and reduces degenerate solutions without changing semantics.'

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
    try:
        state_dict = model_zoo.load_url(settings["url"])
        state_dict = update_state_dict(state_dict)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(
            f"[WARN] Could not download/load pretrained weights from URL. Reason: {e}"
        )
        print("[WARN] Proceeding without those weights.")
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
    try:
        model.load_state_dict(model_zoo.load_url(settings["url"]))
    except Exception as e:
        print(
            f"[WARN] Could not download/load pretrained weights from URL. Reason: {e}"
        )
        print("[WARN] Proceeding without those weights.")
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
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
TEST_CSV_PATH = os.path.join(BASE_INPUT, "test.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_INPUT, "train_images")

test_df = pd.read_csv(TEST_CSV_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
    for id_code in test_df["id_code"].tolist()
]
train_images = [
    os.path.join(TRAIN_IMAGE_PATH, f"{id_code}.png")
    for id_code in train_df["id_code"].tolist()
]




## === cell 5
def make_predictions(
    model, image_paths, transforms, size=256, device=torch.device("cpu"), batch_size=16
):
    model.eval()
    ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    preds = []

    def _load_image(p):
        img = Image.open(p).convert("RGB")
        img = img.resize((size, size), resample=Image.BILINEAR)
        return transforms(img)

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_t = torch.stack([_load_image(p) for p in batch_paths], dim=0).to(
                device
            )

            out = model(batch_t).view(-1)
            out_flip = model(torch.flip(batch_t, dims=(3,))).view(-1)
            final = ((out + out_flip) / 2.0).detach().cpu().numpy().astype(np.float64)
            preds.extend(final.tolist())

    return list(zip(ids, preds))




## === cell 6
import warnings
import torchvision


def _find_checkpoint(rel_candidates):
    for p in rel_candidates:
        if os.path.exists(p):
            return p
    return None


MODEL_PATH = _find_checkpoint(
    [
        "/kaggle/input/seresnet50pseudo-512/model30.pth",
        "../input/seresnet50pseudo-512/model30.pth",
    ]
)

model = get_se_resnet50_gem(pretrain=False).to(device)

loaded_ckpt = False
if MODEL_PATH is not None:
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state)
        loaded_ckpt = True
        print(f"[INFO] Loaded checkpoint: {MODEL_PATH}")
    except Exception as e:
        warnings.warn(
            f"Failed to load checkpoint state_dict; will fall back. Reason: {e}"
        )

if not loaded_ckpt:
    try:
        from torchvision.models import resnet50 as tv_resnet50
        from torchvision.models import ResNet50_Weights

        tv = tv_resnet50(weights=ResNet50_Weights.DEFAULT)
        sd = tv.state_dict()

        mapped = {}
        for k, v in sd.items():
            if k.startswith("conv1."):
                mapped["layer0.conv1." + k.split("conv1.", 1)[1]] = v
            elif k.startswith("bn1."):
                mapped["layer0.bn1." + k.split("bn1.", 1)[1]] = v
            elif k.startswith("layer1."):
                mapped["layer1." + k.split("layer1.", 1)[1]] = v
            elif k.startswith("layer2."):
                mapped["layer2." + k.split("layer2.", 1)[1]] = v
            elif k.startswith("layer3."):
                mapped["layer3." + k.split("layer3.", 1)[1]] = v
            elif k.startswith("layer4."):
                mapped["layer4." + k.split("layer4.", 1)[1]] = v

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        print("[INFO] Fallback torchvision weights loaded (partial, strict=False).")
        if len(missing) > 0:
            print(f"[INFO] Missing keys (count={len(missing)}), e.g.: {missing[:5]}")
        if len(unexpected) > 0:
            print(
                f"[INFO] Unexpected keys (count={len(unexpected)}), e.g.: {unexpected[:5]}"
            )
    except Exception as e:
        warnings.warn(
            f"Could not load fallback torchvision weights. Proceeding random init. Reason: {e}"
        )

model.eval()

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

preds_train = make_predictions(
    model, train_images, norm, size=512, device=device, batch_size=16
)
preds_test = make_predictions(
    model, test_images, norm, size=512, device=device, batch_size=16
)

train_pred_df = pd.DataFrame(preds_train, columns=["id_code", "pred"])
test_pred_df = pd.DataFrame(preds_test, columns=["id_code", "pred"])

train_pred_df = train_df[["id_code", "diagnosis"]].merge(
    train_pred_df, on="id_code", how="left"
)
test_pred_df = test_df[["id_code"]].merge(test_pred_df, on="id_code", how="left")

if train_pred_df["pred"].isna().any():
    missing = (
        train_pred_df.loc[train_pred_df["pred"].isna(), "id_code"].head(10).tolist()
    )
    raise RuntimeError(
        f"Missing train predictions for some ids (showing up to 10): {missing}"
    )
if test_pred_df["pred"].isna().any():
    missing = test_pred_df.loc[test_pred_df["pred"].isna(), "id_code"].head(10).tolist()
    raise RuntimeError(
        f"Missing test predictions for some ids (showing up to 10): {missing}"
    )



## === cell 7
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize


def apply_thresholds(pred, thr):
    t0, t1, t2, t3 = thr
    pred = np.asarray(pred, dtype=np.float64)
    out = np.zeros_like(pred, dtype=np.int64)
    out[pred >= t0] = 1
    out[pred >= t1] = 2
    out[pred >= t2] = 3
    out[pred >= t3] = 4
    return out


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _thresholds_from_train_label_priors(pred_cal, y, eps=1e-6):
    pred_cal = np.asarray(pred_cal, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    props = np.bincount(y, minlength=5).astype(np.float64)
    props = props / np.maximum(props.sum(), 1.0)
    cdf = np.cumsum(props)
    qs = (cdf[0], cdf[1], cdf[2], cdf[3])
    thr = np.quantile(pred_cal, qs).astype(np.float64)
    thr = np.sort(thr)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + eps:
            thr[i] = thr[i - 1] + eps
    return thr


def _make_strictly_increasing(thr, eps=1e-6):
    thr = np.asarray(thr, dtype=np.float64)
    thr = np.sort(thr)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + eps:
            thr[i] = thr[i - 1] + eps
    return thr


def fit_affine(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_var = float(((x - x_mean) ** 2).mean())
    if x_var <= 1e-12:
        a = 1.0
    else:
        a = float(((x - x_mean) * (y - y_mean)).mean() / x_var)
    b = float(y_mean - a * x_mean)
    return a, b


def fit_isotonic_oof_and_test_ensemble(
    pred_train, y_train, pred_test, n_splits=5, seed=42
):
    x_tr = np.asarray(pred_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    x_te = np.asarray(pred_test, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_cal = np.zeros_like(x_tr, dtype=np.float64)
    test_sum = np.zeros_like(x_te, dtype=np.float64)

    for tr_idx, va_idx in skf.split(x_tr, y_tr.astype(np.int64)):
        a, b = fit_affine(x_tr[tr_idx], y_tr[tr_idx])
        x_tr_aff = a * x_tr + b
        x_te_aff = a * x_te + b

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(x_tr_aff[tr_idx], y_tr[tr_idx])

        oof_cal[va_idx] = ir.predict(x_tr_aff[va_idx]).astype(np.float64)
        test_sum += ir.predict(x_te_aff).astype(np.float64)

    test_cal = test_sum / float(n_splits)
    return oof_cal, test_cal


def fit_thresholds_by_qwk_nelder_mead(pred_cal_oof, y_true, init_thr, maxiter=260):
    x = np.asarray(pred_cal_oof, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)

    init_thr = _make_strictly_increasing(init_thr)

    def objective(t):
        t = _make_strictly_increasing(t)
        return -qwk(y, apply_thresholds(x, t))

    res = minimize(
        objective,
        x0=init_thr,
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-5, "fatol": 1e-5, "disp": False},
    )
    thr = _make_strictly_increasing(res.x)
    score = qwk(y, apply_thresholds(x, thr))
    return thr, float(score), res.success


def refine_thresholds_local_grid(
    pred_cal_oof, y_true, thr, steps=(0.02, 0.01), width=3
):
    x = np.asarray(pred_cal_oof, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    best_thr = _make_strictly_increasing(thr)
    best = qwk(y, apply_thresholds(x, best_thr))

    for step in steps:
        for _ in range(2):  # a couple passes
            for j in range(4):
                base = best_thr.copy()
                candidates = []
                for k in range(-width, width + 1):
                    t = base.copy()
                    t[j] = t[j] + k * step
                    t = _make_strictly_increasing(t)
                    candidates.append(t)
                for t in candidates:
                    s = qwk(y, apply_thresholds(x, t))
                    if s > best:
                        best = s
                        best_thr = t
    return best_thr, float(best)


y_train = train_pred_df["diagnosis"].values.astype(np.int64)
p_train = train_pred_df["pred"].values.astype(np.float64)
p_test = test_pred_df["pred"].values.astype(np.float64)

p_train_oof_cal, p_test_cal = fit_isotonic_oof_and_test_ensemble(
    p_train, y_train, p_test, n_splits=5, seed=42
)

p_train_oof_cal = np.clip(p_train_oof_cal, 0.0, 4.0)
p_test_cal = np.clip(p_test_cal, 0.0, 4.0)

thr_prior_oof = _thresholds_from_train_label_priors(p_train_oof_cal, y_train)
thr_nm, oof_qwk_nm, ok = fit_thresholds_by_qwk_nelder_mead(
    p_train_oof_cal, y_train, init_thr=thr_prior_oof, maxiter=260
)

thr_ref, oof_qwk_ref = refine_thresholds_local_grid(
    p_train_oof_cal, y_train, thr_nm, steps=(0.02, 0.01), width=3
)

thr = 0.90 * thr_ref + 0.10 * thr_prior_oof
thr = _make_strictly_increasing(thr)

train_labels_tmp = apply_thresholds(p_train_oof_cal, thr)
train_counts = np.bincount(train_labels_tmp, minlength=5).astype(np.int64)
n_used = int((train_counts > 0).sum())
max_frac = float(train_counts.max() / max(1, train_counts.sum()))
if n_used < 3 or max_frac > 0.92 or (not np.all(np.isfinite(thr))):
    thr = _make_strictly_increasing(thr_prior_oof)

oof_qwk_final = qwk(y_train, apply_thresholds(p_train_oof_cal, thr))
print("[INFO] Threshold optimizer success:", bool(ok))
print("[INFO] OOF QWK after NM:", float(oof_qwk_nm))
print("[INFO] OOF QWK after local refine:", float(oof_qwk_ref))
print("[INFO] Learned thresholds (final):", thr.tolist())
print("[INFO] OOF QWK (indicative):", float(oof_qwk_final))

test_labels = apply_thresholds(p_test_cal, thr)

submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": test_labels.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "[INFO] Test label distribution:",
    submission["diagnosis"].value_counts().sort_index().to_dict(),
)
