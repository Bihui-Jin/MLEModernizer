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

0.906439070074833

# 6. Current score

0.40035

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the shape mismatch that caused the model forward pass to fail by using the image size (224 px) that matches the DenseNet feature map expected by the fixed‑size average pooling in the logits layer. This resolves the runtime error, enables the predictions variable to be created, and consequently allows the submission CSV to be generated correctly. No other logic is altered, preserving the original model architecture and training approach.'
- What this solution (achieved 0.0) has done: 'I adjust the model loading to use `strict=False` so any compatible pretrained weights are applied even if the architecture differs, which should give better feature representations. Then I simplify the prediction‑to‑label conversion by clipping the regression output to the valid 0‑4 range and rounding, providing a more sensible mapping that is likely to raise the quadratic weighted kappa toward the target score. These changes keep the overall architecture and training logic unchanged while improving the quality of the generated submission.'
- What this solution (achieved -0.0071) has done: 'I fix the TypeError caused by using dictionary‑style indexing on the rows returned by `DataFrame.itertuples()`. The rows are namedtuples, so I access the `id_code` field with attribute syntax. This correction defines `final_predictions` correctly, allowing the submission CSV to be generated. No other logic is altered, preserving the model and calibration steps.'
- What this solution (achieved -0.324) has done: 'I prevent the SSL error by skipping the download of ImageNet weights: the model be instantiated with random ImageNet‐style initialization (which is still usable for inference) by calling `get_densenet121_gem(pretrain=None)`. This removes the failing URL request, lets `raw_predictions` be created, and consequently fixes the downstream NameError and submission generation while keeping the original architecture and calibration logic unchanged.'
- What this solution (achieved -0.29118) has done: 'I add a safe fallback for loading ImageNet weights: if the download fails (e.g., SSL error), the code keep the randomly‑initialized model instead of raising an exception. This prevents the earlier crash, allows `raw_predictions` to be generated, and ensures the subsequent calibration and CSV writing steps run, producing a valid `submission.csv`. No other logic is changed.'
- What this solution (achieved 0.64145) has done: 'The changes speed up image handling by batching predictions instead of processing each image one‑by‑one, cutting the many Python‑level I/O loops and duplicate model calls. The training DataLoader now uses multiple workers to preload images faster. These adjustments keep the exact model architecture, loss, and training epochs unchanged, so the final predictions remain identical apart from negligible floating‑point ordering differences.'
- What this solution (achieved 0.40035) has done: 'The script’s main slowdown is the custom image‑loading loop in `make_predictions`, which loads and resizes every image sequentially on the CPU. By switching to a `torch.utils.data.Dataset` together with a `DataLoader` that uses multiple workers and a transform that performs the resize, we keep the exact same prediction logic (including the horizontal‑flip averaging) while dramatically cutting I/O overhead. Minor tweaks like enabling cuDNN benchmark and pre‑defining the transform further speed up the forward passes without altering model architecture or training.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from glob import glob
import types, re

torch.backends.cudnn.benchmark = True



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
    "vgg19",
    "vgg19_bn",
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
}
input_sizes = {}
means = {}
stds = {}
for name in __all__:
    input_sizes[name] = [3, 224, 224]
    means[name] = [0.485, 0.456, 0.406]
    stds[name] = [0.229, 0.224, 0.225]
for name in ["inceptionv3"]:
    input_sizes[name] = [3, 299, 299]
    means[name] = [0.5, 0.5, 0.5]
    stds[name] = [0.5, 0.5, 0.5]
pretrained_settings = {}
for name in __all__:
    pretrained_settings[name] = {
        "imagenet": {
            "url": model_urls[name],
            "input_space": "RGB",
            "input_size": input_sizes[name],
            "input_range": [0, 1],
            "mean": means[name],
            "std": stds[name],
            "num_classes": 1000,
        }
    }


def update_state_dict(state_dict):
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for k in list(state_dict.keys()):
        m = pattern.match(k)
        if m:
            new_k = m.group(1) + m.group(2)
            state_dict[new_k] = state_dict[k]
            del state_dict[k]
    return state_dict


def load_pretrained(model, num_classes, settings):
    assert num_classes == settings["num_classes"]
    sd = torch.utils.model_zoo.load_url(settings["url"])
    sd = update_state_dict(sd)
    model.load_state_dict(sd)
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

    def features(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, f):
        x = self.relu1(f)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        return self.logits(self.features(x))

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def alexnet(num_classes=1000, pretrained="imagenet"):
    m = models.alexnet(pretrained=False)
    if pretrained:
        m = load_pretrained(m, num_classes, pretrained_settings["alexnet"][pretrained])
    return modify_alexnet(m)


def modify_densenets(model):
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, f):
        x = F.relu(f, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        return self.logits(self.features(x))

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet121(num_classes=1000, pretrained="imagenet"):
    m = models.densenet121(pretrained=False)
    if pretrained:
        try:
            m = load_pretrained(
                m, num_classes, pretrained_settings["densenet121"][pretrained]
            )
        except Exception as e:
            print(
                f"Warning: failed to load ImageNet pretrained weights ({e}), using random init."
            )
    return modify_densenets(m)


def modify_resnets(model):
    model.last_linear = model.fc
    model.fc = None

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, f):
        x = self.avgpool(f)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        return self.logits(self.features(x))

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    m = models.resnet50(pretrained=False)
    if pretrained:
        m = load_pretrained(m, num_classes, pretrained_settings["resnet50"][pretrained])
    return modify_resnets(m)




## === cell 2
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_densenet121_gem(pretrain):
    """
    Build a DenseNet‑121 with GeM pooling and a single regression head.
    If `pretrain` is "imagenet", we attempt to load ImageNet weights;
    on failure we fall back to random initialization.
    """
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

test_image_paths = [
    os.path.join(TEST_IMAGE_PATH, f"{row['id_code']}.png")
    for _, row in test_df.iterrows()
]

train_image_paths = [
    os.path.join(TRAIN_IMAGE_PATH, f"{row['id_code']}.png")
    for _, row in train_df.iterrows()
]

train_mean_diagnosis = train_df["diagnosis"].mean()
train_std_diagnosis = train_df["diagnosis"].std()




## === cell 4
class PredictionDataset(torch.utils.data.Dataset):
    """
    Simple dataset returning transformed tensors for a list of image paths.
    The transform should include a Resize to the desired size.
    """

    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


def make_predictions(
    model, img_paths, transform, size=224, device=device, batch_size=64, num_workers=4
):
    """
    Predict on a list of image paths using a DataLoader for parallel I/O.
    Original‑and‑horizontally‑flipped predictions are averaged, exactly
    as before.
    """
    model.eval()
    preds = []

    if not any(isinstance(t, transforms.Resize) for t in transform.transforms):
        transform = transforms.Compose(
            [transforms.Resize((size, size)), *transform.transforms]
        )

    dataset = PredictionDataset(img_paths, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        for batch_tensor in loader:
            batch_tensor = batch_tensor.to(device, non_blocking=True)

            out = model(batch_tensor)  # (B, 1)
            out_flip = model(torch.flip(batch_tensor, dims=(3,)))  # horizontal flip
            batch_pred = (out.squeeze(1) + out_flip.squeeze(1)) / 2.0
            preds.extend(batch_pred.cpu().tolist())
    return preds




## === cell 5
FINE_TUNED_PATH = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"

if os.path.exists(FINE_TUNED_PATH):
    try:
        model = get_densenet121_gem(pretrain="imagenet")
        model.load_state_dict(
            torch.load(FINE_TUNED_PATH, map_location=device), strict=False
        )
        print("Loaded fine‑tuned weights.")
    except Exception as e:
        print(
            f"Warning: could not load fine‑tuned checkpoint ({e}); using ImageNet fallback."
        )
        model = get_densenet121_gem(pretrain="imagenet")
else:
    print("Fine‑tuned weight file not found; using ImageNet (or random) backbone.")
    model = get_densenet121_gem(pretrain="imagenet")

model.to(device)

if not os.path.exists(FINE_TUNED_PATH):
    print("Starting quick training of the regression head on the training set.")
    norm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class TrainDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths, labels, transform):
            self.img_paths = img_paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert("RGB")
            tensor = self.transform(img)
            label = self.labels[idx]
            return tensor, torch.tensor(label, dtype=torch.float32)

    train_dataset = TrainDataset(
        train_image_paths,
        train_df["diagnosis"].values,
        norm,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    epochs = 5  # increased from 3 to give the head a bit more learning
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(
                device, non_blocking=True
            ).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader.dataset):.4f}"
        )

    model.eval()
else:
    norm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

raw_predictions = make_predictions(
    model, test_image_paths, norm, size=224, device=device, batch_size=64, num_workers=4
)

train_predictions = make_predictions(
    model,
    train_image_paths,
    norm,
    size=224,
    device=device,
    batch_size=64,
    num_workers=4,
)

slope, intercept = np.polyfit(train_predictions, train_df["diagnosis"].values, 1)

calibrated_predictions = (np.array(raw_predictions) * slope) + intercept



## === cell 6
final_predictions = list(zip(test_df["id_code"], calibrated_predictions))



## === cell 7
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = np.clip(submission["diagnosis"], 0, 4)
submission["diagnosis"] = np.rint(submission["diagnosis"]).astype(int)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
