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

0.8980419219614995

# 6. Current score

0.05224

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fixed the CUDA‑only code so it runs on CPU, corrected the model‑loading paths, added safe fall‑backs for missing weight files, and ensured the prediction list is created before building the submission file.'
- What this solution (achieved 0.05224) has done: 'I load the ImageNet‑pretrained weights for the DenseNet‑121 backbone (instead of a randomly‑initialised model) and replace the manual threshold bins with a simple rounding‑and‑clipping step, which aligns the regression output to the 0‑4 class range and typically moves the quadratic‑weighted‑kappa closer to the target. This keeps the original model architecture and prediction loop unchanged while fixing the only‑zero predictions issue.'
- What this solution (achieved 0.05224) has done: 'I add proper ImageNet normalization to the image transform (so the pretrained DenseNet receives inputs in the range it expects) and broaden the search for the pretrained checkpoint by checking an additional common input directory. These minimal changes keep the original architecture and prediction logic intact while likely improving the model’s predictions, moving the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.05224) has done: 'I add a lightweight training step that fine‑tunes only the final linear layer of the pretrained DenseNet‑121 (with GeM pooling) on the available training data when a pretrained checkpoint is not found. This keeps the original architecture unchanged, adds only a few epochs of regression training on the true labels, and should move the quadratic‑weighted‑kappa score much closer to the target without altering the rest of the pipeline.'
- What this solution (achieved 0.05224) has done: 'I added a modestly longer fine‑tuning phase (10 epochs instead of 3) and a simple learning‑rate step scheduler so the pretrained DenseNet‑121 + GeM backbone can adapt a bit better to the DR labels. The rest of the pipeline, model architecture, and prediction handling stay unchanged, ensuring the script still produces a valid `submission.csv` while moving the quadratic‑weighted‑kappa score upward toward the target.'
- What this solution (achieved 0.05224) has done: 'I enhance the training phase used when no pretrained checkpoint is found: add simple data‑augmentation, train the final linear layer for a few epochs, then unfreeze the whole DenseNet‑121 + GeM backbone and fine‑tune it a bit longer with a lower learning rate. These modest changes keep the original model architecture and loss unchanged but give the network more capacity to learn the DR grades, which should raise the quadratic‑weighted‑kappa score toward the target. I also keep the same prediction‑and‑submission logic so a valid `submission.csv` is still produced.'
- What this solution (achieved 0.05224) has done: 'I increase the fine‑tuning length and add a light colour‑jitter augmentation, both of which can improve the model’s ability to predict the DR grades and therefore raise the quadratic weighted kappa score toward the target while keeping the overall architecture and training logic unchanged.'
- What this solution (achieved 0.05224) has done: 'I modify the script to (1) change the DenseNet‑GeM head to output 5 class logits and train it with a classification loss, (2) extend the fine‑tuning schedule for a stronger fit, and (3) adjust the prediction routine to use arg‑max on the averaged logits. These changes keep the original architecture (DenseNet‑121 + GeM) but replace the regression head with a lightweight classification head, which is known to improve Quadratic Weighted Kappa. I also make checkpoint loading tolerant to shape mismatches so training always proceeds.'
- What this solution (achieved 0.05224) has done: 'I add a simple stratified train/validation split, evaluate the Quadratic Weighted Kappa on the validation set after each epoch, and keep the best‑performing model. I also extend the fine‑tuning epochs slightly (20 + 20) to give the network more learning opportunity while preserving the original architecture and loss. These minimal changes should move the score closer to the target.'
- What this solution (achieved 0.05224) has done: 'I add a centered crop to match the DenseNet input size, compute the quadratic weighted kappa using scikit‑learn’s reliable implementation (reset each epoch), and extend the second fine‑tuning phase to give the model more learning time. These small tweaks keep the original architecture and training loop intact while likely improving the validation QWK and moving the score toward the target.'
- What this solution (achieved 0.05224) has done: 'I add class‑balance weighting to the cross‑entropy loss (computed from the full training set) so the model learns more from under‑represented DR grades, and I modestly increase the second fine‑tuning phase to 60 epochs to give the weighted training a bit more time to converge. These small, targeted changes keep the original architecture and overall pipeline intact while moving the validation quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.05224) has done: 'I adjust the prediction logic to use the soft‑max expected value (rounded and clipped) instead of a plain arg‑max, both during validation (so the best checkpoint is chosen with a metric that reflects rating proximity) and for the final test‑set predictions. This small change keeps the model architecture and training unchanged while better aligning outputs with the Quadratic Weighted Kappa metric, helping move the score toward the target.'
- What this solution (achieved 0.05224) has done: 'I slightly extend the second fine‑tuning phase (more epochs and a slower LR decay) which lets the pretrained DenseNet‑121 + GeM head learn the DR grades better while keeping the exact model architecture and training loop unchanged. This modest increase should raise the validation Quadratic Weighted Kappa and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import types, re
import torchmetrics
from sklearn.metrics import cohen_kappa_score  # added for reliable QWK computation



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
    assert num_classes == settings["num_classes"]
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
        return self.logits(x)

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def alexnet(num_classes=1000, pretrained="imagenet"):
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["alexnet"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    return modify_alexnet(model)


def modify_densenets(model):
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, input):
        x = self.features(input)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(lambda self, x: self.logits(x), model)
    return model


def densenet121(num_classes=1000, pretrained="imagenet"):
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    return modify_densenets(model)




## === cell 2
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_densenet121_gem(pretrain):
    """Return DenseNet‑121 with GeM pooling and a 5‑class classification head."""
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 5)
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*.png")))


## === cell 4
model_path_candidates = [
    "/kaggle/input/densenet121/model_densenet121_bs64_30.pth",
    "../input/densenet121/model_densenet121_bs64_30.pth",
    "/kaggle/input/aptos2019-blindness-detection/model_densenet121_bs64_30.pth",
    "../input/aptos2019-blindness-detection/model_densenet121_bs64_30.pth",
]
model_path = None
for cand in model_path_candidates:
    if os.path.isfile(cand):
        model_path = cand
        break

try:
    model = get_densenet121_gem(pretrain="imagenet")
    model.to(device)

    if model_path is not None:
        try:
            state = torch.load(model_path, map_location=device)
            model_state = model.state_dict()
            filtered_state = {
                k: v
                for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model_state.update(filtered_state)
            model.load_state_dict(model_state)
            print("Loaded external checkpoint (compatible parts).")
        except Exception as e:
            print(
                f"Warning: checkpoint could not be loaded ({e}); training from scratch."
            )
    else:
        print("No checkpoint found – performing fine‑tuning on the training set.")

    train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

    full_df = pd.read_csv(train_csv)

    class_counts = full_df["diagnosis"].value_counts().sort_index()
    class_weights = (1.0 / class_counts).values
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_df = full_df.groupby("diagnosis", group_keys=False).apply(
        lambda x: x.sample(frac=0.8, random_state=42)
    )
    val_df = pd.concat([full_df, train_df]).drop_duplicates(keep=False)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class AptosDataset(torch.utils.data.Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = torch.tensor(row["diagnosis"], dtype=torch.long)
            return img, label

    train_dataset = AptosDataset(train_df, train_img_dir, train_transform)
    val_dataset = AptosDataset(val_df, train_img_dir, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    for name, param in model.named_parameters():
        if "last_linear" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.last_linear.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_qwk = -1.0
    best_state = None

    epochs_stage1 = 20
    for epoch in range(epochs_stage1):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        scheduler.step()

        model.eval()
        all_preds = []
        all_tgts = []
        with torch.no_grad():
            for v_imgs, v_labels in val_loader:
                v_imgs = v_imgs.to(device)
                v_labels = v_labels.to(device)
                logits = model(v_imgs)
                probs = F.softmax(logits, dim=1)
                expected = torch.sum(
                    probs * torch.arange(5, device=device, dtype=probs.dtype), dim=1
                )
                preds = torch.clamp(torch.round(expected), 0, 4).long()
                all_preds.append(preds.cpu())
                all_tgts.append(v_labels.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_tgts = torch.cat(all_tgts).numpy()
        val_qwk = cohen_kappa_score(all_tgts, all_preds, weights="quadratic")
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_state = model.state_dict()
        print(
            f"Stage1 epoch {epoch+1}/{epochs_stage1}, loss: {epoch_loss/len(train_dataset):.4f}, val QWK: {val_qwk:.4f}"
        )

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    epochs_stage2 = 120
    for epoch in range(epochs_stage2):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        scheduler.step()

        model.eval()
        all_preds = []
        all_tgts = []
        with torch.no_grad():
            for v_imgs, v_labels in val_loader:
                v_imgs = v_imgs.to(device)
                v_labels = v_labels.to(device)
                logits = model(v_imgs)
                probs = F.softmax(logits, dim=1)
                expected = torch.sum(
                    probs * torch.arange(5, device=device, dtype=probs.dtype), dim=1
                )
                preds = torch.clamp(torch.round(expected), 0, 4).long()
                all_preds.append(preds.cpu())
                all_tgts.append(v_labels.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_tgts = torch.cat(all_tgts).numpy()
        val_qwk = cohen_kappa_score(all_tgts, all_preds, weights="quadratic")
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_state = model.state_dict()
        print(
            f"Stage2 epoch {epoch+1}/{epochs_stage2}, loss: {epoch_loss/len(train_dataset):.4f}, val QWK: {val_qwk:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best validation QWK achieved: {best_qwk:.4f}")

    norm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
except Exception as e:
    print(
        f"Warning: model could not be loaded ({e}); using simple brightness fallback."
    )
    model = None
    norm = transforms.Compose([transforms.ToTensor()])




## === cell 5
def make_predictions(model, test_images, transform, size=256, device=device):
    """Predict class indices (0‑4) using averaged soft‑max expected value of original & flipped image."""
    model.eval()
    predictions = []
    for im_path in test_images:
        img = Image.open(im_path).convert("RGB")
        img = img.resize((size, size), resample=Image.BILINEAR)
        img = transform(img).to(device)
        with torch.no_grad():
            out = model(img.unsqueeze(0))
            out_f = model(torch.flip(img.unsqueeze(0), dims=(3,)))
            avg_logits = (out + out_f) / 2.0
            probs = F.softmax(avg_logits, dim=1)
            expected = torch.sum(
                probs * torch.arange(5, device=device, dtype=probs.dtype), dim=1
            )
            pred_class = int(torch.clamp(torch.round(expected), 0, 4).item())
        predictions.append((os.path.splitext(os.path.basename(im_path))[0], pred_class))
    return predictions


if model is not None:
    predictions = make_predictions(model, test_images, norm, size=256, device=device)
else:
    predictions = []
    for p in test_images:
        img = Image.open(p).convert("L")
        arr = np.array(img).astype(np.float32)
        mean_val = arr.mean() / 255.0 * 4.0  # scale to [0,4]
        pred_cls = int(np.clip(np.round(mean_val), 0, 4))
        predictions.append((os.path.splitext(os.path.basename(p))[0], pred_cls))


## === cell 6
submission = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].astype(int)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
print(submission.head())
