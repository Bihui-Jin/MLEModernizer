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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We adjust the script to run on CPU when CUDA is unavailable, safely handle missing model files by falling back to a trivial constant‑output model, and ensure a `submission.csv` is always created with the required columns.'
- What this solution (achieved 0.02681) has done: 'I keep the overall pipeline unchanged but replace the constant‑zero fallback model with a simple random‑output model so the predictions are no longer all the same class. This introduces variance in the forecasts, which raise the quadratic weighted kappa from 0 toward the target score while preserving the existing architecture and inference logic.'
- What this solution (achieved 0.0) has done: 'The script now imports the missing `types` module, avoids the failing online download by falling back to an un‑pretrained Densenet‑121 when the pretrained weights cannot be retrieved, and safely computes class centroids even if a class has no samples (preventing NaNs). These changes stop the SSL error, ensure a model and centroids are always available, and guarantee that a `submission.csv` file is produced with valid predictions.'
- What this solution (achieved 0.0) has done: 'I ensure the model actually loads ImageNet‑pretrained weights (falling back to a randomly‑initialized model only if that fails) and normalize the feature vectors before comparing them to the class centroids. This keeps the overall architecture and inference logic unchanged while giving the embeddings meaningful geometry, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.05244) has done: 'I add a tiny test‑time augmentation by also embedding the horizontally‑flipped version of each image and averaging the two embeddings before finding the nearest centroid. This keeps the core model and centroid logic unchanged, but gives a modest boost in prediction quality which should move the quadratic weighted kappa score closer to the target. I also import the missing `ImageOps` utility.'
- What this solution (achieved 0.16236) has done: 'I add a lightweight linear classifier that is trained on the Densenet‑121 embeddings of the training set and then use this classifier (instead of the simple centroid lookup) to predict the test labels. This keeps the original embedding extraction unchanged, adds only a small training loop, and is expected to raise the quadratic weighted kappa toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.29287) has done: 'I increase the training effort for the linear classifier by setting a deterministic seed, extending the number of epochs, and adding a simple learning‑rate scheduler. These changes keep the same model and embedding extraction while giving the classifier more opportunity to learn, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'The changes normalize the embedding vectors before training and inference, which typically improves a linear classifier’s ability to separate classes, and extend the training to 40 epochs to give the classifier more learning time. Both adjustments are small, keep the original architecture, and are expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.47729) has done: 'I add deterministic CUDA settings, replace the single‑layer linear classifier with a tiny two‑layer MLP (1024 → 256 → 5) and train it a bit longer (80 epochs, slower LR decay). These minimal tweaks keep the overall pipeline unchanged while giving the model more capacity and stability, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.55502) has done: 'I augment the training data by adding horizontally‑flipped image embeddings (mirroring the test‑time augmentation) and increase the training epochs from 80 to 120 so the classifier can learn more from the richer dataset. These minimal changes keep the model architecture and overall pipeline unchanged while expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.4816) has done: 'I add class‑balanced loss weights, a dropout layer in the small MLP, slightly lower the learning rate, and extend training to 200 epochs – all minimal tweaks that keep the overall architecture and training pipeline unchanged but should improve the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.31645) has done: 'I improve the model by using the intended GeM pooling (instead of the fixed avg‑pool) in the embedding extraction, give the classifier a larger hidden layer (512 → ReLU → Dropout 0.3 → 5) and train it with a slightly lower learning rate and higher weight decay. These minimal tweaks keep the overall pipeline unchanged while providing richer embeddings and a more expressive classifier, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'I train the linear classifier a bit longer with a smaller learning rate and add a lightweight ensemble that combines its logits with distance‑based similarity to class centroids (computed from the normalized embeddings). This keeps the original model and feature extraction unchanged while giving the predictions a modest boost, moving the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'I slightly increase the classifier’s learning rate and number of epochs so it can learn better from the embeddings, and I give the centroid‑based similarity a stronger influence when the classifier and centroids are ensemble‑combined. These minimal adjustments keep the overall architecture unchanged while aiming to raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os, glob, re, math, sys, types
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
for name in __all__:
    input_sizes[name] = [3, 224, 224]
    means[name] = [0.485, 0.456, 0.406]
    stds[name] = [0.229, 0.224, 0.225]
input_sizes["inceptionv3"] = [3, 299, 299]
means["inceptionv3"] = [0.5, 0.5, 0.5]
stds["inceptionv3"] = [0.5, 0.5, 0.5]

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
    sd = torch.hub.load_state_dict_from_url(settings["url"], progress=False)
    sd = update_state_dict(sd)
    model.load_state_dict(sd)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
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
        return self.logits(x)

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet121(num_classes=1000, pretrained="imagenet"):
    try:
        model = models.densenet121(weights=models.densenet121.Weights.IMAGENET1K_V1)
    except Exception:
        model = models.densenet121(pretrained=False)
        if pretrained is not None:
            s = pretrained_settings["densenet121"][pretrained]
            try:
                model = load_pretrained(model, num_classes, s)
            except Exception:
                pass
    model = modify_densenets(model)
    return model




## === cell 2
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_densenet121_gem(pretrain=True):
    if pretrain:
        try:
            model = densenet121(num_classes=1000, pretrained="imagenet")
        except Exception:
            model = densenet121(num_classes=1000, pretrained=None)
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Identity()
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = glob.glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 4
def extract_embedding(model, tensor):
    """Return the 1024‑dim feature vector before the final linear layer using the model's pooling (GeM)."""
    with torch.no_grad():
        x = model.features(tensor)
        x = F.relu(x, inplace=True)
        x = model.avg_pool(x)  # use GeM pooling defined in the model
        x = x.view(x.size(0), -1)  # flatten to (batch, 1024)
    return x


def make_predictions(model, predictor, img_paths, transform, size=224, device=device):
    """
    Predict using either:
      * a trained classifier (nn.Module) alone,
      * centroids tensor alone,
      * or a tuple (classifier, centroids) for a simple ensemble.
    """
    model.eval()
    preds = []
    use_classifier = isinstance(predictor, nn.Module) or (
        isinstance(predictor, tuple) and isinstance(predictor[0], nn.Module)
    )
    if isinstance(predictor, torch.Tensor):
        centroids_norm = F.normalize(predictor, p=2, dim=1)
    elif isinstance(predictor, tuple):
        _, centroids = predictor
        centroids_norm = F.normalize(centroids, p=2, dim=1)
    else:
        centroids_norm = None

    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)

            tensor = transform(img).unsqueeze(0).to(device)
            embed_orig = extract_embedding(model, tensor)

            img_flipped = ImageOps.mirror(img)
            tensor_fl = transform(img_flipped).unsqueeze(0).to(device)
            embed_flip = extract_embedding(model, tensor_fl)

            embed = (embed_orig + embed_flip) / 2.0  # TTA average

            if isinstance(predictor, tuple):
                classifier_mod, _ = predictor
                embed_norm = F.normalize(embed, p=2, dim=1)
                logits = classifier_mod(embed_norm)  # (1,5)
                dists = torch.norm(centroids_norm - embed_norm, dim=1)  # (5,)
                similarity = -dists
                combined = logits.squeeze(0) + 1.0 * similarity
                pred_class = torch.argmax(combined).item()
            elif use_classifier:
                embed_norm = F.normalize(embed, p=2, dim=1)
                logits = predictor(embed_norm)
                pred_class = torch.argmax(logits, dim=1).item()
            else:
                embed_norm = F.normalize(embed, p=2, dim=1)
                dists = torch.norm(centroids_norm - embed_norm, dim=1)
                pred_class = torch.argmin(dists).item()
        except Exception:
            pred_class = 0  # safe fallback
        preds.append((os.path.splitext(os.path.basename(p))[0], pred_class))
    return preds




## === cell 5
norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = get_densenet121_gem(pretrain=True).to(device)

TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

feat_dim = 1024

emb_list = []
label_list = []

model.eval()
for idx, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMAGE_PATH, f"{row['id_code']}.png")
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        tensor = norm(img).unsqueeze(0).to(device)
        embed = extract_embedding(model, tensor).squeeze(0)  # (1024,)
        emb_list.append(embed.cpu())
        label_list.append(row["diagnosis"])
        img_flipped = ImageOps.mirror(img)
        tensor_fl = norm(img_flipped).unsqueeze(0).to(device)
        embed_fl = extract_embedding(model, tensor_fl).squeeze(0)
        emb_list.append(embed_fl.cpu())
        label_list.append(row["diagnosis"])
    except Exception:
        continue  # skip unreadable images

train_embeddings = torch.stack(emb_list)  # (N,1024)
train_labels = torch.tensor(label_list, dtype=torch.long)

train_embeddings_norm = F.normalize(train_embeddings, p=2, dim=1)

class_counts = torch.bincount(train_labels, minlength=5).float()
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights * (5.0 / class_weights.sum())  # sum to number of classes
class_weights = class_weights.to(device)

classifier = nn.Sequential(
    nn.Linear(feat_dim, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 5),
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

batch_size = 64
num_epochs = 300
train_dataset = torch.utils.data.TensorDataset(train_embeddings_norm, train_labels)
loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

classifier.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = classifier(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    scheduler.step()  # update learning rate

classifier.eval()

centroids = torch.stack(
    [train_embeddings_norm[train_labels == c].mean(0) for c in range(5)]
)




## === cell 6
predictions = make_predictions(
    model, (classifier, centroids), test_images, norm, size=224, device=device
)




## === cell 7
if not predictions:
    predictions = [(os.path.splitext(os.path.basename(p))[0], 0) for p in test_images]




## === cell 8
submission = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv written with", len(submission), "rows")
