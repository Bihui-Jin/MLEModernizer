# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from fastai.vision.all import *
import torch
from torch import nn, tensor
from torchvision.models import inception_v3, resnet50, mobilenet_v2
from torchvision.models import (
    Inception_V3_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import os

torch.backends.cudnn.benchmark = True
if hasattr(torch, "compile"):
    torch._dynamo.config.suppress_errors = True  # keep execution safe




## === cell 1
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels.head()




## === cell 2
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(split.split(labels, labels["breed"]))
labels["is_valid"] = [i in valid_idx for i in range(len(labels))]
labels["id"] = labels["id"].apply(lambda x: f"{x}.jpg")




## === cell 3
path = Path("../input/dog-breed-identification/train")
max_workers = min(8, os.cpu_count() or 1)
dls = ImageDataLoaders.from_df(
    labels,
    path=path,
    label_col="breed",
    valid_col="is_valid",
    fn_col="id",
    item_tfms=Resize(460, method="squeeze"),
    batch_tfms=[*aug_transforms(size=300), Normalize.from_stats(*imagenet_stats)],
    bs=32,
    num_workers=max_workers,
    pin_memory=True,
    persistent_workers=True,
)
dls.show_batch()




## === cell 4
inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True)
inception.aux_logits = False  # disable auxiliary classifier
inception.fc = nn.Identity()
inception.dropout = nn.Identity()
inception = inception.eval()

resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()).eval()

mobile = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mobile = nn.Sequential(
    *list(mobile.children())[:-1],
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
).eval()

for extractor in (inception, resnet, mobile):
    for p in extractor.parameters():
        p.requires_grad = False




## === cell 5
class NeuralNet(nn.Module):
    def __init__(self, extractors, hidden_size, vocab_size, device):
        super().__init__()
        self.device = device
        self.extractors = nn.ModuleList([e.to(device) for e in extractors])
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, vocab_size),
        ).to(device)

    def forward(self, x):
        feats = [conv(x) for conv in self.extractors]
        features = torch.cat(feats, dim=1)
        return self.classifier(features)




## === cell 6
extractors = [inception, resnet, mobile]
hidden_size = 2048 + 2048 + 1280  # sum of output channels
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNet(extractors, hidden_size, len(dls.vocab), device)

if hasattr(torch, "compile"):
    model = torch.compile(model)

weights = [
    labels.shape[0] / (120 * labels["breed"].value_counts()[breed])
    for breed in dls.vocab
]
class_weights = tensor(weights, device=device, dtype=torch.float)

set_seed(42, reproducible=True)
learn = Learner(
    dls,
    model,
    loss_func=nn.CrossEntropyLoss(weight=class_weights),
    metrics=accuracy,
    path=".",
).to_fp16()

suggested_lr = 5e-4
learn.fit_one_cycle(30, lr_max=suggested_lr)

learn.unfreeze()
learn.fit_one_cycle(10, lr_max=slice(1e-5, 1e-4))




## === cell 7
test_files = get_image_files("../input/dog-breed-identification/test")
test_dl = dls.test_dl(test_files, bs=32)

logits, _ = learn.get_preds(dl=test_dl)
probs = torch.nn.functional.softmax(logits, dim=1)

ids = [p.stem for p in test_files]
sub = pd.DataFrame(probs.cpu().numpy(), columns=dls.vocab)
sub.insert(0, "id", ids)
sub.to_csv("submission.csv", index=False)

print("Submission saved to submission.csv")
