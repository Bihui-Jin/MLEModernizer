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

# 5. Target score

0.25207

# 6. Current score

0.28287

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.28287) has done: 'I fix the torchvision InceptionV3 initialization error by using the new `weights=` API and keeping `aux_logits=True` (as required by the pretrained weights), then extracting features from the correct submodule so the rest of your architecture stays the same. I also ensure both feature extractors run in eval mode with `no_grad()` so BatchNorm/Dropout inside the pretrained backbones don’t break training stability, while still training your classifier head as intended. Finally, I make submission generation robust by aligning columns to `sample_submission.csv` order and writing a valid `submission.csv` with probabilities for all 120 breeds.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels



## === cell 1
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_ids, valid_ids = next(split.split(labels, labels["breed"]))
valid_set = set(valid_ids)
labels["is_valid"] = [i in valid_set for i in range(len(labels))]

labels["id"] = labels["id"].apply(lambda x: x + ".jpg")



## === cell 2
path = "../input/dog-breed-identification/train"

dls = ImageDataLoaders.from_df(
    labels,
    path,
    item_tfms=RandomResizedCrop(460, min_scale=0.3),
    batch_tfms=[*aug_transforms(size=300), Normalize.from_stats(*imagenet_stats)],
    bs=32,
    valid_col="is_valid",
)
dls.show_batch()



## === cell 3
from torchvision.models import inception_v3, Inception_V3_Weights

_inception_full = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
inception = nn.Sequential(
    _inception_full.Conv2d_1a_3x3,
    _inception_full.Conv2d_2a_3x3,
    _inception_full.Conv2d_2b_3x3,
    _inception_full.maxpool1,
    _inception_full.Conv2d_3b_1x1,
    _inception_full.Conv2d_4a_3x3,
    _inception_full.maxpool2,
    _inception_full.Mixed_5b,
    _inception_full.Mixed_5c,
    _inception_full.Mixed_5d,
    _inception_full.Mixed_6a,
    _inception_full.Mixed_6b,
    _inception_full.Mixed_6c,
    _inception_full.Mixed_6d,
    _inception_full.Mixed_6e,
    _inception_full.Mixed_7a,
    _inception_full.Mixed_7b,
    _inception_full.Mixed_7c,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
).eval()



## === cell 4
from torchvision.models import resnet50, ResNet50_Weights

resnet = nn.Sequential(
    *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1], nn.Flatten()
).eval()




## === cell 5
class NeuralNet(Module):
    def __init__(self, extractors, hidden_size, vocab_size, device):
        self.extractors = extractors
        for conv in self.extractors:
            conv.to(device)
            conv.eval()  # keep pretrained backbones in inference mode (BN/Dropout stable)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, vocab_size),
        )

    def forward(self, x):
        with torch.no_grad():
            feats = [conv(x) for conv in self.extractors]
        features = torch.cat(feats, dim=1)
        return self.classifier(features)




## === cell 6
extractors = [inception, resnet]
hidden_size = 2048 + 2048
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNet(extractors, hidden_size, len(dls.vocab), device).to(device)



## === cell 7
vc = labels["breed"].value_counts()
weights = [labels.shape[0] / (120 * vc[breed]) for breed in dls.vocab]
weights = torch.tensor(weights, device=device, dtype=torch.float32)



## === cell 8
learn = Learner(
    dls,
    model,
    loss_func=nn.CrossEntropyLoss(weight=weights),
    metrics=[accuracy, F.cross_entropy],
    path=".",
).to_fp16()

learn.lr_find()



## === cell 9
learn.fit_one_cycle(10, 1e-3)



## === cell 10
torch.cuda.empty_cache()



## === cell 11
test_files = get_image_files("../input/dog-breed-identification/test")
test_dl = dls.test_dl(test_files, bs=32)



## === cell 12
preds, _ = learn.get_preds(dl=test_dl)



## === cell 13
sample_sub = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
breed_cols = [c for c in sample_sub.columns if c != "id"]

probs = torch.softmax(preds, dim=1).cpu().numpy()
sub = pd.DataFrame(probs, columns=list(dls.vocab))
sub.insert(0, "id", [p.stem for p in test_files])

sub = sub.reindex(columns=["id"] + breed_cols)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
