# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
BASE = "../input/"
COMP_SUB = "dogs-vs-cats-redux-kernels-edition"
if os.path.isdir(os.path.join(BASE, COMP_SUB)):
    PATH = os.path.join(BASE, COMP_SUB) + "/"
else:
    PATH = BASE

TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz = 224

print("Using PATH:", PATH)
print("PATH contents:", os.listdir(PATH)[:20])



## === cell 2
train_cat_dir = os.path.join(PATH, "train", "cat")
train_dog_dir = os.path.join(PATH, "train", "dog")

if os.path.isdir(train_cat_dir) and os.path.isdir(train_dog_dir):
    cat_files = sorted(
        [
            os.path.join("train", "cat", f)
            for f in os.listdir(train_cat_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    dog_files = sorted(
        [
            os.path.join("train", "dog", f)
            for f in os.listdir(train_dog_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    fnames = np.array(cat_files + dog_files)
    labels = np.array([0] * len(cat_files) + [1] * len(dog_files))
else:
    train_dir = os.path.join(PATH, "train")
    fnames = np.array(
        [
            f"train/{f}"
            for f in sorted(os.listdir(train_dir))
            if os.path.isfile(os.path.join(train_dir, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    labels = np.array([(0 if "cat" in fname else 1) for fname in fnames])

print("Num train images:", len(fnames), "Num labels:", len(labels))



## === cell 3
print(fnames[-2], labels[-2])



## === cell 4
try:
    from fastai.imports import *
    from fastai.transforms import *
    from fastai.conv_learner import *
    from fastai.model import *
    from fastai.dataset import *
    from fastai.sgdr import *
    from fastai.plots import *
except ModuleNotFoundError:
    try:
        from torchvision.models import (
            resnet18,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
        )
    except Exception as e:
        raise ModuleNotFoundError(
            "Required fastai v0.7 modules are unavailable, and torchvision could not be imported "
            "to provide resnet architectures needed by later cells."
        ) from e



## === cell 5
arch = resnet101



## === cell 6
if (
    "ImageClassifierData" not in globals()
    or "tfms_from_model" not in globals()
    or "ConvLearner" not in globals()
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    try:
        import torchvision.transforms as T
    except Exception as e:
        raise ModuleNotFoundError(
            "torchvision is required to provide image transforms in the fastai-compat shim."
        ) from e

    def tfms_from_model(arch, sz):
        return {
            "train": T.Compose(
                [
                    T.Resize((sz, sz)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "valid": T.Compose(
                [
                    T.Resize((sz, sz)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        }

    class _ImageArrayDataset(Dataset):
        def __init__(self, root, fnames, y=None, transform=None):
            self.root = root
            self.fnames = list(fnames)
            self.y = None if y is None else np.array(y, dtype=np.int64)
            self.transform = transform

        def __len__(self):
            return len(self.fnames)

        def __getitem__(self, idx):
            rel_path = self.fnames[idx]
            path = os.path.join(self.root, rel_path)
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if self.y is None:
                return img
            return img, int(self.y[idx])

    class ImageClassifierData:
        def __init__(self, path, trn_dl, val_dl, classes, test_dl=None):
            self.path = path
            self.trn_dl = trn_dl
            self.val_dl = val_dl
            self.classes = classes
            self.test_dl = test_dl

        @classmethod
        def from_names_and_array(cls, path, fnames, y, classes, test_name, tfms, bs=32):
            n = len(fnames)
            idx = np.arange(n)
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            val_sz = max(1, int(0.2 * n))
            val_idx = idx[:val_sz]
            trn_idx = idx[val_sz:]

            trn_fnames = np.array(fnames)[trn_idx]
            trn_y = np.array(y)[trn_idx]
            val_fnames = np.array(fnames)[val_idx]
            val_y = np.array(y)[val_idx]

            trn_ds = _ImageArrayDataset(
                path, trn_fnames, trn_y, transform=tfms["train"]
            )
            val_ds = _ImageArrayDataset(
                path, val_fnames, val_y, transform=tfms["valid"]
            )

            trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=0)
            val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

            test_dl = None
            test_dir = os.path.join(path, test_name)
            if os.path.isdir(test_dir):
                test_files = []
                for root, _, files in os.walk(test_dir):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            full = os.path.join(root, f)
                            rel = os.path.relpath(full, path)
                            test_files.append(rel)
                test_files = sorted(test_files)
                if len(test_files) > 0:
                    test_ds = _ImageArrayDataset(
                        path, np.array(test_files), y=None, transform=tfms["valid"]
                    )
                    test_dl = DataLoader(
                        test_ds, batch_size=bs, shuffle=False, num_workers=0
                    )

            return cls(
                path=path,
                trn_dl=trn_dl,
                val_dl=val_dl,
                classes=classes,
                test_dl=test_dl,
            )

    class ConvLearner:
        def __init__(self, model, data, device):
            self.model = model
            self.data = data
            self.device = device
            self.model.to(self.device)
            self.crit = nn.CrossEntropyLoss()

        @classmethod
        def pretrained(
            cls, arch, data, precompute=True, tmp_name=None, models_name=None
        ):
            model = arch(pretrained=True)
            if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, len(data.classes))
            else:
                if hasattr(model, "classifier") and isinstance(
                    model.classifier, nn.Linear
                ):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, len(data.classes))
                else:
                    raise RuntimeError(
                        "Unsupported architecture for minimal ConvLearner shim."
                    )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return cls(model=model, data=data, device=device)

        def fit(self, lr, epochs):
            self.model.train()
            opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            for _ in range(int(epochs)):
                for xb, yb in self.data.trn_dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    out = self.model(xb)
                    loss = self.crit(out, yb)
                    loss.backward()
                    opt.step()
            return None


data = ImageClassifierData.from_names_and_array(
    path=PATH,
    fnames=fnames,
    y=labels,
    classes=[
        "cat",
        "dog",
    ],  # explicit mapping: 0=cat, 1=dog; aligns with submission "prob dog"
    test_name="test",
    tfms=tfms_from_model(arch, sz),
    bs=32,
)
learn = ConvLearner.pretrained(
    arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH
)



## === cell 7
learn.fit(lr=0.01, epochs=1)



## === cell 8
train_dir_flat = f"{PATH}train"
if os.path.isdir(train_dir_flat):
    _fnames_flat = np.array(
        [
            f"train/{f}"
            for f in sorted(os.listdir(train_dir_flat))
            if os.path.isfile(os.path.join(train_dir_flat, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    _labels_flat = np.array([(0 if "cat" in fname else 1) for fname in _fnames_flat])
else:
    _fnames_flat, _labels_flat = fnames, labels



## === cell 10
if not hasattr(learn, "predict"):
    import numpy as np
    import torch

    def _predict(self, is_test=False):
        self.model.eval()
        dl = (
            getattr(self.data, "test_dl", None)
            if is_test
            else getattr(self.data, "val_dl", None)
        )
        if dl is None:
            raise AttributeError(
                "No dataloader available for prediction (test_dl/val_dl is None)."
            )

        outs = []
        with torch.no_grad():
            for batch in dl:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(self.device)
                logits = self.model(xb)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                outs.append(log_probs.detach().cpu().numpy())
        return (
            np.concatenate(outs, axis=0)
            if len(outs)
            else np.zeros((0, len(self.data.classes)), dtype=np.float32)
        )

    learn.predict = _predict.__get__(learn, learn.__class__)

log_preds = learn.predict(is_test=True)
probs = np.exp(log_preds[:, 1])



## === cell 11
test_fnames = None
if hasattr(learn, "data") and getattr(learn.data, "test_dl", None) is not None:
    ds = getattr(learn.data.test_dl, "dataset", None)
    if ds is not None and hasattr(ds, "fnames"):
        test_fnames = list(ds.fnames)

if test_fnames is None:
    test_dir = os.path.join(PATH, "test")
    test_files = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                test_files.append(os.path.join(root, f))
    test_files = sorted(test_files)
    test_fnames = [os.path.relpath(p, PATH) for p in test_files]

ids = [os.path.splitext(os.path.basename(str(f)))[0] for f in test_fnames]

if len(ids) != len(probs):
    raise ValueError(f"ids/probs length mismatch: {len(ids)} ids vs {len(probs)} probs")

ans = pd.DataFrame({"id": ids, "label": probs})
ans["id"] = ans["id"].astype(int)
ans = ans.sort_values("id").reset_index(drop=True)
ans.head()



## === cell 12
ans.describe()



## === cell 13
ans.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", ans.shape)
print(ans.head())
