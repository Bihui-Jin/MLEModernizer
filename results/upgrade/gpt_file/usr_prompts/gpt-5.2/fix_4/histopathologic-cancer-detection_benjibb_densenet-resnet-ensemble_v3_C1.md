# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, gc, random, json, time
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from fastai.vision.all import *
from fastai.tabular.all import *

SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

set_seed(SEED, reproducible=True)




## === cell 1
DATA_ROOT = Path("../input/histopathologic-cancer-detection")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABELS_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

assert (
    TRAIN_DIR.exists() and TEST_DIR.exists()
), f"Missing train/test dirs under {DATA_ROOT}"
assert LABELS_CSV.exists() and SAMPLE_SUB.exists(), "Missing required CSVs"

labels_df = pd.read_csv(LABELS_CSV)
labels_df["id"] = labels_df["id"].astype(str)
labels_df["label"] = labels_df["label"].astype(int)

label_map = labels_df.set_index("id")["label"].to_dict()

sample_df = pd.read_csv(SAMPLE_SUB)
sample_df["id"] = sample_df["id"].astype(str)

print(labels_df.head())
print("Train images (from CSV):", len(labels_df))
print("Test images  (from CSV):", len(sample_df))




## === cell 2
train_files = [TRAIN_DIR / f"{i}.tif" for i in labels_df["id"].values]
train_files = L(train_files)

idx = np.arange(len(train_files))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
valid_pct = 0.2
n_valid = int(len(idx) * valid_pct)
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

train_items = train_files[train_idx]
valid_items = train_files[valid_idx]

len(train_items), len(valid_items)




## === cell 3
def _cache_paths(name: str):
    cache_dir = Path("./logit_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{name}_train_logits.npy",
        cache_dir / f"{name}_test_logits.npy",
        cache_dir / f"{name}_meta.json",
    )


def _save_logits(name, p_all, p_test, test_ids):
    tr_path, te_path, meta_path = _cache_paths(name)
    np.save(tr_path, p_all)
    np.save(te_path, p_test)
    meta = {
        "name": name,
        "seed": SEED,
        "n_train": int(p_all.shape[0]),
        "n_test": int(p_test.shape[0]),
        "test_ids_first10": test_ids[:10],
        "test_ids_last10": test_ids[-10:],
        "created_utc": time.time(),
    }
    meta_path.write_text(json.dumps(meta))
    return tr_path, te_path


def _load_logits(name, expected_n_train, expected_test_ids):
    tr_path, te_path, meta_path = _cache_paths(name)
    if not (tr_path.exists() and te_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("seed") != SEED:
        return None
    p_all = np.load(tr_path, mmap_mode="r")
    p_test = np.load(te_path, mmap_mode="r")
    if p_all.shape[0] != expected_n_train:
        return None
    if p_test.shape[0] != len(expected_test_ids):
        return None
    if meta.get("test_ids_first10") != expected_test_ids[:10]:
        return None
    if meta.get("test_ids_last10") != expected_test_ids[-10:]:
        return None
    return np.array(p_all), np.array(p_test)


def get_test_logits(learn, test_files, bs=256):
    test_dl = learn.dls.test_dl(test_files, with_labels=False, bs=bs, shuffle=False)
    p_test, _ = learn.get_preds(dl=test_dl, act=None)
    return p_test.cpu().numpy()


vision_cfg = [
    ("dense161", densenet161),
    ("dense201", densenet201),
    ("res50", resnet50),
]

vision_logits = {}
test_files_ref = None

test_ids = sample_df["id"].tolist()
test_files = [TEST_DIR / f"{i}.tif" for i in test_ids]
test_files = L(test_files)
test_files_ref = test_files

for name, arch in vision_cfg:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cached = _load_logits(
        name, expected_n_train=len(train_files), expected_test_ids=test_ids
    )
    if cached is not None:
        p_all, p_test = cached
        vision_logits[name] = {"train": p_all, "test": p_test}
        continue

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=lambda o: o,
        get_y=lambda o: label_map[o.stem],
        splitter=IndexSplitter(list(valid_idx)),
        item_tfms=Resize(96),
        batch_tfms=[
            *aug_transforms(size=96, min_scale=0.9),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    dls = dblock.dataloaders(
        train_files,
        bs=128,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    learn = vision_learner(dls, arch, metrics=[RocAucBinary()], pretrained=True)
    learn.fit_one_cycle(1, 3e-3)

    all_dl = dls.test_dl(train_files, with_labels=True, bs=256, shuffle=False)
    p_all, t_all = learn.get_preds(dl=all_dl, act=None)
    p_all = p_all.cpu().numpy()

    if p_all.ndim == 1:
        p_all = p_all.reshape(-1, 1)
    if p_all.shape[1] == 1:
        p_all = np.concatenate(
            [np.zeros((p_all.shape[0], 1), dtype=p_all.dtype), p_all],
            axis=1,
        )

    p_test = get_test_logits(learn, test_files, bs=256)
    if p_test.ndim == 1:
        p_test = p_test.reshape(-1, 1)
    if p_test.shape[1] == 1:
        p_test = np.concatenate(
            [np.zeros((p_test.shape[0], 1), dtype=p_test.dtype), p_test],
            axis=1,
        )

    vision_logits[name] = {"train": p_all, "test": p_test}
    _save_logits(name, p_all, p_test, test_ids)

{k: (v["train"].shape, v["test"].shape) for k, v in vision_logits.items()}




## === cell 4
train = pd.DataFrame(
    {
        "dense161_0": vision_logits["dense161"]["train"][:, 0],
        "dense161_1": vision_logits["dense161"]["train"][:, 1],
        "dense201_0": vision_logits["dense201"]["train"][:, 0],
        "dense201_1": vision_logits["dense201"]["train"][:, 1],
        "res50_0": vision_logits["res50"]["train"][:, 0],
        "res50_1": vision_logits["res50"]["train"][:, 1],
        "y": labels_df["label"].values.astype(np.int64),
    }
)

test = pd.DataFrame(
    {
        "dense161_0": vision_logits["dense161"]["test"][:, 0],
        "dense161_1": vision_logits["dense161"]["test"][:, 1],
        "dense201_0": vision_logits["dense201"]["test"][:, 0],
        "dense201_1": vision_logits["dense201"]["test"][:, 1],
        "res50_0": vision_logits["res50"]["test"][:, 0],
        "res50_1": vision_logits["res50"]["test"][:, 1],
    }
)
test["y"] = 0

train.head(), test.head()




## === cell 5
dep_var = "y"
cont_names = [
    "dense161_0",
    "dense161_1",
    "dense201_0",
    "dense201_1",
    "res50_0",
    "res50_1",
]

splits = RandomSplitter(valid_pct=0.2, seed=SEED)(range_of(train))
to = TabularPandas(
    train,
    procs=[Normalize],
    cont_names=cont_names,
    y_names=dep_var,
    splits=splits,
    y_block=CategoryBlock,
)

dls = to.dataloaders(
    bs=512,
    num_workers=min(4, os.cpu_count() or 2),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)
test_dl = dls.test_dl(test)




## === cell 6
def roc_auc_prob(inp, targ):
    if inp.ndim == 1 or inp.shape[1] == 1:
        prob1 = torch.sigmoid(inp.view(-1))
    else:
        prob1 = torch.softmax(inp, dim=1)[:, 1]
    targ = targ.view(-1).long()
    return roc_auc_score(targ.cpu().numpy(), prob1.detach().cpu().numpy())


roc_auc_metric = AccumMetric(roc_auc_prob, flatten=False)




## === cell 7
learn = tabular_learner(
    dls, layers=[10, 10, 10], metrics=[accuracy, roc_auc_metric], ps=0.5, wd=1e-1
)




## === cell 8
cbs = [
    SaveModelCallback(monitor="roc_auc_prob", fname="best", with_opt=True),
]
learn.fit_one_cycle(20, 1e-3, cbs=cbs)




## === cell 9
learn.load("best")




## === cell 10
preds, _ = learn.get_preds(dl=test_dl)
preds = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()

val_res = learn.validate()
auc_val = float(val_res[2])
auc_val




## === cell 11
sub = pd.read_csv(SAMPLE_SUB)
sub["id"] = sub["id"].astype(str)

test_ids = [p.stem for p in test_files_ref]
pred_map = pd.DataFrame({"id": test_ids, "label": preds})

sub = sub[["id"]].merge(pred_map, on="id", how="left")
assert (
    sub["label"].notna().all()
), "Some test ids did not receive predictions; ordering/matching issue."

sub.head()




## === cell 12
out_name = "submission.csv"
sub.to_csv(out_name, index=False)
print("Wrote:", out_name, "rows:", len(sub))
print("Validation AUC (internal stacker split):", auc_val)
print(sub.describe())
