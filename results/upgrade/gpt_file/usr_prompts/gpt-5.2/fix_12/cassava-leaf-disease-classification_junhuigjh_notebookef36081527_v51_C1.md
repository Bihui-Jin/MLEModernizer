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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8566032033847084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59492) has done: 'I fix the runtime failures by removing hard dependencies on missing Kaggle input models/files and instead training the same DecisionTreeClassifier on probabilities produced by a lightweight, locally available image model. To keep the core approach intact (probabilities → concatenate temps → decision tree), the pipeline still generate per-image softmax probabilities at two temperatures, concatenate them, and fit/predict with a decision tree. I also make all paths robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts, ensure deterministic execution, and guarantee a correctly formatted `submission.csv` is written. This should run end-to-end in the provided environment and yield a reasonable accuracy score (likely below top deep models but valid and improved over “no submission”).'
- What this solution (achieved 0.60538) has done: 'Your current score is far below the target, and the biggest limiter is that the pipeline uses an ImageNet classifier’s 1000-way logits randomly projected to 5 classes, which has weak alignment to cassava labels. To keep your core approach intact (image → two-temperature probabilities → concatenate → DecisionTreeClassifier), I switch the feature extractor to a pretrained ImageNet model with much stronger representations (ResNet50) and extract its penultimate embedding, then map to 5-class probabilities with a fixed random projection at two temperatures just like you do now. This is a minimal, safe change that typically yields a large accuracy gain without changing the downstream decision tree logic or submission semantics. I also make the model’s preprocessing match the chosen weights to avoid silent normalization mismatch, which improves score stability.'
- What this solution (achieved 0.76196) has done: 'We keep your core pipeline intact (ResNet50 embedding → fixed random projection → two-temperature 5-way probabilities → concatenate → DecisionTreeClassifier) and make only changes that improve label alignment while staying deterministic. The biggest win with minimal risk is to replace the random projection with a learned linear map from embeddings to 5 classes (multinomial logistic regression) trained on the same training labels; we still compute two-temperature probabilities from that 5-logit output exactly like before and still fit/predict with the decision tree. This tends to move accuracy materially upward toward your target without changing the downstream tree logic or submission semantics. We also switch feature extraction to efficient batched inference (same model/transform) so it finishes within time limits.'
- What this solution (achieved 0.77055) has done: 'Your current score (0.76196) is below the target (0.8566), so we should improve accuracy with minimal risk while keeping the same core pipeline (ResNet50 embeddings → logistic-regression 5-logits → two-temperature probabilities → concatenate → decision tree). The biggest low-change gain is to add the standard ImageNet test-time augmentation used for pretrained models (horizontal flip) and average the embeddings from original+flipped images; this preserves the exact model and training approach but typically improves robustness. I’m also switching the final classifier from `DecisionTreeClassifier` to `ExtraTreesClassifier` (still a tree-based classifier with the same fit/predict semantics on the same 10-dim probability features) which usually generalizes better with minimal tuning and is fast enough. Everything else (paths, submission formatting, temperatures, LR mapping, and deterministic seeds) stays the same.'
- What this solution (achieved 0.68685) has done: 'Your current score (0.77055) is well below the target (0.8566), so we should improve accuracy with the smallest changes that keep your exact pipeline structure (ResNet50 embeddings → logistic-regression 5-logits → two-temperature probabilities → concatenate → tree classifier). The main low-risk gain is to fit the logistic regression on standardized embeddings (a pure linear reparameterization that usually improves alignment/calibration) and to enable class balancing to counter cassava’s label imbalance. To keep core semantics intact, we still use the same ResNet50, the same two-temperature softmax features, and the same ExtraTrees downstream; we only adjust the LR mapping quality and slightly strengthen ExtraTrees capacity without changing the approach. All paths, determinism, and submission formatting remain the same.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import ImageFile

import torch
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.cuda.manual_seed_all(RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 1
def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


weights = models.ResNet50_Weights.DEFAULT
torch_transforms = weights.transforms()

model2 = models.resnet50(weights=weights).to(device)
model2.eval()

feature_extractor = torch.nn.Sequential(*list(model2.children())[:-1]).to(device)
feature_extractor.eval()

if device.type == "cuda":
    feature_extractor = feature_extractor.to(memory_format=torch.channels_last)
try:
    if hasattr(torch, "compile"):
        feature_extractor = torch.compile(feature_extractor, mode="reduce-overhead")
except Exception:
    pass

base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def first_existing_file(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.exists(p):
            return p
    return None


def first_existing_dir(relpath):
    for b in base_candidates:
        p = os.path.join(b, relpath)
        if os.path.isdir(p):
            return p
    return None


train_csv_path = first_existing_file("train.csv")
if train_csv_path is None:
    for p in ["/kaggle/input/train.csv", "/kaggle/data/train.csv"]:
        if os.path.exists(p):
            train_csv_path = p
            break
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in expected locations.")

train_dir = first_existing_dir("train_images")
if train_dir is None:
    for p in ["/kaggle/input/train_images", "/kaggle/data/train_images"]:
        if os.path.isdir(p):
            train_dir = p
            break
if train_dir is None:
    raise FileNotFoundError("train_images directory not found in expected locations.")

train_df = pd.read_csv(train_csv_path)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv must have columns image_id,label. Got: {train_df.columns.tolist()}"
    )

train_df["filepath"] = train_dir.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)
if len(train_df) == 0:
    raise FileNotFoundError(
        "No training images found matching train.csv in train_images."
    )

train_labels = train_df["label"].astype(int).values

T_model2 = 1.0
T_model1_fallback = 1.6

BATCH_SIZE = 64 if torch.cuda.is_available() else 16




## === cell 2
def extract_embeddings_batch(filepaths, batch_size=BATCH_SIZE, use_tta_flip=True):
    from torch.utils.data import Dataset, DataLoader

    try:
        from torchvision.io import read_image  # uint8 CHW RGB

        _HAS_TV_READ = True
    except Exception:
        _HAS_TV_READ = False
        read_image = None

    class _PathDataset(Dataset):
        __slots__ = ("fps",)

        def __init__(self, fps):
            self.fps = fps

        def __len__(self):
            return len(self.fps)

        def __getitem__(self, idx):
            fp = self.fps[idx]
            if _HAS_TV_READ:
                img = read_image(fp)  # uint8 CHW
            else:
                from PIL import Image

                img = Image.open(fp).convert("RGB")
                img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()
            return img

    ds = _PathDataset(list(filepaths))

    if torch.cuda.is_available():
        num_workers = min(8, (os.cpu_count() or 8))
        prefetch_factor = 4
    else:
        num_workers = min(4, (os.cpu_count() or 4))
        prefetch_factor = 2

    def _collate(batch_imgs):
        xs = [torch_transforms(im) for im in batch_imgs]
        x = torch.stack(xs, dim=0)
        if use_tta_flip:
            xs2 = [torch_transforms(torch.flip(im, dims=[2])) for im in batch_imgs]
            x2 = torch.stack(xs2, dim=0)
            return x, x2
        return x

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        collate_fn=_collate,
    )

    embs = np.empty((len(ds), 2048), dtype=np.float32)
    write_pos = 0

    with torch.inference_mode():
        for batch in loader:
            if use_tta_flip:
                x, x2 = batch
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    x2 = x2.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    x = x.to(device)
                    x2 = x2.to(device)

                emb = feature_extractor(x)
                emb2 = feature_extractor(x2)
                emb = 0.5 * (emb + emb2)
            else:
                x = batch
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    x = x.to(device)
                emb = feature_extractor(x)

            emb = (
                emb.squeeze(-1)
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            bsz = emb.shape[0]
            embs[write_pos : write_pos + bsz] = emb
            write_pos += bsz

            if write_pos % 500 == 0 or write_pos == len(ds):
                print(f"Embeddings: {write_pos}/{len(ds)}", end="\r")

    return embs


train_embs = extract_embeddings_batch(
    train_df["filepath"].values, batch_size=BATCH_SIZE, use_tta_flip=True
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

oof_logits5 = np.zeros((len(train_embs), 5), dtype=np.float32)


def make_lr():
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=1200,
        class_weight="balanced",
        random_state=RNG_SEED,
    )


for fold, (tr_idx, va_idx) in enumerate(skf.split(train_embs, train_labels), start=1):
    scaler_f = StandardScaler(with_mean=True, with_std=True)
    Xtr = np.asarray(
        scaler_f.fit_transform(train_embs[tr_idx]), dtype=np.float32, order="C"
    )
    Xva = np.asarray(
        scaler_f.transform(train_embs[va_idx]), dtype=np.float32, order="C"
    )

    lr_f = make_lr()
    lr_f.fit(Xtr, train_labels[tr_idx])

    oof_logits5[va_idx] = lr_f.decision_function(Xva).astype(np.float32, copy=False)
    print(f"OOF fold {fold}/5 done", end="\r")

p2_train = softmax_np(oof_logits5 / float(T_model2), axis=1).astype(
    np.float32, copy=False
)
p1_train = softmax_np(oof_logits5 / float(T_model1_fallback), axis=1).astype(
    np.float32, copy=False
)
train_feats = np.concatenate([p1_train, p2_train], axis=1).astype(
    np.float32, copy=False
)  # (N,10)

tree_model = ExtraTreesClassifier(
    n_estimators=900,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=2,
    bootstrap=True,
    max_samples=0.8,
    class_weight="balanced_subsample",
    random_state=RNG_SEED,
    n_jobs=-1,
)
tree_model.fit(train_feats, train_labels)

scaler = StandardScaler(with_mean=True, with_std=True)
train_embs_s = np.asarray(scaler.fit_transform(train_embs), dtype=np.float32, order="C")

lr_map = make_lr()
lr_map.fit(train_embs_s, train_labels)

(train_feats.shape, np.unique(train_labels, return_counts=True)[0])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/902129148.py in <cell line: 0>()
    115 
    116 
--> 117 train_embs = extract_embeddings_batch(
    118     train_df["filepath"].values, batch_size=BATCH_SIZE, use_tta_flip=True
    119 )

/tmp/ipykernel_55/902129148.py in extract_embeddings_batch(filepaths, batch_size, use_tta_flip)
     85                 emb = feature_extractor(x)
     86                 emb2 = feature_extractor(x2)
---> 87                 emb = 0.5 * (emb + emb2)
     88             else:
     89                 x = batch

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/external_utils.py", line 45, in inner
    return fn(*args, **kwargs). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 3
test_dir = first_existing_dir("test_images")
if test_dir is None:
    for p in ["/kaggle/input/test_images", "/kaggle/data/test_images"]:
        if os.path.isdir(p):
            test_dir = p
            break
if test_dir is None:
    raise FileNotFoundError("Test images directory not found in expected locations.")

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_filepaths = [os.path.join(test_dir, f) for f in test_images]

test_embs = extract_embeddings_batch(
    test_filepaths, batch_size=BATCH_SIZE, use_tta_flip=True
)

test_embs_s = np.asarray(scaler.transform(test_embs), dtype=np.float32, order="C")

test_logits5 = lr_map.decision_function(test_embs_s).astype(
    np.float32, copy=False
)  # (M,5)
p2_test = softmax_np(test_logits5 / float(T_model2), axis=1).astype(
    np.float32, copy=False
)
p1_test = softmax_np(test_logits5 / float(T_model1_fallback), axis=1).astype(
    np.float32, copy=False
)
combined_probs = np.concatenate([p1_test, p2_test], axis=1).astype(
    np.float32, copy=False
)  # (M,10)

combined_probs.shape




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/750080268.py in <cell line: 0>()
     11 test_filepaths = [os.path.join(test_dir, f) for f in test_images]
     12 
---> 13 test_embs = extract_embeddings_batch(
     14     test_filepaths, batch_size=BATCH_SIZE, use_tta_flip=True
     15 )

/tmp/ipykernel_55/902129148.py in extract_embeddings_batch(filepaths, batch_size, use_tta_flip)
     85                 emb = feature_extractor(x)
     86                 emb2 = feature_extractor(x2)
---> 87                 emb = 0.5 * (emb + emb2)
     88             else:
     89                 x = batch

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torch/_dynamo/external_utils.py", line 45, in inner
    return fn(*args, **kwargs). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 4
prediction = tree_model.predict(combined_probs)
prediction[:10], prediction.shape




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1247617739.py in <cell line: 0>()
----> 1 prediction = tree_model.predict(combined_probs)
      2 prediction[:10], prediction.shape
      3 
      4 

NameError: name 'tree_model' is not defined

## === cell 5
sample_path = first_existing_file("sample_submission.csv")
if sample_path is None:
    for p in [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]:
        if os.path.exists(p):
            sample_path = p
            break
if sample_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

sample_sub = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": test_images, "label": prediction.astype(int)})
submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    mode_label = int(pd.Series(train_labels).mode().iloc[0])
    submission["label"] = submission["label"].fillna(mode_label).astype(int)

submission["label"] = submission["label"].clip(0, 4).astype(int)

submission.to_csv("submission.csv", index=False)
submission.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1361931950.py in <cell line: 0>()
     13 sample_sub = pd.read_csv(sample_path)
     14 
---> 15 pred_df = pd.DataFrame({"image_id": test_images, "label": prediction.astype(int)})
     16 submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
     17 

NameError: name 'prediction' is not defined

## === cell 6
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["label"].between(0, 4).all()
print("Wrote submission.csv with shape:", submission.shape)
submission.tail()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/3927714983.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 assert list(submission.columns) == ["image_id", "label"]
      3 assert len(submission) == len(sample_sub)
      4 assert submission["label"].between(0, 4).all()
      5 print("Wrote submission.csv with shape:", submission.shape)

AssertionError:
