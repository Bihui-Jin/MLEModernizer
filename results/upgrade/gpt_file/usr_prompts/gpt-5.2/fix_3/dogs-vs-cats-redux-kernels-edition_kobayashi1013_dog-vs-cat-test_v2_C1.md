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

3.13

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

# 5. Target score

0.6108271339930265

# 6. Current score

0.68986

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.68852) has done: 'I fix the TensorFlow import crash by avoiding `tensorflow` entirely (it’s incompatible in this environment, causing the protobuf `MessageFactory` error) and replace the training/inference with a tiny pure-Python baseline that still produces valid dog probabilities. I also fix the dataset path logic so it reads the provided `train/cat`, `train/dog`, and `test/unknown` folders directly instead of extracting ZIPs to non-existent paths. Finally, I ensure the submission has exactly the required `id,label` columns, with numeric `id` sorted ascending and `label` clipped away from 0/1 for logloss safety, writing `submission.csv` to the working directory.'
- What this solution (achieved 0.68986) has done: 'Your current score (0.68852, lower is better) is worse than the target (0.61083), so we should improve the predictions slightly without changing the core “single scalar feature + Gaussian NB” logic. The biggest low-risk gain here is to avoid distribution shift between train and test by fitting a simple 1D calibration (Platt scaling / logistic regression) on top of your existing NB logit using a deterministic train/validation split; this keeps the same feature extractor and generative model, only calibrating the probability mapping to reduce log loss. I also ensure the submission aligns to the sample_submission ids (including filling any missing ids) and clip probabilities more conservatively for logloss safety. All changes are small, deterministic, and should move logloss down toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import math
import numpy as np
import pandas as pd

from PIL import Image

np.random.seed(42)



## === cell 1
IMG_SIZE = 20

DATA_ROOT = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
TRAIN_CAT_DIR = os.path.join(DATA_ROOT, "train", "cat")
TRAIN_DOG_DIR = os.path.join(DATA_ROOT, "train", "dog")
TEST_DIR = os.path.join(DATA_ROOT, "test", "unknown")

SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_CAT_DIR), f"Missing train cat dir: {TRAIN_CAT_DIR}"
assert os.path.isdir(TRAIN_DOG_DIR), f"Missing train dog dir: {TRAIN_DOG_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"




## === cell 2
def load_image_feature(path, img_size=IMG_SIZE):
    """
    Minimal feature extractor: average grayscale intensity in resized image.
    Pure-Python (PIL + numpy), fast, deterministic.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        im = im.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return float(arr.mean())


def list_images(dir_path):
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(exts)
    ]


cat_paths = list_images(TRAIN_CAT_DIR)
dog_paths = list_images(TRAIN_DOG_DIR)

assert len(cat_paths) > 0 and len(dog_paths) > 0, "Training folders are empty."

cat_paths = np.array(sorted(cat_paths))
dog_paths = np.array(sorted(dog_paths))

rng = np.random.RandomState(42)
rng.shuffle(cat_paths)
rng.shuffle(dog_paths)

val_frac = 0.2
n_cat_val = int(len(cat_paths) * val_frac)
n_dog_val = int(len(dog_paths) * val_frac)

cat_val_paths = cat_paths[:n_cat_val].tolist()
cat_tr_paths = cat_paths[n_cat_val:].tolist()
dog_val_paths = dog_paths[:n_dog_val].tolist()
dog_tr_paths = dog_paths[n_dog_val:].tolist()

cat_tr_feats = np.array([load_image_feature(p) for p in cat_tr_paths], dtype=np.float32)
dog_tr_feats = np.array([load_image_feature(p) for p in dog_tr_paths], dtype=np.float32)

mu_cat = float(cat_tr_feats.mean())
mu_dog = float(dog_tr_feats.mean())
var_cat = float(cat_tr_feats.var() + 1e-8)
var_dog = float(dog_tr_feats.var() + 1e-8)

prior_dog = float(len(dog_tr_paths) / (len(dog_tr_paths) + len(cat_tr_paths)))

(mu_cat, mu_dog, var_cat, var_dog, prior_dog)




## === cell 3
def gaussian_logpdf(x, mu, var):
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mu) ** 2) / var)


def nb_logit_from_feature(x):
    logp_dog = math.log(prior_dog + 1e-12) + gaussian_logpdf(x, mu_dog, var_dog)
    logp_cat = math.log(1.0 - prior_dog + 1e-12) + gaussian_logpdf(x, mu_cat, var_cat)
    return logp_dog - logp_cat


def sigmoid(z):
    if z > 50:
        return 1.0
    if z < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_proba_dog_from_feature(x):
    return sigmoid(nb_logit_from_feature(x))


p_at_cat_mean = predict_proba_dog_from_feature(mu_cat)
p_at_dog_mean = predict_proba_dog_from_feature(mu_dog)
(p_at_cat_mean, p_at_dog_mean)




## === cell 4
def fit_platt_scaling(logits, y, l2=1e-2, iters=200, lr=0.1):
    """
    Fit p = sigmoid(a * logit + b) by minimizing log loss + L2(a,b).
    Pure numpy, deterministic, small overhead.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    a = 1.0
    b = 0.0

    for _ in range(iters):
        z = a * logits + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

        g = p - y

        da = (g * logits).mean() + l2 * a
        db = g.mean() + l2 * b

        a -= lr * da
        b -= lr * db

    return float(a), float(b)


val_paths = cat_val_paths + dog_val_paths
val_y = np.array([0] * len(cat_val_paths) + [1] * len(dog_val_paths), dtype=np.float32)

val_logits = []
for p in val_paths:
    feat = load_image_feature(p)
    val_logits.append(nb_logit_from_feature(feat))
val_logits = np.array(val_logits, dtype=np.float64)

a_cal, b_cal = fit_platt_scaling(val_logits, val_y, l2=1e-2, iters=300, lr=0.2)


def predict_proba_dog_calibrated_from_feature(x):
    return sigmoid(a_cal * nb_logit_from_feature(x) + b_cal)


(
    predict_proba_dog_calibrated_from_feature(mu_cat),
    predict_proba_dog_calibrated_from_feature(mu_dog),
    a_cal,
    b_cal,
)




## === cell 5
def extract_id_from_filename(filename):
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
assert len(test_files) > 0, "No test images found."

test_ids = []
test_paths = []
for f in test_files:
    i = extract_id_from_filename(f)
    if i is None:
        continue
    test_ids.append(i)
    test_paths.append(os.path.join(TEST_DIR, f))

order = np.argsort(test_ids)
test_ids = [test_ids[i] for i in order]
test_paths = [test_paths[i] for i in order]

probs = []
for p in test_paths:
    feat = load_image_feature(p)
    probs.append(predict_proba_dog_calibrated_from_feature(feat))

probs = np.array(probs, dtype=np.float32)

probs = np.clip(probs, 1e-5, 1.0 - 1e-5)

sub = pd.DataFrame({"id": test_ids, "label": probs})
sub = sub.sort_values("id").reset_index(drop=True)
sub["id"] = sub["id"].astype(int)

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["id"].astype(int).values

sub = sub.set_index("id").reindex(sample_ids).reset_index()
sub["label"] = sub["label"].astype(np.float32).fillna(0.5).clip(1e-5, 1.0 - 1e-5)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
