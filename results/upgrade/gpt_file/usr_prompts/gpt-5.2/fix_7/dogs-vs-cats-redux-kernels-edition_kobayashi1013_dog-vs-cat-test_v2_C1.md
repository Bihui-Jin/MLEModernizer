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

0.68791

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.68852) has done: 'I fix the TensorFlow import crash by avoiding `tensorflow` entirely (it’s incompatible in this environment, causing the protobuf `MessageFactory` error) and replace the training/inference with a tiny pure-Python baseline that still produces valid dog probabilities. I also fix the dataset path logic so it reads the provided `train/cat`, `train/dog`, and `test/unknown` folders directly instead of extracting ZIPs to non-existent paths. Finally, I ensure the submission has exactly the required `id,label` columns, with numeric `id` sorted ascending and `label` clipped away from 0/1 for logloss safety, writing `submission.csv` to the working directory.'
- What this solution (achieved 0.68986) has done: 'Your current score (0.68852, lower is better) is worse than the target (0.61083), so we should improve the predictions slightly without changing the core “single scalar feature + Gaussian NB” logic. The biggest low-risk gain here is to avoid distribution shift between train and test by fitting a simple 1D calibration (Platt scaling / logistic regression) on top of your existing NB logit using a deterministic train/validation split; this keeps the same feature extractor and generative model, only calibrating the probability mapping to reduce log loss. I also ensure the submission aligns to the sample_submission ids (including filling any missing ids) and clip probabilities more conservatively for logloss safety. All changes are small, deterministic, and should move logloss down toward the target.'
- What this solution (achieved 0.68779) has done: 'Your current logloss (0.68986, lower is better) is worse than the target (0.61083), so we should improve the probability calibration without changing the core “1D grayscale mean feature + Gaussian NB + sigmoid + Platt scaling” pipeline. The biggest low-risk issue here is that Platt scaling is currently fit on a validation set whose logits are ordered as all cats then all dogs; with batch gradient descent, that ordering can bias/slow convergence and yield a suboptimal calibrator. I keep the same optimizer and objective, but shuffle the calibration training pairs deterministically and standardize logits (a linear transform) only inside the calibrator to make optimization better-conditioned (this does not change core semantics, it’s still sigmoid(a*logit+b)). Finally, I compute the NB parameters from all training data (cat+dog) while still fitting calibration on the held-out split, which typically reduces noise in the base logit and improves logloss with minimal change.'
- What this solution (achieved 0.68778) has done: 'Your current score (0.68779, lower is better) is still worse than the target (0.61083), so we should improve log loss cautiously without changing the core pipeline (1D grayscale-mean feature → Gaussian NB logit → Platt sigmoid calibration). The smallest likely win is to make the Platt-scaling fit numerically better and less underfit by (a) removing unnecessary L2 shrinkage on (a,b) and (b) increasing iterations (keeping the same objective and gradient descent training loop). Additionally, I switch the NB variance estimate to an unbiased estimate (ddof=1) to reduce underestimation of variance, which commonly makes probabilities less overconfident and improves log loss. Submission formatting and ID alignment remain unchanged.'
- What this solution (achieved 0.68778) has done: 'Your current logloss (0.68778, lower is better) is still worse than the target (0.61083), so we should make a small, low-risk improvement that keeps your exact pipeline (1D grayscale mean → Gaussian NB logit → Platt sigmoid calibration). The biggest likely issue is the Platt scaling optimizer: plain gradient descent with fixed LR can converge to a suboptimal (a,b), hurting calibration and logloss. I keep the same objective and model, but fit (a,b) with a stable Newton/IRLS update for logistic regression in 2 parameters (still pure-numpy), which is deterministic and typically finds a much better optimum. Submission formatting/ID alignment stays identical, and probabilities remain safely clipped for logloss.'
- What this solution (achieved 0.68791) has done: 'Your current score (0.68778, lower is better) is still worse than the target (0.61083), so we should improve log loss with minimal risk while preserving your exact pipeline (1D grayscale mean → Gaussian NB logit → Platt sigmoid calibration). The smallest high-impact issue is that the NB logit is missing the Gaussian normalization term, which matters whenever class variances differ; adding it is still the same Gaussian NB model, just the correct likelihood ratio, and typically reduces logloss. I also fit calibration on out-of-fold (OOF) logits from the same split you already created (train vs val), instead of using NB parameters fit on all data for the validation logits; this avoids leakage into the calibrator and tends to improve generalization. Submission formatting and probability clipping remain unchanged.'

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
cat_val_feats = np.array(
    [load_image_feature(p) for p in cat_val_paths], dtype=np.float32
)
dog_val_feats = np.array(
    [load_image_feature(p) for p in dog_val_paths], dtype=np.float32
)

mu_cat_cal = float(cat_tr_feats.mean())
mu_dog_cal = float(dog_tr_feats.mean())
var_cat_cal = float(cat_tr_feats.var(ddof=1) + 1e-8)
var_dog_cal = float(dog_tr_feats.var(ddof=1) + 1e-8)
prior_dog_cal = float(len(dog_tr_feats) / (len(dog_tr_feats) + len(cat_tr_feats)))

cat_all_feats = np.concatenate([cat_tr_feats, cat_val_feats]).astype(
    np.float32, copy=False
)
dog_all_feats = np.concatenate([dog_tr_feats, dog_val_feats]).astype(
    np.float32, copy=False
)

mu_cat = float(cat_all_feats.mean())
mu_dog = float(dog_all_feats.mean())
var_cat = float(cat_all_feats.var(ddof=1) + 1e-8)
var_dog = float(dog_all_feats.var(ddof=1) + 1e-8)
prior_dog = float(len(dog_all_feats) / (len(dog_all_feats) + len(cat_all_feats)))

(mu_cat, mu_dog, var_cat, var_dog, prior_dog)




## === cell 3
def nb_logit_from_feature_params(
    x, mu_cat_p, mu_dog_p, var_cat_p, var_dog_p, prior_dog_p
):
    """
    Change to move logloss toward target: use the correct Gaussian NB log-likelihood ratio.
    The previous code used only the quadratic term and omitted -0.5*log(var), which matters
    when variances differ; this keeps the same core Gaussian NB model but fixes the logit.
    """
    logp_dog = math.log(prior_dog_p + 1e-12) - 0.5 * (
        math.log(var_dog_p) + ((x - mu_dog_p) ** 2) / var_dog_p
    )
    logp_cat = math.log(1.0 - prior_dog_p + 1e-12) - 0.5 * (
        math.log(var_cat_p) + ((x - mu_cat_p) ** 2) / var_cat_p
    )
    return logp_dog - logp_cat


def sigmoid(z):
    if z > 50:
        return 1.0
    if z < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_proba_dog_from_feature(x):
    return sigmoid(
        nb_logit_from_feature_params(x, mu_cat, mu_dog, var_cat, var_dog, prior_dog)
    )


p_at_cat_mean = predict_proba_dog_from_feature(mu_cat)
p_at_dog_mean = predict_proba_dog_from_feature(mu_dog)
(p_at_cat_mean, p_at_dog_mean)




## === cell 4
def fit_platt_scaling_newton(logits, y, l2=0.0, iters=50, damping=1e-6):
    """
    Newton/IRLS solve for Platt scaling p=sigmoid(a*logit+b).
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    theta = np.array([0.0, 1.0], dtype=np.float64)

    for _ in range(iters):
        z = theta[0] + theta[1] * x
        z = np.clip(z, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))

        g0 = (p - y).mean() + l2 * theta[0]
        g1 = ((p - y) * x).mean() + l2 * theta[1]
        g = np.array([g0, g1], dtype=np.float64)

        w = p * (1.0 - p)
        h00 = w.mean() + l2 + damping
        h01 = (w * x).mean()
        h11 = (w * x * x).mean() + l2 + damping
        H = np.array([[h00, h01], [h01, h11]], dtype=np.float64)

        step = np.linalg.solve(H, g)
        theta -= step

        if float(np.max(np.abs(step))) < 1e-10:
            break

    b = float(theta[0])
    a = float(theta[1])
    return a, b


val_logits = np.concatenate(
    [
        np.array(
            [
                nb_logit_from_feature_params(
                    x, mu_cat_cal, mu_dog_cal, var_cat_cal, var_dog_cal, prior_dog_cal
                )
                for x in cat_val_feats
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                nb_logit_from_feature_params(
                    x, mu_cat_cal, mu_dog_cal, var_cat_cal, var_dog_cal, prior_dog_cal
                )
                for x in dog_val_feats
            ],
            dtype=np.float64,
        ),
    ]
)
val_y = np.array([0] * len(cat_val_feats) + [1] * len(dog_val_feats), dtype=np.float64)

cal_rng = np.random.RandomState(42)
perm = cal_rng.permutation(len(val_logits))
val_logits_shuf = val_logits[perm]
val_y_shuf = val_y[perm]

m = float(val_logits_shuf.mean())
s = float(val_logits_shuf.std() + 1e-12)
val_logits_std = (val_logits_shuf - m) / s

a_std, b_std = fit_platt_scaling_newton(
    val_logits_std, val_y_shuf, l2=0.0, iters=50, damping=1e-6
)

a_cal = float(a_std / s)
b_cal = float(b_std - (a_std * m / s))


def predict_proba_dog_calibrated_from_feature(x):
    base_logit = nb_logit_from_feature_params(
        x, mu_cat, mu_dog, var_cat, var_dog, prior_dog
    )
    return sigmoid(a_cal * base_logit + b_cal)


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
