# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]


def _resolve_base_dir(candidates):
    for c in candidates:
        if not os.path.exists(c):
            continue

        if os.path.isfile(os.path.join(c, "sample_submission.csv")) and os.path.isfile(
            os.path.join(c, "test.csv")
        ):
            return c

        nested = os.path.join(c, "plant-pathology-2020-fgvc7")
        if os.path.isdir(nested):
            if os.path.isfile(
                os.path.join(nested, "sample_submission.csv")
            ) and os.path.isfile(os.path.join(nested, "test.csv")):
                return nested

    return None


BASE_DIR = _resolve_base_dir(BASE_CANDIDATES)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory containing sample_submission.csv and test.csv. "
        f"Tried: {BASE_CANDIDATES}"
    )

print("Resolved BASE_DIR:", BASE_DIR)
print(
    "Files present:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(BASE_DIR, f))
    ],
)



## === cell 1
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

img_dir = os.path.join(BASE_DIR, "images")
if not os.path.isdir(img_dir):
    nested_img_dir = os.path.join(BASE_DIR, "plant-pathology-2020-fgvc7", "images")
    if os.path.isdir(nested_img_dir):
        img_dir = nested_img_dir

if not os.path.isdir(img_dir):
    raise FileNotFoundError(
        f"Could not find images directory. Tried: {os.path.join(BASE_DIR,'images')}"
    )

print("Resolved img_dir:", img_dir)


def _image_path(image_id: str) -> str:
    candidates = [image_id]
    if "." not in os.path.basename(image_id):
        candidates = [
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            f"{image_id}.png",
            image_id,
        ]

    for name in candidates:
        p = os.path.join(img_dir, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"Image file not found for image_id='{image_id}'. Tried: "
        + ", ".join([os.path.join(img_dir, n) for n in candidates])
    )


def _extract_features(image_id: str, size=(96, 96)) -> np.ndarray:
    """
    Core logic preserved: deterministic, hand-crafted image stats + thumbnail.

    Score-relevant minimal extension (to move ROC AUC up toward target):
    - Add compact blob/spot descriptors via Difference-of-Gaussians (DoG) on grayscale.
      This targets rust/scab lesion texture without changing model/training semantics.
    """
    p = _image_path(image_id)
    with Image.open(p) as im:
        im = im.convert("RGB")
        im_small = im.resize(size, resample=Image.BILINEAR)

        arr = np.asarray(im_small, dtype=np.float32) / 255.0  # (H,W,3)

        rgb_mean = arr.mean(axis=(0, 1))
        rgb_std = arr.std(axis=(0, 1))

        hsv_img = im_small.convert("HSV")
        hsv = np.asarray(hsv_img, dtype=np.float32)
        hsv[..., 0] = hsv[..., 0] / 255.0
        hsv[..., 1] = hsv[..., 1] / 255.0
        hsv[..., 2] = hsv[..., 2] / 255.0
        hsv_mean = hsv.mean(axis=(0, 1))
        hsv_std = hsv.std(axis=(0, 1))

        gray = np.asarray(im_small.convert("L"), dtype=np.float32) / 255.0
        gray_mean = np.array([gray.mean()], dtype=np.float32)
        gray_std = np.array([gray.std()], dtype=np.float32)

        eps = 1e-6
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        rg_ratio = np.array([(r.mean() + eps) / (g.mean() + eps)], dtype=np.float32)
        rb_ratio = np.array([(r.mean() + eps) / (b.mean() + eps)], dtype=np.float32)
        gb_ratio = np.array([(g.mean() + eps) / (b.mean() + eps)], dtype=np.float32)
        intensity = (r + g + b) / 3.0
        r_over_i = np.array(
            [(r.mean() + eps) / (intensity.mean() + eps)], dtype=np.float32
        )
        g_over_i = np.array(
            [(g.mean() + eps) / (intensity.mean() + eps)], dtype=np.float32
        )
        b_over_i = np.array(
            [(b.mean() + eps) / (intensity.mean() + eps)], dtype=np.float32
        )

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = np.sqrt(gx * gx + gy * gy)
        grad_mean = np.array([grad.mean()], dtype=np.float32)
        grad_std = np.array([grad.std()], dtype=np.float32)
        grad_p90 = np.array([np.quantile(grad, 0.90)], dtype=np.float32)

        q_levels = (0.10, 0.50, 0.90)
        rgb_q = np.concatenate(
            [np.quantile(arr[..., c], q_levels).astype(np.float32) for c in range(3)],
            axis=0,
        )

        def _grad_stats(channel_2d: np.ndarray) -> np.ndarray:
            gx_c = np.diff(channel_2d, axis=1)
            gy_c = np.diff(channel_2d, axis=0)
            gx_c = np.pad(gx_c, ((0, 0), (0, 1)), mode="edge")
            gy_c = np.pad(gy_c, ((0, 1), (0, 0)), mode="edge")
            gmag = np.sqrt(gx_c * gx_c + gy_c * gy_c)
            return np.array(
                [gmag.mean(), gmag.std(), np.quantile(gmag, 0.90)], dtype=np.float32
            )

        grad_rgb = np.concatenate([_grad_stats(arr[..., c]) for c in range(3)], axis=0)

        hue = hsv[..., 0].ravel()
        hue_hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), density=True)
        hue_hist = hue_hist.astype(np.float32)

        exg = (2.0 * g - r - b).astype(np.float32)
        exg_mean = np.array([exg.mean()], dtype=np.float32)
        exg_std = np.array([exg.std()], dtype=np.float32)
        exg_q = np.array(
            [np.quantile(exg, 0.10), np.quantile(exg, 0.50), np.quantile(exg, 0.90)],
            dtype=np.float32,
        )

        g8 = (gray * 255.0).astype(np.uint8)
        gc = g8[1:-1, 1:-1].astype(np.int16)
        lbp = np.zeros_like(gc, dtype=np.uint8)
        lbp |= ((g8[0:-2, 0:-2].astype(np.int16) >= gc) << 7).astype(np.uint8)
        lbp |= ((g8[0:-2, 1:-1].astype(np.int16) >= gc) << 6).astype(np.uint8)
        lbp |= ((g8[0:-2, 2:].astype(np.int16) >= gc) << 5).astype(np.uint8)
        lbp |= ((g8[1:-1, 2:].astype(np.int16) >= gc) << 4).astype(np.uint8)
        lbp |= ((g8[2:, 2:].astype(np.int16) >= gc) << 3).astype(np.uint8)
        lbp |= ((g8[2:, 1:-1].astype(np.int16) >= gc) << 2).astype(np.uint8)
        lbp |= ((g8[2:, 0:-2].astype(np.int16) >= gc) << 1).astype(np.uint8)
        lbp |= ((g8[1:-1, 0:-2].astype(np.int16) >= gc) << 0).astype(np.uint8)
        lbp16 = (lbp >> 4).ravel()
        lbp_hist, _ = np.histogram(lbp16, bins=16, range=(0, 16), density=True)
        lbp_hist = lbp_hist.astype(np.float32)

        rgb_hist = []
        for c in range(3):
            h, _ = np.histogram(
                arr[..., c].ravel(), bins=8, range=(0.0, 1.0), density=True
            )
            rgb_hist.append(h.astype(np.float32))
        rgb_hist = np.concatenate(rgb_hist, axis=0)

        gray_q = np.array(
            [np.quantile(gray, 0.05), np.quantile(gray, 0.50), np.quantile(gray, 0.95)],
            dtype=np.float32,
        )
        dark_frac = np.array([(gray < 0.25).mean()], dtype=np.float32)
        bright_frac = np.array([(gray > 0.75).mean()], dtype=np.float32)

        lab = np.asarray(im_small.convert("LAB"), dtype=np.float32)  # 0..255
        lab_hist_parts = []
        for c in range(3):
            h, _ = np.histogram(
                lab[..., c].ravel(), bins=8, range=(0.0, 255.0), density=True
            )
            lab_hist_parts.append(h.astype(np.float32))
        lab_hist = np.concatenate(lab_hist_parts, axis=0)

        s = hsv[..., 1]
        v = hsv[..., 2]
        mask = (s > 0.2) & (v > 0.2)
        if mask.mean() < 1e-3:
            mask = np.ones_like(mask, dtype=bool)
        hmask = hsv[..., 0][mask]
        yellow_frac = np.array(
            [((hmask >= 0.10) & (hmask <= 0.20)).mean()], dtype=np.float32
        )
        orange_frac = np.array(
            [((hmask > 0.03) & (hmask < 0.10)).mean()], dtype=np.float32
        )
        red_frac = np.array(
            [((hmask <= 0.03) | (hmask >= 0.97)).mean()], dtype=np.float32
        )

        gx2 = np.diff(gray, axis=1)
        gy2 = np.diff(gray, axis=0)
        gx2 = np.pad(gx2, ((0, 0), (0, 1)), mode="edge")
        gy2 = np.pad(gy2, ((0, 1), (0, 0)), mode="edge")
        mag = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-12
        ang = (np.arctan2(gy2, gx2) + np.pi) / (2.0 * np.pi)  # 0..1
        mthr = np.quantile(mag, 0.75)
        edge_mask = mag >= mthr
        if edge_mask.mean() < 1e-3:
            edge_mask = np.ones_like(edge_mask, dtype=bool)
        ang_hist, _ = np.histogram(
            ang[edge_mask].ravel(), bins=8, range=(0.0, 1.0), density=True
        )
        ang_hist = ang_hist.astype(np.float32)

        q = np.clip((gray * 15.0).astype(np.int32), 0, 15)
        a = q[:, :-1].ravel()
        bq = q[:, 1:].ravel()
        idx = a * 16 + bq
        bc = np.bincount(idx, minlength=256).astype(np.float32)
        glcm = bc.reshape(16, 16)
        glcm = glcm + glcm.T  # symmetrize
        glcm_sum = glcm.sum()
        if glcm_sum <= 0:
            glcm_sum = 1.0
        P = glcm / glcm_sum

        i = np.arange(16, dtype=np.float32)
        j = np.arange(16, dtype=np.float32)
        I, J = np.meshgrid(i, j, indexing="ij")
        mu_i = (P.sum(axis=1) * i).sum()
        mu_j = (P.sum(axis=0) * j).sum()
        si = np.sqrt(((i - mu_i) ** 2 * P.sum(axis=1)).sum()) + 1e-6
        sj = np.sqrt(((j - mu_j) ** 2 * P.sum(axis=0)).sum()) + 1e-6

        contrast = ((I - J) ** 2 * P).sum()
        energy = (P * P).sum()
        homogeneity = (P / (1.0 + np.abs(I - J))).sum()
        correlation = (((I - mu_i) * (J - mu_j) * P).sum()) / (si * sj)
        glcm_stats = np.array(
            [contrast, energy, homogeneity, correlation], dtype=np.float32
        )

        def _gauss1d(sigma: float) -> np.ndarray:
            radius = int(max(1, np.ceil(3.0 * sigma)))
            x = np.arange(-radius, radius + 1, dtype=np.float32)
            k = np.exp(-(x * x) / (2.0 * sigma * sigma))
            k /= k.sum() + 1e-12
            return k

        def _sep_conv(img2d: np.ndarray, k: np.ndarray) -> np.ndarray:
            pad = len(k) // 2
            tmp = np.pad(img2d, ((0, 0), (pad, pad)), mode="reflect")
            tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, tmp)
            tmp2 = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
            out = np.apply_along_axis(
                lambda m: np.convolve(m, k, mode="valid"), 0, tmp2
            )
            return out.astype(np.float32)

        k1 = _gauss1d(1.0)
        k2 = _gauss1d(2.0)
        blur1 = _sep_conv(gray, k1)
        blur2 = _sep_conv(gray, k2)
        dog = blur1 - blur2
        dog_abs = np.abs(dog)

        dog_stats = np.array(
            [
                dog.mean(),
                dog.std(),
                np.quantile(dog_abs, 0.50),
                np.quantile(dog_abs, 0.90),
                (dog_abs > np.quantile(dog_abs, 0.90)).mean(),
            ],
            dtype=np.float32,
        )
        dog_hist, _ = np.histogram(
            dog_abs.ravel(),
            bins=8,
            range=(0.0, float(np.quantile(dog_abs, 0.99)) + 1e-6),
            density=True,
        )
        dog_hist = dog_hist.astype(np.float32)

        thumb = (
            np.asarray(
                im.resize((32, 32), resample=Image.BILINEAR).convert("L"),
                dtype=np.float32,
            )
            / 255.0
        ).ravel()

        feat = np.concatenate(
            [
                rgb_mean,
                rgb_std,
                hsv_mean,
                hsv_std,
                gray_mean,
                gray_std,
                rg_ratio,
                rb_ratio,
                gb_ratio,
                r_over_i,
                g_over_i,
                b_over_i,
                grad_mean,
                grad_std,
                grad_p90,
                rgb_q,
                grad_rgb,
                hue_hist,
                exg_mean,
                exg_std,
                exg_q,
                lbp_hist,
                rgb_hist,
                gray_q,
                dark_frac,
                bright_frac,
                lab_hist,
                yellow_frac,
                orange_frac,
                red_frac,
                ang_hist,
                glcm_stats,
                dog_stats,  # added
                dog_hist,  # added
                thumb,
            ],
            axis=0,
        )

        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feat


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for iid in df["image_id"].values:
        feats.append(_extract_features(iid))
    return np.vstack(feats)


X_train = _build_feature_matrix(train)
y_train = train[target_cols].values.astype(np.float32)

X_test = _build_feature_matrix(test)

y_class = np.argmax(y_train, axis=1).astype(np.int64)

base_lr = LogisticRegression(
    max_iter=6000,
    solver="lbfgs",
    C=0.5,
    multi_class="multinomial",
    class_weight=None,
    random_state=42,
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False, with_std=True)),
        ("lr", base_lr),
    ]
)

clf.fit(X_train, y_class)
proba = clf.predict_proba(X_test).astype(np.float32)  # shape (n_test, 4)

class_to_index = {c: i for i, c in enumerate(clf.named_steps["lr"].classes_)}
proba_ordered = np.zeros((proba.shape[0], len(target_cols)), dtype=np.float32)
for k in range(len(target_cols)):
    proba_ordered[:, k] = proba[:, class_to_index[k]]

proba_ordered = np.clip(proba_ordered, 1e-6, 1 - 1e-6)

sub = pd.DataFrame(proba_ordered, columns=target_cols)
sub.insert(0, "image_id", test["image_id"].values)

sub = sub[["image_id"] + target_cols]

assert sub.shape[0] == test.shape[0], "Submission row count must match test.csv"
assert list(sub.columns) == ["image_id"] + target_cols, "Submission columns mismatch"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
