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

3.9

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

0.8170142036869145

# 6. Current score

0.13528

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The first failure (`MessageFactory` / `GetPrototype`) is an environment-level protobuf/TensorFlow incompatibility triggered just by importing TensorFlow/Keras; we can avoid it entirely by not using TF for inference. The second failure is a missing external SavedModel path (`../input/cassava-abhinay/model`), so even if TF imported, the model would not load. To make the notebook run end-to-end and always emit a valid `submission.csv` in the required format, I switch to a deterministic, lightweight baseline that reads the provided `sample_submission.csv` and outputs a constant label (this is score-limited but stable and valid). All paths remain within the provided dataset, and the output schema matches Kaggle’s required columns exactly.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, and the main issue is that the submission is a constant label, which cannot reach the desired accuracy. To move toward the target with minimal logic changes and no external dependencies, I replace the constant label with a deterministic, legitimate baseline: compute the most frequent class (mode) from the provided `train.csv` and predict that class for every test image. This keeps the “single-rule classifier” core approach intact (still no model/training loop), but typically boosts accuracy substantially on this dataset due to class imbalance, moving the score closer to the target. I also add a small safety fallback to the sample’s label distribution if `train.csv` is unexpectedly unavailable, and keep the exact submission schema and output path unchanged.'
- What this solution (achieved 0.30157) has done: 'Your current approach predicts a single label for all test images, which caps accuracy well below the target; to move toward 0.817 with minimal changes, we keep the “no training loop / no deep model” core logic but upgrade the rule from a global majority class to a simple, legitimate nearest-neighbor classifier in pixel space. This uses only the provided `train_images/` and `test_images/` folders plus `train.csv`/`sample_submission.csv`, and relies only on standard libraries already available (PIL is typically present in Kaggle). To stay within the 600s limit, we downsample images to a small fixed size, normalize, and classify each test image by the label of the most similar training image (1-NN), which should substantially improve accuracy on this dataset compared to majority class. The submission format and output path remain unchanged and a safe fallback to majority-class is kept if image decoding is unavailable.'
- What this solution (achieved 0.14126) has done: 'Your 1-NN pixel baseline is doing worse than even the majority-class baseline largely because it compares raw pixels without any normalization to reduce background/lighting effects and without restricting comparisons to a manageable, representative reference set. To move the score upward toward the 0.817 target with minimal change in “core logic” (still cosine-similarity 1-NN on downsampled images), I (1) apply a simple per-image standardization (subtract mean, divide by std) before similarity, and (2) build the neighbor set from a stratified subset of training images (equal per class) to reduce noise and improve neighbor quality while keeping runtime within limits. I also keep the same safe fallback to majority label and preserve the exact submission format/path.'
- What this solution (achieved 0.11472) has done: 'Your current 1-NN baseline is underperforming because raw pixel cosine similarity remains very sensitive to background/pose/lighting, and your reference set may include many redundant or low-signal images. To move the score up toward the 0.817 target without changing the overall “image → vector → cosine 1-NN” core logic, I (1) switch the image vector to a simple multi-feature representation (still computed from the downsampled image) that includes normalized RGB plus a lightweight edge-magnitude channel, and (2) use 3-NN with cosine similarity and majority vote (still the same nearest-neighbor approach, just slightly more robust). I also modestly increase the stratified reference set size while keeping runtime bounded, and keep the same strict submission formatting and majority-label fallback.'
- What this solution (achieved 0.13042) has done: 'Your current 3-NN cosine baseline is far below the target, so we need a small but meaningful representation upgrade while keeping the same “image → vector → cosine KNN” core logic. I keep your KNN pipeline intact, but (1) increase input resolution moderately and switch resize to bicubic for better detail retention, and (2) replace the single edge channel with simple multi-scale gradient magnitude (Sobel-like) plus an extra downsampled color feature to add a bit of shape robustness without changing the modeling approach. I also make the K vote slightly similarity-weighted (still KNN voting) to reduce ties and noisy neighbors, and bump the stratified reference set modestly while staying within the 600s budget. The submission writing/format stays identical.'
- What this solution (achieved 0.11659) has done: 'Your current KNN is far below the target, so we need a small, legitimate accuracy lift without changing the overall “image → vector → cosine KNN” pipeline. The biggest minimal win is to reduce noise in the reference set: instead of sampling randomly per class, we select “prototype” training images that are most representative of each class (closest to the class mean in your existing feature space). This keeps the same features, same cosine similarity, and same KNN voting, but makes neighbors more reliable and typically increases accuracy. I also make the similarity weights use a stable softmax over the top‑K similarities (still similarity-weighted voting) to reduce sensitivity to the arbitrary min-shift and improve class scoring consistency.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11659) is far below the target (0.8170), so we should make a small, legitimate accuracy improvement while preserving the same “image → feature vector → cosine similarity → KNN vote” core pipeline. The most impactful minimal issue here is feature mismatch between train and test: training prototypes use standardized features, but test vectors are not standardized before similarity, making cosine comparisons inconsistent and hurting KNN. I standardize the test feature vector with the exact same `_standardize_flat` steps applied to its subparts (big+small) before concatenation, matching the training representation. I also make prototype mean computation use the full available pool (instead of only the first `proto_k`) to select more representative prototypes without changing the approach.'
- What this solution (achieved 0.12332) has done: 'Your current KNN is extremely slow/heavy in feature dimensionality (128×128×5 + 64×64×3) and thus ends up effectively behaving noisy under the 600s constraint, keeping accuracy near random. To move the score upward toward the 0.817 target while preserving the exact same core pipeline (image→feature vector→cosine similarity→KNN vote), I only change the feature resolution (downsample) and slightly adjust K/temperature so the same logic runs more reliably and produces more meaningful neighbors. This keeps the same standardization, gradient-magnitude channels, cosine similarity, prototype selection, and similarity-weighted voting—just with a lighter representation that typically improves nearest-neighbor stability here. The script still runs end-to-end and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.14462) has done: 'Your current score (0.12332) is far below the target (0.8170), so we should increase performance, but with minimal changes that keep your exact “image → handcrafted vector → cosine similarity → KNN vote” core logic. The biggest issue is that the current feature vector is extremely high-dimensional, making cosine similarities noisy and unstable; we keep the same ingredients (RGB + gradient magnitudes + small RGB), but add a fixed random projection to a much lower dimension to make neighbors more meaningful and speed up dot products. We also slightly adjust K and temperature to be less “peaky” so voting is more robust, without changing the inference approach. Submission formatting and paths remain identical, and the majority-label fallback stays intact.'
- What this solution (achieved 0.13191) has done: 'Your current KNN pipeline is legitimate but its random projection and cosine voting are likely too noisy, keeping accuracy near chance; to move toward the 0.817 target with minimal core-logic change, I keep the exact “image → handcrafted vector → random projection → cosine KNN (similarity-weighted)” approach but make the representation more stable. Specifically, I (1) increase the projection dimension so neighborhoods are less distorted, (2) L2-normalize the raw feature vector before projection to reduce scale sensitivity between the big/small parts, and (3) slightly increase K and soften the temperature so voting is less brittle. These are small, localized changes that usually improve nearest-neighbor reliability without changing the overall method or I/O, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.13528) has done: 'The timeout is dominated by (1) per-image PIL feature extraction repeated thousands of times, and (2) an expensive full similarity matrix computed in float32 with default BLAS threading. I preserve the exact KNN/prototype logic and math, but make it run faster by (a) precomputing/caching all test embeddings once, (b) using a single batched matrix multiply for all test similarities (same result as looping), (c) using faster vectorized gradient computation, and (d) controlling BLAS thread counts to avoid oversubscription stalls. These changes are provably equivalent (same inputs → same embeddings → same similarities/top‑K/voting), just with less Python overhead and fewer repeated transforms.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"

if not os.path.exists(sample_path):
    raise FileNotFoundError(
        f"sample_submission.csv not found at expected paths. Last tried: {sample_path}"
    )

sample = pd.read_csv(sample_path)
assert {"image_id", "label"}.issubset(
    sample.columns
), f"Unexpected submission columns: {sample.columns.tolist()}"
print("Loaded sample_submission:", sample.shape)




## === cell 2
train_path = os.path.join(DATA_DIR, "train.csv")
if not os.path.exists(train_path):
    train_path = "../input/train.csv"

DEFAULT_LABEL_FALLBACK = 0

if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    if not {"image_id", "label"}.issubset(train_df.columns):
        raise ValueError(f"Unexpected train columns: {train_df.columns.tolist()}")
    mode_labels = train_df["label"].mode(dropna=True)
    if len(mode_labels) == 0:
        majority_label = int(DEFAULT_LABEL_FALLBACK)
    else:
        majority_label = int(mode_labels.min())
    print(f"Loaded train.csv: {train_df.shape}. Using majority_label={majority_label}.")
else:
    raise FileNotFoundError(
        f"train.csv not found at expected paths. Last tried: {train_path}"
    )

train_img_dir = os.path.join(DATA_DIR, "train_images")
test_img_dir = os.path.join(DATA_DIR, "test_images")
if not os.path.isdir(train_img_dir):
    train_img_dir = "../input/train_images"
if not os.path.isdir(test_img_dir):
    test_img_dir = "../input/test_images"

use_image_knn = os.path.isdir(train_img_dir) and os.path.isdir(test_img_dir)
print("train_images dir:", train_img_dir, "exists:", os.path.isdir(train_img_dir))
print("test_images  dir:", test_img_dir, "exists:", os.path.isdir(test_img_dir))

pred_labels = None

if use_image_knn:
    try:
        from PIL import Image

        IMG_SIZE = (64, 64)
        SMALL_SIZE = (32, 32)
        CROP_SIZE = (48, 48)

        def _standardize_flat(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            m = float(x.mean())
            s = float(x.std())
            if s < 1e-6:
                s = 1e-6
            return (x - m) / s

        def _grad_mag(gray: np.ndarray) -> np.ndarray:
            gray = gray.astype(np.float32, copy=False)
            gx = np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1)
            gy = np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0)
            gx[:, 0] = 0.0
            gx[:, -1] = 0.0
            gy[0, :] = 0.0
            gy[-1, :] = 0.0
            return np.sqrt(gx * gx + gy * gy)

        def _rgb_hist(arr_rgb01: np.ndarray, bins: int = 8) -> np.ndarray:
            hists = []
            for c in range(3):
                h, _ = np.histogram(
                    arr_rgb01[..., c].ravel(),
                    bins=bins,
                    range=(0.0, 1.0),
                    density=False,
                )
                hists.append(h.astype(np.float32))
            h = np.concatenate(hists, axis=0)
            h = h / (h.sum() + 1e-6)
            return h.astype(np.float32)

        def _center_crop(im: Image.Image, frac: float = 0.75) -> Image.Image:
            w, h = im.size
            cw = int(round(w * frac))
            ch = int(round(h * frac))
            left = max(0, (w - cw) // 2)
            top = max(0, (h - ch) // 2)
            return im.crop((left, top, left + cw, top + ch))

        d_raw = (
            IMG_SIZE[0] * IMG_SIZE[1] * 5
            + SMALL_SIZE[0] * SMALL_SIZE[1] * 3
            + CROP_SIZE[0] * CROP_SIZE[1] * 3
            + 24  # 8 bins * 3 channels
        )

        proj_dim = 4096
        rng_proj = np.random.RandomState(0)
        R = rng_proj.normal(
            loc=0.0, scale=1.0 / np.sqrt(proj_dim), size=(d_raw, proj_dim)
        ).astype(np.float32)

        _vec_cache = {}

        def img_to_vec(path):
            v = _vec_cache.get(path)
            if v is not None:
                return v
            with Image.open(path) as im:
                im = im.convert("RGB")
                im_big = im.resize(IMG_SIZE, resample=Image.BICUBIC)
                im_small = im.resize(SMALL_SIZE, resample=Image.BICUBIC)
                im_crop = _center_crop(im, frac=0.75).resize(
                    CROP_SIZE, resample=Image.BICUBIC
                )

                arr_big = np.asarray(im_big, dtype=np.float32) / 255.0  # (H,W,3)
                arr_small = np.asarray(im_small, dtype=np.float32) / 255.0  # (h,w,3)
                arr_crop = np.asarray(im_crop, dtype=np.float32) / 255.0  # (cH,cW,3)

            rgb = arr_big

            gray = (
                0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            ).astype(np.float32)
            g1 = _grad_mag(gray)

            gray_small = (
                0.2989 * arr_small[..., 0]
                + 0.5870 * arr_small[..., 1]
                + 0.1140 * arr_small[..., 2]
            ).astype(np.float32)
            g2_small = _grad_mag(gray_small)
            g2 = np.asarray(
                Image.fromarray(g2_small).resize(IMG_SIZE, resample=Image.BILINEAR),
                dtype=np.float32,
            )

            feat_big = np.concatenate([rgb, g1[..., None], g2[..., None]], axis=-1)
            x_big = feat_big.reshape(-1).astype(np.float32)
            x_small = arr_small.reshape(-1).astype(np.float32)

            x_crop = arr_crop.reshape(-1).astype(np.float32)
            x_hist = _rgb_hist(arr_small, bins=8)

            x_big = _standardize_flat(x_big)
            x_small = _standardize_flat(x_small)
            x_crop = _standardize_flat(x_crop)
            x_hist = _standardize_flat(x_hist)

            x_raw = np.concatenate([x_big, x_small, x_crop, x_hist], axis=0).astype(
                np.float32
            )

            x_raw = x_raw / (np.linalg.norm(x_raw) + 1e-8)

            x_proj = x_raw @ R
            x_proj = _standardize_flat(x_proj)
            x_proj = x_proj.astype(np.float32)

            _vec_cache[path] = x_proj
            return x_proj

        per_class_pool = 1600
        per_class_ref = 800

        proto_k = None  # None => use full pool mean

        ref_ids = []
        ref_labels = []

        pooled_vec_by_id = {}

        for lbl, g in train_df.groupby("label"):
            g = g.sample(n=min(per_class_pool, len(g)), random_state=0).reset_index(
                drop=True
            )
            ids = g["image_id"].tolist()

            X_pool = np.empty((len(ids), proj_dim), dtype=np.float32)
            for i, img_id in enumerate(ids):
                p = os.path.join(train_img_dir, img_id)
                if not os.path.exists(p):
                    X_pool[i, :] = 0.0
                else:
                    v = img_to_vec(p)
                    X_pool[i, :] = v
                    pooled_vec_by_id[img_id] = v

            Xn = X_pool / (np.linalg.norm(X_pool, axis=1, keepdims=True) + 1e-8)

            if proto_k is None:
                m = Xn.mean(axis=0)
            else:
                m = Xn[: min(proto_k, len(Xn))].mean(axis=0)

            m = m / (np.linalg.norm(m) + 1e-8)

            sims_to_mean = Xn @ m
            take = min(per_class_ref, len(ids))

            top_idx = np.argpartition(-sims_to_mean, take - 1)[:take]
            top_idx = top_idx[np.argsort(-sims_to_mean[top_idx])]

            for i in top_idx:
                ref_ids.append(ids[i])
                ref_labels.append(int(lbl))

        ref_df = pd.DataFrame({"image_id": ref_ids, "label": ref_labels})
        ref_df = ref_df.sample(frac=1.0, random_state=0).reset_index(drop=True)

        train_ids = ref_df["image_id"].values
        train_labels = ref_df["label"].astype(int).values
        print(
            "Using prototype reference set size:",
            len(train_ids),
            "label counts:",
            ref_df["label"].value_counts().to_dict(),
        )

        X_train = np.empty((len(train_ids), proj_dim), dtype=np.float32)
        missing_train = 0
        extracted_train = 0
        reused_train = 0
        for i, img_id in enumerate(train_ids):
            v = pooled_vec_by_id.get(img_id)
            if v is not None:
                X_train[i, :] = v
                reused_train += 1
                continue
            p = os.path.join(train_img_dir, img_id)
            if not os.path.exists(p):
                missing_train += 1
                X_train[i, :] = 0.0
            else:
                X_train[i, :] = img_to_vec(p)
                extracted_train += 1
        if missing_train:
            print(
                f"Warning: {missing_train} reference train images missing; filled with zeros."
            )
        print(
            f"Reference vectors reused={reused_train}, newly_extracted={extracted_train}"
        )

        X_train_unit = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        X_train_unit_T = np.ascontiguousarray(X_train_unit.T)

        test_ids = sample["image_id"].values
        pred_labels = np.empty(len(test_ids), dtype=np.int32)

        K = 21
        temp = 0.22

        n_classes = int(train_df["label"].max()) + 1
        train_labels_i32 = train_labels.astype(np.int32, copy=False)

        X_test = np.empty((len(test_ids), proj_dim), dtype=np.float32)
        present = np.ones(len(test_ids), dtype=bool)
        for i, img_id in enumerate(test_ids):
            p = os.path.join(test_img_dir, img_id)
            if not os.path.exists(p):
                present[i] = False
                X_test[i, :] = 0.0
            else:
                x = img_to_vec(p)
                x = x / (np.linalg.norm(x) + 1e-8)
                X_test[i, :] = x

        if not np.all(present):
            pred_labels[~present] = majority_label

        if np.any(present):
            Xp = X_test[present]
            sims = Xp @ X_train_unit_T  # (N_present, N_train)

            topk_idx = np.argpartition(-sims, K - 1, axis=1)[:, :K]
            topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
            topk_labels = train_labels_i32[topk_idx]

            max_s = np.max(topk_sims, axis=1, keepdims=True)
            z = (topk_sims - max_s) / temp
            z = np.clip(z, -50.0, 0.0)
            w = np.exp(z).astype(np.float32)

            scores = np.zeros((w.shape[0], n_classes), dtype=np.float32)
            np.add.at(scores, (np.arange(w.shape[0])[:, None], topk_labels), w)
            preds_present = np.argmax(scores, axis=1).astype(np.int32)

            pred_labels[present] = preds_present

        print("Image K-NN predictions computed:", pred_labels.shape)

    except Exception as e:
        print(
            "Falling back to majority-label baseline due to image KNN error:", repr(e)
        )
        pred_labels = None

if pred_labels is None:
    sample["label"] = int(majority_label)
else:
    sample["label"] = pred_labels.astype(int)

sub = sample[["image_id", "label"]].copy()
sub["label"] = sub["label"].astype(int)




## === cell 3
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote submission to {out_path} with shape: {sub.shape}")
print(sub.head())




## === cell 4
if sub["image_id"].isna().any():
    raise ValueError("Found NaNs in image_id column.")
if sub["label"].isna().any():
    raise ValueError("Found NaNs in label column.")
if not np.issubdtype(sub["label"].dtype, np.integer):
    raise TypeError(f"label dtype must be integer, got {sub['label'].dtype}")
if len(sub) != len(sample):
    raise ValueError("Submission row count mismatch vs sample_submission.")
print("Submission validation passed.")
