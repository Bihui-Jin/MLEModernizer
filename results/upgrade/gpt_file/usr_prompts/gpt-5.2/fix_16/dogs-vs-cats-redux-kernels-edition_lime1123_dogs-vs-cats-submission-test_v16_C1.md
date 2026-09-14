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

0.0339717376907435

# 6. Current score

0.66545

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'Your notebook isn’t yielding a Kaggle score because it currently won’t run in a clean environment: it imports packages that are not installed (timm, albumentations, cv2, pytorch_lightning, sklearn), and it also points to an external checkpoint directory that likely doesn’t exist. To get a valid submission while preserving your inference semantics (probability of dog for each test image id), I’m making the smallest possible change: generate a properly formatted `submission.csv` directly from `sample_submission.csv` with a constant, safe probability (0.5) so the pipeline always produces a valid file. This does not try to maximize score; it just unblocks submission generation end-to-end under the “no external packages installed” constraint. Once you confirm a valid score is produced, we can re-enable model inference if/when those dependencies/checkpoints are available.'
- What this solution (achieved 0.69315) has done: 'I fix the crash by replacing the PNG-only decoder with a minimal JPEG/PNG reader that works using only the Python standard library (via `tkinter.PhotoImage` for PNG/GIF and a small baseline-JPEG decoder for typical Kaggle JPGs). I also make sure `model` is always defined (even when training fails) so submission generation never hits a `NameError`. To keep the core logic intact, I preserve the exact histogram Naive Bayes approach and only change image loading/decoding so it can actually read the provided `.jpg` files. Finally, I ensure the submission is written to `/kaggle/working/submission.csv` with `id,label` and sorted IDs.'
- What this solution (achieved 0.69027) has done: 'The timeout is dominated by per-image pure-Python decoding and per-scanline filter reconstruction (PNG) / entropy parsing (JPEG), repeated thousands of times. I keep the same exact feature (grayscale mean) and the same Naive Bayes histogram model, but make the extraction much faster by (1) using Pillow if available (common on Kaggle) to compute the same grayscale mean directly from decoded pixels, and (2) parallelizing feature extraction across CPU cores while keeping deterministic ordering. I also precompute gray means for all train+test images once and reuse them (instead of re-reading/decoding during both fit and predict), which is provably equivalent because the feature is a pure function of the file bytes. Finally, I speed up the histogram counting and prediction loops by avoiding repeated dictionary lookups and Python overhead.'
- What this solution (achieved 0.68878) has done: 'Your current log loss (~0.69) indicates the model is essentially guessing; the smallest way to move toward the much better target is to keep the exact same Naive Bayes-on-gray-mean core logic but (1) use all available training images instead of capping at 4000/class (this is not an approximation; it’s more complete fitting of the same model), and (2) make the grayscale-mean feature consistent between train and test by always decoding via PIL (your fallback JPEG “DC-only mean” is a much noisier proxy and can hurt calibration). I also fix the test image discovery so it pick the correct folder regardless of whether Kaggle provided `test/unknown` or `test/test`. These changes are directly aimed at improving log loss without changing the model family, loss, or prediction semantics, and still write a valid `submission.csv`.'
- What this solution (achieved 0.68934) has done: 'Your current score (~0.689 logloss) suggests the model’s predictions are close to uninformative; the simplest way to move toward the much better target is to keep the exact same Naive Bayes-on-gray-mean approach but make the “gray mean” feature more discriminative while still being the same kind of scalar summary. I therefore switch the feature from *mean grayscale intensity* to *mean “redness” (R−B) mapped to 0..255*, which is still a single per-image mean computed from decoded pixels and keeps the same histogram+Laplace NB core logic unchanged. To ensure stability and avoid silent fallbacks that hurt score, I also require PIL for decoding (otherwise fail fast) and keep the same test discovery/submission formatting. These minimal changes are directly aimed at improving separability/calibration under logloss without changing the model family, training loop, or inference semantics (still outputs P(dog) per test id).'
- What this solution (achieved 0.69144) has done: 'Your current logloss (~0.689) is far from the target (0.034), so we need a real but still minimal improvement while keeping the same histogram Naive Bayes core logic. The biggest issue is that the “redness mean” feature is likely not aligned with the dataset and your current `bins=32` is too coarse for a 1D continuous signal; increasing bins and slightly reducing Laplace smoothing improves calibration/separability without changing the model family or training loop. I also fix a platform issue: `multiprocessing.get_context("fork")` can fail in some Kaggle/Python 3.13 environments, so we fall back safely to `spawn` to ensure end-to-end execution. Finally, I make test discovery more robust by also checking the common nested folder path so all 2500 ids are found (avoiding 0.5 fallbacks that harm logloss).'
- What this solution (achieved 0.69201) has done: 'We need to move logloss down from ~0.691 toward 0.0339, so we must improve predictive power while keeping the same overall histogram Naive Bayes approach and a single scalar feature per image. The biggest likely issue is the current “redness (R−B)” feature is weak for this dataset; we switch back to a more standard, still-single-scalar feature: mean grayscale intensity computed with PIL (same model, bins, Laplace smoothing, training loop unchanged). To avoid harming score with accidental 0.5 fallbacks, we also broaden test image discovery by recursively indexing files under the chosen test root (still using only filenames like `123.jpg` to map ids). All changes are minimal and keep the same submission semantics (probability of dog per id) while improving separability and thus logloss.'
- What this solution (achieved 0.69201) has done: 'Your current score (~0.692 logloss) is far from the target (~0.034), so we should cautiously improve predictive power without changing the overall “single-scalar feature + histogram Naive Bayes” core logic. The smallest high-impact fix is to correct the training data source: this competition’s labels are in the filenames inside the original `train.zip` (cats/dogs mixed), but your code is training on a pre-split folder that may be incomplete/mismatched; we instead load from `/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/train` when present and parse labels from filenames. We keep the same grayscale-mean feature, histogram NB, bins, Laplace smoothing, and prediction semantics; we only change how train files are discovered to use the full labeled set. This should materially reduce logloss while keeping execution stable and still writing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.65985) has done: 'We need to move logloss down from ~0.692 toward 0.0339 (lower is better), so the model must become genuinely predictive; the current single-scalar “gray mean + histogram NB” is too weak. Keeping the same core logic (one scalar feature per image, histogram Naive Bayes, same training/prediction loops and loss semantics), I switch the scalar feature from mean grayscale intensity to mean **edge strength** (mean gradient magnitude) computed from the same decoded grayscale image; this is still a single number per image but much more discriminative for cats vs dogs. I also fix a likely major data issue: your sample submission has only 2500 rows, but the true test set is 12500; I build the submission IDs from discovered test images when possible (falling back to sample_submission only if needed) to avoid missing rows harming score. Finally, I keep the exact same submission format (`id,label`), deterministic ordering, and write `/kaggle/working/submission.csv` end-to-end.'
- What this solution (achieved 0.66545) has done: 'We need to move logloss down from 0.65985 toward 0.03397 (lower is better), so the smallest meaningful improvement is to keep your same “single scalar feature + histogram Naive Bayes” pipeline but make the scalar feature better calibrated and less clipped. I change only the feature binning so it uses robust percentiles computed from the training feature distribution (instead of a fixed 0..512 cap), which keeps the same NB logic but reduces saturation and improves probability estimates. I also build the submission ID list from discovered test images when available, but if only 2500 are present in this environment, we still produce a valid CSV (this doesn’t change semantics; it prevents accidental mismatch). All I/O paths, the model family, training loops, and submission format remain the same.'
- What this solution (achieved 0.66545) has done: 'Your current logloss (0.66545) is far above the target (0.03397), so we need a small but real accuracy lift while keeping your exact “single scalar feature + histogram Naive Bayes” core logic intact. The biggest issue is the feature: the mean edge-strength you compute is derived from an incorrect indexing bug (you compute gx using the same row for x±1 but forget to add `x` to the row offset for the center pixel), which makes the feature much noisier and close to random. I fix only that indexing so the gradient is computed at the correct pixel locations; everything else (PIL decoding, scalar-per-image, histogram NB, binning, Laplace smoothing, train/test discovery, and submission formatting) stays the same. This should move the score downward (better) toward the target without changing the modeling approach.'

# 9. Code solution

## === cell 0
import os
import re
import math
import struct
import zlib
import pandas as pd
import multiprocessing as mp

os.environ.setdefault("PYTHONHASHSEED", "0")


class Config:
    input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
    working_dir = "/kaggle/working"
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
    out_path = os.path.join(working_dir, "submission.csv")

    train_labeled_dir_candidates = [
        os.path.join(input_dir, "train", "train"),
        os.path.join(input_dir, "train"),
    ]

    train_cat_dir = os.path.join(input_dir, "train", "cat")
    train_dog_dir = os.path.join(input_dir, "train", "dog")

    test_candidates = [
        os.path.join(input_dir, "test", "unknown"),
        os.path.join(input_dir, "test", "test"),
        os.path.join(input_dir, "test", "test", "unknown"),
        os.path.join(input_dir, "test", "test", "test", "unknown"),
        os.path.join(input_dir, "test"),  # allow indexing under test root if nested
    ]

    max_train_per_class = None  # None => use all available

    bins = 256  # bins for the scalar feature
    laplace = 0.5  # smoothing for calibrated probabilities

    max_pixels_feature = 256 * 256  # subsampling cap for feature computation

    bin_low_q = 0.005
    bin_high_q = 0.995


cfg = Config()
os.makedirs(cfg.working_dir, exist_ok=True)



## === cell 1
_FEATURE_CACHE = {}

try:
    from PIL import Image  # commonly available on Kaggle

    _HAVE_PIL = True
except Exception:
    Image = None
    _HAVE_PIL = False


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_gray_mean_from_bytes(data, max_pixels=256 * 256):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")

    pos = 8
    width = height = None
    bit_depth = color_type = None
    idat = bytearray()

    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        pos += 8
        chunk = data[pos : pos + length]
        pos += length + 4  # skip CRC

        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if bit_depth != 8:
                raise ValueError(f"Unsupported bit depth {bit_depth}")
            if color_type not in (0, 2, 4, 6):
                raise ValueError(f"Unsupported color type {color_type}")
            if comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("Unsupported PNG compression/filter/interlace")
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("Malformed PNG (no IHDR)")

    raw = zlib.decompress(bytes(idat))

    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    elif color_type == 4:
        channels = 2
    elif color_type == 6:
        channels = 4
    bpp = channels  # 8-bit per channel

    stride = width * bpp
    expected = height * (1 + stride)
    if len(raw) < expected:
        raise ValueError("Truncated PNG data")

    prev = bytearray(stride)
    inpos = 0

    total_pixels = width * height
    step = 1
    if total_pixels > max_pixels:
        step = int(math.sqrt(total_pixels / max_pixels)) + 1

    s = 0
    cnt = 0

    for y in range(height):
        ftype = raw[inpos]
        inpos += 1
        scan = raw[inpos : inpos + stride]
        inpos += stride

        recon = bytearray(stride)
        if ftype == 0:  # None
            recon[:] = scan
        elif ftype == 1:  # Sub
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                recon[i] = (scan[i] + left) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                recon[i] = (scan[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                recon[i] = (scan[i] + ((left + up) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                recon[i] = (scan[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter {ftype}")

        if y % step == 0:
            for x in range(0, width, step):
                base = x * bpp
                if channels == 1 or channels == 2:
                    g = recon[base]
                else:
                    r = recon[base]
                    gch = recon[base + 1]
                    b = recon[base + 2]
                    g = (299 * r + 587 * gch + 114 * b) // 1000
                s += g
                cnt += 1

        prev = recon

    return s / max(1, cnt)


def _pil_scalar_edge_strength(path, max_pixels=256 * 256):
    with Image.open(path) as im:
        im = im.convert("L")  # grayscale 0..255
        w, h = im.size
        if w < 3 or h < 3:
            return 0.0

        total = w * h
        step = 1
        if total > max_pixels:
            step = int(math.sqrt(total / max_pixels)) + 1

        pix = im.tobytes()

        s = 0.0
        cnt = 0
        for y in range(1, h - 1, step):
            row_off = y * w
            for x in range(1, w - 1, step):
                center = row_off + x
                gx = pix[center + 1] - pix[center - 1]
                gy = pix[(y + 1) * w + x] - pix[(y - 1) * w + x]
                s += math.sqrt(gx * gx + gy * gy)
                cnt += 1

        if cnt == 0:
            return 0.0
        return s / cnt


def image_to_gray_mean(path, max_pixels=256 * 256):
    v = _FEATURE_CACHE.get(path)
    if v is not None:
        return v

    if not _HAVE_PIL:
        raise RuntimeError(
            "PIL is required for consistent image decoding in this environment."
        )

    v = _pil_scalar_edge_strength(path, max_pixels=max_pixels)
    _FEATURE_CACHE[path] = v
    return v




## === cell 2
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_pngs(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            low = e.name.lower()
            if low.endswith(_IMG_EXTS):
                files.append(e.path)
    files.sort()
    return files


def mean_to_bin(m, bins, lo=0.0, hi=512.0):
    if m < lo:
        m = lo
    if m > hi:
        m = hi
    rng = hi - lo
    if rng <= 1e-12:
        return 0
    b = int((m - lo) * bins / rng)
    if b < 0:
        b = 0
    if b >= bins:
        b = bins - 1
    return b


def fit_hist_nb(
    cat_files,
    dog_files,
    bins=32,
    laplace=1.0,
    precomputed_means=None,
    bin_lo=0.0,
    bin_hi=512.0,
):
    cat_hist = [0.0] * bins
    dog_hist = [0.0] * bins

    n_cat = len(cat_files)
    n_dog = len(dog_files)
    total = n_cat + n_dog
    prior_cat = n_cat / total if total else 0.5
    prior_dog = n_dog / total if total else 0.5

    get_mean = precomputed_means.get if precomputed_means is not None else None
    for p in cat_files:
        m = get_mean(p) if get_mean is not None else image_to_gray_mean(p)
        cat_hist[mean_to_bin(m, bins, lo=bin_lo, hi=bin_hi)] += 1.0
    for p in dog_files:
        m = get_mean(p) if get_mean is not None else image_to_gray_mean(p)
        dog_hist[mean_to_bin(m, bins, lo=bin_lo, hi=bin_hi)] += 1.0

    cat_den = sum(cat_hist) + laplace * bins
    dog_den = sum(dog_hist) + laplace * bins
    cat_prob = [(c + laplace) / cat_den for c in cat_hist]
    dog_prob = [(d + laplace) / dog_den for d in dog_hist]

    eps = 1e-12
    log_prior_cat = math.log(max(prior_cat, eps))
    log_prior_dog = math.log(max(prior_dog, eps))
    log_cat_prob = [math.log(max(p, eps)) for p in cat_prob]
    log_dog_prob = [math.log(max(p, eps)) for p in dog_prob]

    return {
        "bins": bins,
        "prior_cat": prior_cat,
        "prior_dog": prior_dog,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
        "log_prior_cat": log_prior_cat,
        "log_prior_dog": log_prior_dog,
        "log_cat_prob": log_cat_prob,
        "log_dog_prob": log_dog_prob,
        "bin_lo": float(bin_lo),
        "bin_hi": float(bin_hi),
    }


def predict_proba_dog(model, img_path, precomputed_means=None):
    if precomputed_means is not None and img_path in precomputed_means:
        m = precomputed_means[img_path]
    else:
        m = image_to_gray_mean(img_path)
    b = mean_to_bin(
        m, model["bins"], lo=model.get("bin_lo", 0.0), hi=model.get("bin_hi", 512.0)
    )

    log_cat = model["log_prior_cat"] + model["log_cat_prob"][b]
    log_dog = model["log_prior_dog"] + model["log_dog_prob"][b]

    mx = max(log_cat, log_dog)
    p_cat = math.exp(log_cat - mx)
    p_dog = math.exp(log_dog - mx)
    prob_dog = p_dog / (p_cat + p_dog)

    if prob_dog < 1e-6:
        prob_dog = 1e-6
    if prob_dog > 1 - 1e-6:
        prob_dog = 1 - 1e-6
    return prob_dog


def _gray_mean_worker(args):
    p, max_pixels = args
    return (p, image_to_gray_mean(p, max_pixels=max_pixels))


def precompute_gray_means(paths, max_pixels=256 * 256):
    paths = list(paths)
    if not paths:
        return {}

    nprocs = min(max(1, (os.cpu_count() or 1)), 8)
    if len(paths) < 256 or nprocs == 1:
        out = {}
        for p in paths:
            out[p] = image_to_gray_mean(p, max_pixels=max_pixels)
        return out

    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp.get_context("spawn")

    with ctx.Pool(processes=nprocs, maxtasksperchild=200) as pool:
        out = {}
        for p, m in pool.imap_unordered(
            _gray_mean_worker, ((p, max_pixels) for p in paths), chunksize=64
        ):
            out[p] = m
        return out




## === cell 3
model = None

_label_pat = re.compile(r"^(cat|dog)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def discover_labeled_train_files():
    for root in cfg.train_labeled_dir_candidates:
        if not os.path.isdir(root):
            continue
        cat_files = []
        dog_files = []
        with os.scandir(root) as it:
            for e in it:
                if not e.is_file():
                    continue
                m = _label_pat.match(e.name)
                if not m:
                    continue
                lab = m.group(1).lower()
                if lab == "cat":
                    cat_files.append(e.path)
                else:
                    dog_files.append(e.path)
        if cat_files and dog_files:
            cat_files.sort()
            dog_files.sort()
            return cat_files, dog_files, root

    cat_files = list_pngs(cfg.train_cat_dir)
    dog_files = list_pngs(cfg.train_dog_dir)
    return cat_files, dog_files, "pre_split_fallback"


cat_files_all, dog_files_all, train_source = discover_labeled_train_files()

if cfg.max_train_per_class is None:
    n = min(len(cat_files_all), len(dog_files_all))
else:
    n = min(cfg.max_train_per_class, len(cat_files_all), len(dog_files_all))

cat_files = cat_files_all[:n]
dog_files = dog_files_all[:n]

train_paths = cat_files + dog_files
train_means = precompute_gray_means(train_paths, max_pixels=cfg.max_pixels_feature)

if train_means:
    train_vals = sorted(train_means.values())
    lo_idx = int(
        max(0, min(len(train_vals) - 1, round(cfg.bin_low_q * (len(train_vals) - 1))))
    )
    hi_idx = int(
        max(0, min(len(train_vals) - 1, round(cfg.bin_high_q * (len(train_vals) - 1))))
    )
    bin_lo = float(train_vals[lo_idx])
    bin_hi = float(train_vals[hi_idx])
    if bin_hi <= bin_lo:
        bin_lo, bin_hi = float(min(train_vals)), float(
            max(train_vals)
            if max(train_vals) > min(train_vals)
            else (min(train_vals) + 1.0)
        )
else:
    bin_lo, bin_hi = 0.0, 512.0

if n == 0:
    model = None
else:
    model = fit_hist_nb(
        cat_files,
        dog_files,
        bins=cfg.bins,
        laplace=cfg.laplace,
        precomputed_means=train_means,
        bin_lo=bin_lo,
        bin_hi=bin_hi,
    )

print("Train source:", train_source)
print("Train files found (cat/dog):", len(cat_files_all), len(dog_files_all))
print("Train files used per class:", n)
print("Model ready:", model is not None)
print("PIL available:", _HAVE_PIL)
print("Binning lo/hi:", bin_lo, bin_hi)



## === cell 4
_pat = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def index_test_images(root_dir):
    out = {}
    for cur_root, _dirs, files in os.walk(root_dir):
        for fn in files:
            m = _pat.match(fn)
            if not m:
                continue
            out[int(m.group(1))] = os.path.join(cur_root, fn)
    return out


test_dir = None
for cand in cfg.test_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break

id_to_path = {}
if test_dir is not None:
    direct_hit = False
    with os.scandir(test_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            m = _pat.match(e.name)
            if m:
                id_to_path[int(m.group(1))] = e.path
                direct_hit = True
    if not direct_hit:
        id_to_path = index_test_images(test_dir)

if id_to_path:
    ids = sorted(id_to_path.keys())
    sample = pd.DataFrame({"id": ids})
else:
    sample = pd.read_csv(cfg.sample_sub_path)
    sample["id"] = pd.to_numeric(sample["id"])
    sample = sample.sort_values("id").reset_index(drop=True)

test_paths = [id_to_path.get(int(_id)) for _id in sample["id"].tolist()]
test_paths_existing = [p for p in test_paths if p is not None and os.path.exists(p)]
test_means = precompute_gray_means(
    test_paths_existing, max_pixels=cfg.max_pixels_feature
)

labels = []
missing = 0

_get = id_to_path.get
_exists = os.path.exists
_pred = predict_proba_dog

for _id in sample["id"].tolist():
    p = _get(int(_id))
    if p is None or (not _exists(p)) or model is None:
        labels.append(0.5)
        missing += 1
    else:
        try:
            labels.append(_pred(model, p, precomputed_means=test_means))
        except Exception:
            labels.append(0.5)
            missing += 1

sample["label"] = labels

os.makedirs(cfg.working_dir, exist_ok=True)
sample.to_csv(cfg.out_path, index=False)

print("Test dir used:", test_dir)
print("Indexed test images:", len(id_to_path))
print("Submission rows:", len(sample))
print("Wrote:", cfg.out_path)
print("Missing test images or model fallback count:", missing)
print(sample.head())



## === cell 5
submission = pd.read_csv(cfg.out_path)
print(submission.shape)
print(submission.head())
print("id monotonic increasing:", submission["id"].is_monotonic_increasing)
print(
    "label range:", float(submission["label"].min()), float(submission["label"].max())
)
