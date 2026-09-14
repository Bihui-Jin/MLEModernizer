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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8762123503241274

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime failures by switching the deprecated `keras.preprocessing.image.ImageDataGenerator` call to the supported `tensorflow.keras.preprocessing.image.ImageDataGenerator`, and by standardizing all Keras imports to `tensorflow.keras` to avoid protobuf/standalone-keras incompatibilities that trigger the `MessageFactory/GetPrototype` error. I keep the model, preprocessing, jitter-TTA prediction loop, and thresholding logic identical. I also make the input path resolution slightly more robust (fall back to `/kaggle/input/...` if needed) without changing any semantics, ensuring a `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'The crash happens before any training/inference because TensorFlow’s protobuf dependency is incompatible in this environment, triggering `MessageFactory.GetPrototype` during `import tensorflow as tf`. To make the notebook run end-to-end and still preserve the core inference logic (same preprocessing, DenseNet121, TTA/jitter loop, and threshold conversion), I switch the backend to PyTorch using torchvision’s DenseNet-121 and load the provided `.h5` weights if available (if not found, it still generate a valid submission but score poorly). I also make the input-path resolution robust for both `/kaggle/input/...` and the provided `/kaggle/data/...` layout, and ensure the submission CSV matches `id_code,diagnosis` exactly. These changes are strictly to unblock execution and restore non-zero scoring by enabling the intended pretrained-weight inference path.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running a randomly initialized PyTorch DenseNet (your code explicitly warns it can’t load the intended Keras `.h5` weights), so the smallest way to move toward the target is to keep your exact preprocessing/TTA/thresholding, but switch back to TensorFlow/Keras inference and actually load `normal_end_2019.h5`. To avoid the protobuf crash that previously blocked TF import, this patch forces the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle workaround) and standardizes imports to `tensorflow.keras`. Everything else (image processing, jitter loop, median aggregation, thresholds, and submission format/path) is preserved so evaluation semantics stay the same while restoring the intended pretrained model behavior.'
- What this solution (achieved 0.0) has done: 'The crash happens at `import tensorflow as tf` due to an old protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`) in this runtime, so the core fix is to avoid importing TensorFlow entirely. To keep the pipeline running end-to-end and produce a valid `submission.csv`, I switch the model implementation to PyTorch’s `torchvision.models.densenet121` with the same “DenseNet121 backbone → GAP → Dropout → Dense(5) → Sigmoid” head and keep your preprocessing, jitter-TTA loop, median aggregation, and threshold-to-class conversion unchanged. I also make the weight loader robust: it try to load a compatible PyTorch `.pth` if present, otherwise it run with random weights (score be low but non-zero and valid). Finally, I ensure the prediction call uses a torch no-grad forward pass and always writes `id_code,diagnosis` to `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I remove the hard dependency on TensorFlow (which is crashing at import due to a protobuf incompatibility in this environment) and implement the same DenseNet121→GAP→Dropout→Dense(5)→Sigmoid model in PyTorch so the notebook runs end-to-end. I also make the weight-file discovery more robust by searching the provided dataset trees, and load weights if a compatible PyTorch checkpoint is found; otherwise it still generate a valid submission (but likely with low score). Finally, I keep your exact preprocessing, jitter-TTA loop, median aggregation, and threshold-to-class conversion unchanged, and ensure `submission.csv` is always written with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model running with random weights (the code only loads a PyTorch `.pth/.pt/.bin`, but the provided competition weight file is typically the Keras `normal_end_2019.h5`). To move toward the target score with minimal semantic change, I keep your exact preprocessing, jitter-TTA loop, median aggregation, and thresholding, but switch inference back to TensorFlow/Keras and actually load the `.h5` weights when present. To avoid the protobuf/TensorFlow `MessageFactory.GetPrototype` crash you previously saw, I force the pure-Python protobuf implementation before importing TensorFlow and standardize all Keras imports to `tf.keras`. The script still writes a valid `submission.csv` with `id_code,diagnosis` and fall back to the existing PyTorch path only if TensorFlow import or `.h5` loading fails.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the fallback path running a randomly initialized PyTorch DenseNet because the intended pretrained `.h5` weights are not being loaded (or TensorFlow import still fails). To move the score upward toward the target with minimal semantic change, I keep your exact preprocessing, jitter-TTA loop, median aggregation, and threshold conversion, but make the Keras weight-file discovery more robust (search for common alternate filenames) and force TensorFlow/Keras to be attempted first and deterministically. I also add a small safety fix so Keras inference uses the same `[0,1]` input but explicitly casts to `float32` to avoid dtype surprises, without changing the actual values. The script still always writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the fallback PyTorch path running with random weights (because the intended pretrained Keras `.h5` did not load), so the smallest change to move toward the target is to reliably load and use the `.h5` model. I keep your preprocessing, jitter-TTA loop, aggregation, and threshold-to-class conversion identical, but (1) make weight discovery also search for any `.h5` under inputs and (2) force TensorFlow to run in CPU mode (avoids common GPU/protobuf/runtime issues) while keeping the same Keras model build/load. Finally, I add a hard check: if Keras weights exist but Keras model still can’t be created, we fail loudly instead of silently producing a low-scoring random-weight submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script silently falling back to random-weight PyTorch inference (or loading Keras weights into a mismatched head), so the smallest score-improving change is to reliably use the pretrained Keras `.h5` model when it exists. I keep your preprocessing, jitter-TTA loop, median aggregation, and threshold conversion unchanged, but adjust the Keras model construction to match the standard DenseNet121 head (no extra ReLU after the backbone) and ensure inference runs with `training=False` (so Dropout is disabled). I also add a lightweight sanity check that the loaded Keras model produces non-constant outputs on a tiny batch, to catch bad weight loads early instead of producing a near-0.0 submission. The script still always write a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing essentially random predictions (e.g., failing to load the intended pretrained Keras `.h5` and/or falling back), so the smallest score-improving change is to ensure we actually use the provided pretrained Keras model when available and don’t silently proceed with an uninitialized head. I keep your exact preprocessing, jitter-TTA loop, median aggregation, and threshold-to-class conversion unchanged, but make the Keras weight discovery/load stricter and more robust, and ensure the DenseNet121 head matches the saved weights exactly (removing the extra post-backbone ReLU in the Torch fallback only, and keeping dropout disabled at inference). Finally, I add a hard check that the submission row order matches `test.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script producing essentially random predictions due to not finding/using the intended pretrained weights, so the smallest change to move toward the target is to reliably locate and load the provided `normal_end_2019.h5` (or any `.h5`) from the dataset tree. I keep your preprocessing, jitter-TTA loop, median aggregation, and thresholding exactly the same, but tighten weight discovery to search all `.h5` files first (not just “hinted” names) and prefer the exact filename when present. I also add a lightweight check that we are not silently using uninitialized Torch weights when an `.h5` is available (fail fast only in that case), because that’s the direct cause of a 0.0 submission. The pipeline still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the run is still not using the intended pretrained model (most commonly: it doesn’t find the `.h5` weights and silently falls back to random PyTorch weights, yielding near-random labels). To move the score upward toward the 0.876 target with minimal semantic change, I make weight discovery stricter and more reliable by (1) preferring `normal_end_2019.h5` explicitly, and (2) searching for `.h5` only within the competition dataset tree first before scanning all of `/kaggle/input` (which can accidentally pick an unrelated `.h5`). I also add a small, fast “weights sanity” check that ensures we don’t proceed to write a submission if we ended up with a random/untrained model while a valid `.h5` is present. Core preprocessing, TTA/jitter loop, median aggregation, thresholding, and submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force TF CPU

import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5


def resolve_input_folder():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
            return p
    for base in ["/kaggle/input", "/kaggle/data", "../input"]:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if "train.csv" in files and root.endswith(
                    "aptos2019-blindness-detection"
                ):
                    return root + "/"
    return "/kaggle/input/aptos2019-blindness-detection/"


INPUT_FOLDER = resolve_input_folder()


def _walk_find_file(base, predicate):
    if not os.path.exists(base):
        return None
    for root, _, files in os.walk(base):
        for f in files:
            if predicate(root, f):
                return os.path.join(root, f)
    return None


def find_weight_file(filename):
    candidates = [
        os.path.join("../input", "densenetmulti", filename),
        os.path.join("/kaggle/input", "densenetmulti", filename),
        os.path.join("/kaggle/data", "densenetmulti", filename),
        os.path.join(INPUT_FOLDER, filename),
        os.path.join(INPUT_FOLDER, "weights", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    in_comp = _walk_find_file(INPUT_FOLDER, lambda root, f: f == filename)
    if in_comp is not None:
        return in_comp

    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        p = _walk_find_file(base, lambda root, f: f == filename)
        if p is not None:
            return p
    return None


def find_any_h5_by_hint(
    hints=("normal_end_2019", "densenet", "dense", "aptos", "blindness", "model")
):
    def _find_in_base(base):
        return _walk_find_file(
            base,
            lambda root, f: f.lower().endswith(".h5")
            and any(h in f.lower() for h in hints),
        )

    p = _find_in_base(INPUT_FOLDER)
    if p is not None:
        return p

    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        p = _find_in_base(base)
        if p is not None:
            return p
    return None


def find_any_h5_anywhere():
    p = _walk_find_file(INPUT_FOLDER, lambda root, f: f.lower().endswith(".h5"))
    if p is not None:
        return p
    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        p = _walk_find_file(base, lambda root, f: f.lower().endswith(".h5"))
        if p is not None:
            return p
    return None


KERAS_WEIGHTS = (
    find_weight_file("normal_end_2019.h5")
    or find_weight_file("Normal_end_2019.h5")
    or find_weight_file("normal_end.h5")
    or find_weight_file("densenet121.h5")
    or find_weight_file("model.h5")
    or find_any_h5_by_hint()
    or find_any_h5_anywhere()
)

TORCH_WEIGHTS = (
    find_weight_file("normal_end_2019.pth")
    or find_weight_file("normal_end_2019.pt")
    or find_weight_file("normal_end_2019.bin")
    or find_weight_file("densenet121.pth")
)

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Resolved TORCH_WEIGHTS:", TORCH_WEIGHTS)
print("Resolved KERAS_WEIGHTS:", KERAS_WEIGHTS)




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while top < bottom and middleCol[top] == 0:
        top += 1
    while bottom > top and middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while left < right and middleRow[left] == 0:
        left += 1
    while right > left and middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100 or bottom <= top or right <= left:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=20):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def reflectAndSquareUp(img):
    height = img.shape[0]
    width = img.shape[1]

    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        h1 = int((width - height) / 2)
        h2 = h1 + height
        new_img[h1:h2, :] = img

        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(width - h2):
            new_img[h2 + i] = img[height - i - 1]
        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def processBenNormal(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        if height <= 0 or width <= 0:
            test_crop = bgr
        else:
            h = int((cropped.shape[0] - height) / 2)
            w = int((cropped.shape[1] - width) / 2)
            test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    reflected = reflectAndSquareUp(test_crop)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(
        resized, 1 + np.log(90) - np.log(max(1.0, np.median(resized)))
    )
    bens = benYCC(equalised, weight=3, gamma=20)

    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def _apply_jitter_batch(img_block_uint8, jitter):
    """
    img_block_uint8: (N, H, W, 3) uint8 RGB in [0,255]
    returns float32 in [0,1] after augment
    """
    x = img_block_uint8.astype(np.float32)

    if jitter <= 0.01:
        return x / 255.0

    N, H, W, C = x.shape

    do_h = np.random.rand(N) < 0.5
    x[do_h] = x[do_h, :, ::-1, :]
    do_v = np.random.rand(N) < 0.5
    x[do_v] = x[do_v, ::-1, :, :]

    rot_range = int(600 * jitter)
    rot_range = max(0, rot_range)
    if rot_range > 0:
        for i in range(N):
            angle = np.random.uniform(-rot_range, rot_range)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            x[i] = cv2.warpAffine(
                x[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

    zmin = max(0.8, 1 - 5 * jitter)
    zmax = 1.0
    for i in range(N):
        z = np.random.uniform(zmin, zmax)
        if z < 0.999:
            nh, nw = int(H * z), int(W * z)
            y0 = (H - nh) // 2
            x0 = (W - nw) // 2
            crop = x[i, y0 : y0 + nh, x0 : x0 + nw, :]
            x[i] = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    bmin = 1 - jitter / 3
    bmax = 1 + jitter / 3
    b = np.random.uniform(bmin, bmax, size=(N, 1, 1, 1)).astype(np.float32)
    x = np.clip(x * b, 0, 255)

    cs = int(30 * jitter)
    if cs > 0:
        shift = np.random.uniform(-cs, cs, size=(N, 1, 1, C)).astype(np.float32)
        x = np.clip(x + shift, 0, 255)

    return x / 255.0




## === cell 3
def test_datagen_plot(processing_function, jitter=0.3):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i, :, :, :] = processing_function(bgr)

    x = _apply_jitter_batch(img_block, jitter)
    figure = plt.figure(figsize=(10, 10))
    for j in range(16):
        ax = figure.add_subplot(4, 4, j + 1)
        plt.imshow(x[j])
        ax.axis("off")
    plt.show()




## === cell 4
def _try_create_keras_model(weights_path):
    if weights_path is None or (not os.path.exists(weights_path)):
        return None, "KERAS_WEIGHTS not found"

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models as kmodels
        from tensorflow.keras.applications import DenseNet121

        tf.get_logger().setLevel("ERROR")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    except Exception as e:
        return None, f"TensorFlow import failed: {repr(e)}"

    try:
        inp = layers.Input(shape=(IMG_DIM, IMG_DIM, CHANNELS))
        base = DenseNet121(include_top=False, weights=None, input_tensor=inp)

        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
        model = kmodels.Model(inputs=inp, outputs=out)

        model.load_weights(weights_path)

        dummy0 = np.zeros((1, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
        dummy1 = np.ones((1, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32) * 0.5
        y0 = model(dummy0, training=False).numpy()
        y1 = model(dummy1, training=False).numpy()
        if (not np.isfinite(y0).all()) or (not np.isfinite(y1).all()):
            return None, "Keras outputs contain non-finite values."
        if np.std(y0) < 1e-6 or np.std(y1) < 1e-6:
            return None, "Keras outputs look degenerate (std too small)."
        if np.max(np.abs(y0 - y1)) < 1e-6:
            return (
                None,
                "Keras outputs are (nearly) input-invariant; likely wrong weights.",
            )

        return model, f"Loaded Keras weights: {weights_path}"
    except Exception as e:
        return None, f"Keras model build/load failed: {repr(e)}"


import torch
import torch.nn as nn
import torchvision.models as models

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseNet121Sigmoid(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        self.base = models.densenet121(weights=None)
        n_feats = self.base.classifier.in_features
        self.base.classifier = nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(n_feats, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base.features(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class TorchModelWrapper:
    def __init__(self, torch_model, device):
        self.model = torch_model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, xb, verbose=0):
        with torch.no_grad():
            xt = torch.from_numpy(xb).permute(0, 3, 1, 2).contiguous()
            xt = xt.to(self.device, dtype=torch.float32)
            out = self.model(xt).detach().cpu().numpy()
        return out


def _load_torch_weights(model, weights_path):
    if weights_path is None or (not os.path.exists(weights_path)):
        print(
            "WARNING: No compatible PyTorch weights found. Proceeding with random weights. "
            "Submission will be valid but score may be poor."
        )
        return False

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]

    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Loaded torch weights:", weights_path)
    if missing:
        print("Missing keys (non-fatal):", len(missing))
    if unexpected:
        print("Unexpected keys (non-fatal):", len(unexpected))
    return True


class KerasLikeWrapper:
    def __init__(self, keras_model):
        self.model = keras_model

    def predict(self, xb, verbose=0):
        import tensorflow as tf

        x = xb.astype(np.float32, copy=False)
        y = self.model(tf.convert_to_tensor(x), training=False)
        return y.numpy()


def create_model(torch_weights_path, keras_weights_path):
    keras_model, msg = _try_create_keras_model(keras_weights_path)
    print(msg)

    if (
        keras_weights_path is not None
        and os.path.exists(keras_weights_path)
        and keras_model is None
    ):
        raise RuntimeError(
            "Found Keras weights but failed to create/load a non-degenerate Keras model.\n"
            f"KERAS_WEIGHTS={keras_weights_path}\n"
            f"Error: {msg}"
        )

    if keras_model is not None:
        return KerasLikeWrapper(keras_model)

    torch_model = DenseNet121Sigmoid(num_classes=NUM_CLASSES, dropout=0.5)
    _load_torch_weights(torch_model, torch_weights_path)
    return TorchModelWrapper(torch_model, DEVICE)




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i, :, :, :] = processing_function(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            xb = _apply_jitter_batch(img_block, jit)

            out_list = []
            for bs in range(0, xb.shape[0], BATCH_SIZE):
                be = min(bs + BATCH_SIZE, xb.shape[0])
                pred = model.predict(xb[bs:be], verbose=0)
                out_list.append(pred.astype(np.float32))
            pred = np.concatenate(out_list, axis=0)
            prediction_jitters[:, i] = pred

            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)

    y_val = thresholded.sum(axis=1) - 1
    y_val = np.clip(y_val, 0, 4).astype(np.int32)
    return y_val




## === cell 7
def label_convert(preds):
    y_val = preds > 0.5
    return (y_val.astype(int).sum(axis=1) - 1).astype(np.int32)


def label_convert_two_stage(stage_1_preds, stage_2_preds):
    thresh_1 = np.zeros((stage_1_preds.shape[0], 3))
    thresh_2 = np.zeros((stage_2_preds.shape[0], 3))

    for i in range(3):
        thresh_1[:, i] = stage_1_preds[:, i] > 0.5
        thresh_2[:, i] = stage_2_preds[:, i] > 0.5

    y_val = thresh_1.astype(int).sum(axis=1) - 1
    y_val_2 = thresh_2.astype(int).sum(axis=1) + 1

    for i in range(stage_1_preds.shape[0]):
        if y_val[i] == 2:
            y_val[i] = y_val_2[i]
    return np.clip(y_val, 0, 4).astype(np.int32)


def label_convert_two_stage_top(stage_1_preds, stage_2_preds):
    thresh_1 = np.zeros((stage_1_preds.shape[0], 3))
    thresh_2 = np.zeros((stage_2_preds.shape[0], 3))

    for i in range(3):
        thresh_1[:, i] = stage_1_preds[:, i] > 0.5
        thresh_2[:, i] = stage_2_preds[:, i] > 0.5

    y_val = np.zeros(stage_1_preds.shape[0])
    y_val_2 = np.zeros(stage_1_preds.shape[0])
    for i in range(stage_1_preds.shape[0]):
        for j in range(2, -1, -1):
            if thresh_1[i, j]:
                y_val[i] = j
                break
        for j in range(2, -1, -1):
            if thresh_2[i, j]:
                y_val_2[i] = j + 2
                break

    for i in range(stage_1_preds.shape[0]):
        if y_val[i] == 2:
            y_val[i] = y_val_2[i]
    return np.clip(y_val, 0, 4).astype(np.int32)




## === cell 8
model = create_model(TORCH_WEIGHTS, KERAS_WEIGHTS)

preds = make_predictions("test", processBenNormal, model, jitters=5)

thresholds = [0.5, 0.765625, 0.8125, 0.6875, 0.625]
test_classes = prediction_convert(preds, thresholds)
print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
assert len(test_df) == len(test_classes), "Prediction length mismatch vs test.csv"
test_df["diagnosis"] = test_classes.astype(int)

submission_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", test_df.shape)
print(test_df.head())
