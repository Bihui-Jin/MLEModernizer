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

0.8924754745779371

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.0078) has done: 'I prevent the TensorFlow import from crashing by skipping it entirely and forcing the fallback dummy model, and I fix the shape‑mismatch in `make_predictions` by predicting directly on the image blocks (removing the generator + steps logic). This makes the pipeline run end‑to‑end and still produces a valid `submission.csv` while keeping the original workflow intact.'
- What this solution (achieved 0.20343) has done: 'The changes focus on eliminating the inner‑loop that copies the same prediction into every jitter slot. By broadcasting the prediction array across all jitter columns for a given model, we cut unnecessary Python loop overhead while keeping the exact same output shape and values. The rest of the logic, model training, image processing, and evaluation remain untouched, ensuring identical results.'
- What this solution (achieved 0.05981) has done: 'The changes speed up the pipeline by (1) vectorizing the expensive pixel‑wise cropping loop, (2) caching the processed training images during the simple model’s training so they aren’t recomputed when making predictions on the training set, and (3) reusing that cache in `make_predictions`. These optimizations keep exactly the same model logic and data flow, merely eliminating duplicated work while preserving identical results.'
- What this solution (achieved 0.68011) has done: 'I add a richer feature extractor that keeps the same logistic‑regression model but includes per‑channel colour histograms along with the existing mean/std features. This keeps the core model architecture unchanged while giving it more discriminative information, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.58677) has done: 'I add a balanced class weight and a higher iteration limit to the logistic‑regression model, which helps the classifier handle the imbalanced DR grades and converge better, likely raising the quadratic weighted kappa. I also let the threshold‑search run a few more refinement steps (10 instead of 5) to fine‑tune the decision thresholds, giving a modest boost toward the target score while keeping all core logic unchanged.'
- What this solution (achieved 0.61203) has done: 'I increase the logistic‑regression capacity by allowing more iterations (max_iter = 2000) and a weaker regularisation (C = 2.0). These changes keep the original feature set and workflow intact while giving the model a better chance to fit the richer colour‑histogram features, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
def crop(gray, img, percent_smaller):
    """Vectorised version of the original pixel‑wise cropping."""
    thresh = 8
    rows = np.where((gray > thresh).any(axis=1))[0]
    cols = np.where((gray > thresh).any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return img

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        return img
    return img[top:bottom, left:right]


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def claheYCC(bgr, clipLimit=5, grid=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    y = clahe.apply(y)
    y = adjust_gamma(y, 1 + np.log(110) - np.log(np.median(y)))
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def bensSimple(bgr, weight=4, gamma=15):
    return cv2.addWeighted(
        bgr, weight, cv2.GaussianBlur(bgr, (0, 0), gamma), -weight, 128
    )


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def process(bgr, model):
    green = bgr[:, :, 1]  # use green channel as greyscale
    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    if model == "normal":
        colouring_fn = bensYCC
    elif model == "weird":
        colouring_fn = bensSimple
    elif model == "clahe":
        colouring_fn = claheYCC
    else:
        raise ValueError(f"Invalid model type: {model}")

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    img = colouring_fn(resized)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




## === cell 1
def _train_simple_model():
    """Train a LogisticRegression on richer RGB features from the training set."""
    global _TRAIN_PROCESSED
    train_csv = os.path.join(INPUT_FOLDER, "train.csv")
    train_df = pd.read_csv(train_csv)
    train_df.id_code = train_df.id_code.apply(lambda x: x + ".png")
    images_dir = f"{INPUT_FOLDER}train_images/"

    X = []
    y = []
    proc_imgs = []  # cache processed RGB images for later reuse

    for idx, row in train_df.iterrows():
        filename = row.id_code
        bgr = cv2.imread(images_dir + filename)
        if bgr is None:
            bgr = np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
        proc = process(bgr, "normal")  # RGB, already resized to IMG_DIM
        proc_imgs.append(proc)  # keep for caching

        feats = _extract_raw_features(proc[np.newaxis, ...])[0]  # (30,)
        X.append(feats)
        y.append(int(row.diagnosis))

    X = np.array(X)  # (n_samples, 30)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=5000,  # increased from 2000
        C=5.0,  # increased from 2.0
        n_jobs=1,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X_scaled, y)

    _TRAIN_PROCESSED = np.stack(proc_imgs).astype(np.uint8)

    return SimpleModel(clf, scaler)




## === cell 2
def make_predictions(d_set, models):
    """
    Generates predictions for a dataset.

    Optimisation:
    - Cached processed training images avoid a second full pass over the training
      set when generating predictions for the training data.
    - Redundant jitter‑based predictions are removed (already handled in the core
      logic). The function now predicts once per model per block and broadcasts
      the result across the jitter dimension.
    """
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.shape[0]
    jitter_amounts = [
        0,
        0.01,
        0.01,
        0.01,
        0.02,
        0.02,
        0.02,
        0.05,
        0.05,
        0.05,
        0.2,
        0.2,
        0.2,
    ]
    J = len(jitter_amounts)

    ensemble_predictions = np.zeros((total, J * len(models), NUM_CLASSES))

    for m, model_name in enumerate(models):
        print(f"Loading {model_name} model for {d_set} set.")
        neural_net = load_network(model_name)

        for start in range(0, total, block_size):
            end = min(start + block_size, total)

            if d_set == "train" and _TRAIN_PROCESSED is not None:
                img_block = _TRAIN_PROCESSED[start:end]
            else:
                img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS))
                for i, filename in enumerate(df[start:end].id_code):
                    try:
                        bgr = cv2.imread(images_dir + filename)
                        img_block[i] = process(bgr, model_name)
                    except Exception:
                        img_block[i] = np.full((IMG_DIM, IMG_DIM, CHANNELS), 128.0)

            preds = neural_net.predict(img_block, verbose=0)  # (batch, NUM_CLASSES)

            col_start = m * J
            col_end = col_start + J
            ensemble_predictions[start:end, col_start:col_end, :] = np.broadcast_to(
                preds[:, None, :], (end - start, J, NUM_CLASSES)
            )

            print(f"Processed rows {start}–{end}")
            gc.collect()
    return np.median(ensemble_predictions, axis=1)




## === cell 3
train_predictions = make_predictions("train", ["normal", "weird", "clahe"])
thresholds = find_best_thresholds(train_predictions)

test_predictions = make_predictions("test", ["normal", "weird", "clahe"])
test_classes = prediction_convert_highest(test_predictions, thresholds)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/222884705.py in <cell line: 0>()
      1 # Use all three preprocessing variants to enrich the ensemble.
----> 2 train_predictions = make_predictions("train", ["normal", "weird", "clahe"])
      3 thresholds = find_best_thresholds(train_predictions)
      4 
      5 test_predictions = make_predictions("test", ["normal", "weird", "clahe"])

/tmp/ipykernel_11/2995911812.py in make_predictions(d_set, models)
     10       the result across the jitter dimension.
     11     """
---> 12     images_dir = f"{INPUT_FOLDER}{d_set}_images/"
     13     df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
     14     df.id_code = df.id_code.apply(lambda x: x + ".png")

NameError: name 'INPUT_FOLDER' is not defined
