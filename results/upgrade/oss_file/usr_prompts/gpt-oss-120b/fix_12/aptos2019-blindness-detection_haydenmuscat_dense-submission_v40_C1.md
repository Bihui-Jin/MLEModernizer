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

0.8912375928845857

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the problematic imports with TensorFlow‑Keras ones, remove unused libraries that cause the protobuf error, add a safe check for the pretrained weight file (skip loading if it is missing), and use the modern `model.predict` API. These fixes stop the runtime crashes and guarantee that a `submission.csv` file is written, while keeping the original model architecture and processing logic unchanged.'
- What this solution (achieved 0.0) has done: 'I add a protobuf‑compatibility fix by setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable **before** importing TensorFlow. This prevents the `MessageFactory` AttributeError, allowing the model to load the pretrained weights and generate predictions, after which the script write a proper `submission.csv`. No other logic is altered.'
- What this solution (achieved 0.0) has done: 'I fix the runtime errors by converting the label column to strings before using Keras’ `flow_from_dataframe`, which expects categorical string labels when `class_mode="categorical"`. This resolves the `TypeError` in cell 4 and allows the model to be created, trained, and used for predictions, enabling a valid `submission.csv` to be written. No other logic is altered, keeping the original architecture and training approach intact.'
- What this solution (achieved 0.00584) has done: 'I fixed the datatype mismatch that caused the `UFuncTypeError` by normalising images to float32 inside the preprocessing function and removing the conflicting `rescale` parameter from the data generators. I also adjusted the prediction handling to use `argmax` (the natural multi‑class decision) instead of the handcrafted thresholds, which simplifies post‑processing and generally yields a higher quadratic weighted kappa. These changes keep the original model architecture and training loop intact while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.00815) has done: 'Implemented fixes to resolve runtime errors and improve the model’s classification logic:
- Added safe handling for median‑zero cases in the image preprocessing to prevent OpenCV LUT failures.  
- Updated the model architecture to use a softmax output with categorical_crossentropy loss, matching the multi‑class nature of the problem.  
- Included a quick validation κ calculation after training to verify the model’s performance.  
- Minor clean‑ups keep the original workflow intact while markedly improving the score potential.'

# 9. Code solution

## === cell 0
def crop(gray, img, percent_smaller):
    thresh = 8
    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        return img
    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def reflectAndSquareUp(img):
    h, w = img.shape[:2]
    if h > w:
        offset = int((h - w) / 2)
        return img[offset : offset + w]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((w, w, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((w, w), np.uint8)

        h1 = int((w - h) / 2)
        h2 = h1 + h
        new_img[h1:h2, :] = img
        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(w - h2):
            new_img[h2 + i] = img[h - i - 1]
        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img
    dim = img.shape[0]
    half = dim // 2
    mask = np.zeros((dim, dim), np.uint8)
    mask = cv2.circle(mask, (half, half), half, 1, thickness=-1)
    return cv2.bitwise_and(img, img, mask=mask)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def processBenNormal(bgr):
    """
    Robust preprocessing for a BGR image.
    Falls back to a simple resize+normalize if any step fails
    (e.g., unexpected channel count or LUT errors).
    """
    try:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Invalid image shape")
        green = bgr[:, :, 1]
        cropped = crop(green, bgr, 0.02)
        squared = reflectAndSquareUp(cropped)
        resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
        circled = circleMask(resized)
        med = np.median(circled)
        if med == 0:
            gamma_val = 1.0
        else:
            gamma_val = 1 + np.log(90) - np.log(med)
        equalised = adjust_gamma(circled, gamma_val)
        final = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return final.astype(np.float32) / 255.0
    except Exception:
        resized = cv2.resize(bgr, (IMG_DIM, IMG_DIM))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0


pre_process_function = processBenNormal




## === cell 1
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        horizontal_flip=jitter > 0.01,
        vertical_flip=jitter > 0.01,
        zoom_range=[max(0.7, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="constant",
        cval=128.0,
        channel_shift_range=30 * jitter,
    )
    return datagen




## === cell 2
def create_model(weights_path):
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    else:
        print(
            f"Warning: pretrained weights not found at {weights_path}. Using ImageNet weights only."
        )

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 3
if os.path.isfile(NORMAL_WEIGHTS):
    model = create_model(NORMAL_WEIGHTS)
else:
    print(
        "Pretrained weights not found – skipping fine‑tuning to stay within time limits."
    )
    model = create_model(NORMAL_WEIGHTS)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3040560437.py in <cell line: 0>()
      1 # Load or initialise the model – skip expensive training if pretrained weights are absent.
----> 2 if os.path.isfile(NORMAL_WEIGHTS):
      3     model = create_model(NORMAL_WEIGHTS)
      4 else:
      5     print(

NameError: name 'os' is not defined

## === cell 4
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["id_code"] = df["id_code"].astype(str) + ".png"

    block_size = 1024  # larger blocks reduce Python loop overhead
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        batch_df = df.iloc[start:end]

        img_block = np.empty(
            (end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )
        for i, filename in enumerate(batch_df["id_code"]):
            path = os.path.join(images_dir, filename)
            bgr = cv2.imread(path)
            if bgr is None:
                img_block[i] = np.full(
                    (IMG_DIM, IMG_DIM, CHANNELS), 0.5, dtype=np.float32
                )
            else:
                img_block[i] = processing_function(bgr)

        pred_jitters = np.zeros((end - start, jitters, NUM_CLASSES), dtype=np.float32)
        jitter_val = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jitter_val).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            pred_jitters[:, j, :] = model.predict(
                datagen, steps=len(datagen), verbose=0
            )
            jitter_val += 0.02

        predictions[start:end] = np.median(pred_jitters, axis=1)
        print(f"{start} - {end} finished")

    return predictions




## === cell 5
def label_convert(preds):
    """
    Convert probability vectors to integer class labels by
    computing the expected rating and rounding to the nearest integer.
    This aligns better with quadratic weighted kappa than a simple argmax.
    """
    expected = np.dot(preds, _CLASS_INDICES)
    return np.rint(expected).astype(int)




## === cell 6
test_predictions = make_predictions("test", processBenNormal, model, jitters=5)

test_classes = label_convert(test_predictions)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2626805877.py in <cell line: 0>()
----> 1 test_predictions = make_predictions("test", processBenNormal, model, jitters=5)
      2 
      3 test_classes = label_convert(test_predictions)
      4 
      5 test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))

NameError: name 'model' is not defined
