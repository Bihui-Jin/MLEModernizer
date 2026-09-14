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

0.8723692104941825

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the Jupyter-only `%matplotlib inline` magic and fix the import stack so `Input` (and the rest of Keras symbols) are always defined. I also add a safe fallback for model weights since the referenced `../input/aptos2019-resnet50/resNet50.h5` dataset is not present in your provided input tree; the script proceed with randomly initialized weights rather than crashing, ensuring a valid `submission.csv` is always produced. Finally, I make prediction robust to missing/corrupt images and guarantee `ans` has exactly the same length as the submission template so `submit['diagnosis']=ans` cannot fail.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before TensorFlow is imported, which is a common Kaggle environment incompatibility. I also move the env-var setup to the very top so it takes effect, and add a small compatibility fallback to pin protobuf parsing behavior if needed. These changes are runtime/stability fixes only (model, preprocessing, and prediction logic remain the same), and the script still write a valid `submission.csv`. With TensorFlow successfully importing, your model at least run end-to-end and produce non-empty predictions (improving from the current 0.0 which is effectively “broken run”).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before *any* TensorFlow-related import and by forcing the pure-Python protobuf implementation early. I also add a safe fallback that, if TensorFlow still can’t be imported in this environment, produces a valid `submission.csv` using the sample submission template (so you don’t get a broken 0.0 due to no file). These changes are runtime/stability only and keep your model definition, preprocessing, and prediction logic unchanged when TensorFlow loads successfully. The script always write a correctly formatted `.csv` submission with `id_code,diagnosis`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd


def _ensure_protobuf_env_and_reexec():
    desired = {
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION": "3",
        "PYTHONHASHSEED": "42",
    }
    need_reexec = False
    for k, v in desired.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
            need_reexec = True

    if need_reexec and os.environ.get("_KAGGLE_REEXEC_DONE", "0") != "1":
        os.environ["_KAGGLE_REEXEC_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_protobuf_env_and_reexec()

np.random.seed(42)

BASE_INPUT = "/kaggle/input"
BASE_DATA_FALLBACK = "/kaggle/data"

print(
    "Input dirs:",
    os.listdir(BASE_INPUT) if os.path.isdir(BASE_INPUT) else "BASE_INPUT not found",
)



## === cell 1
from PIL import Image  # noqa: F401
import cv2
from tqdm import tqdm

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical  # noqa: F401
    from sklearn.model_selection import train_test_split  # noqa: F401
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: F401
    from tensorflow.keras.layers import (
        Input,
        Dense,
        PReLU,
        Dropout,
        GlobalAveragePooling2D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import (  # noqa: F401
        LearningRateScheduler,
        ModelCheckpoint,
        TensorBoard,
        EarlyStopping,
        ReduceLROnPlateau,
        Callback,
    )
    from tensorflow.keras.optimizers import SGD, Adam  # noqa: F401
    from tensorflow.keras.applications.xception import Xception  # noqa: F401
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.applications.inception_v3 import InceptionV3  # noqa: F401
    from sklearn.metrics import (  # noqa: F401
        cohen_kappa_score,
        accuracy_score,
        classification_report,
    )

    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("WARNING: TensorFlow import failed; will write a fallback submission.csv.")
    print("TF import error:", TF_IMPORT_ERROR)




## === cell 2
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    else:
        return img




## === cell 3
IMG_SIZE = 224
batch_size = 32
epochs = 10




## === cell 4
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
    return image




## === cell 5
def getResNet50(input_shape=(224, 224, 3), classes=5, weights=None):
    input_layer = Input(shape=input_shape)
    resnet_backbone = ResNet50(include_top=False, weights=weights)(input_layer)
    x = GlobalAveragePooling2D(name="avg_pool")(resnet_backbone)
    x = Dense(1024, name="fc1")(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax", name="output")(x)
    model = Model(input_layer, x)
    return model




## === cell 6
def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


comp_dir = _find_first_existing(
    [
        os.path.join(BASE_INPUT, "aptos2019-blindness-detection"),
        os.path.join(
            BASE_INPUT, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
        ),
        os.path.join(BASE_DATA_FALLBACK, "aptos2019-blindness-detection"),
        os.path.join(
            BASE_DATA_FALLBACK,
            "aptos2019-blindness-detection",
            "aptos2019-blindness-detection",
        ),
    ]
)
if comp_dir is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory in /kaggle/input or /kaggle/data"
    )

test_images_dir = _find_first_existing(
    [
        os.path.join(comp_dir, "test_images"),
        os.path.join(comp_dir, "aptos2019-blindness-detection", "test_images"),
    ]
)
sample_sub_path = _find_first_existing(
    [
        os.path.join(comp_dir, "sample_submission.csv"),
        os.path.join(
            comp_dir, "aptos2019-blindness-detection", "sample_submission.csv"
        ),
    ]
)

print("Competition dir:", comp_dir)
print(
    "Test images dir:",
    test_images_dir,
    "exists:",
    os.path.isdir(test_images_dir) if test_images_dir else False,
)
print(
    "Sample submission:",
    sample_sub_path,
    "exists:",
    os.path.isfile(sample_sub_path) if sample_sub_path else False,
)

if sample_sub_path is None or not os.path.isfile(sample_sub_path):
    raise FileNotFoundError("sample_submission.csv not found")

submit = pd.read_csv(sample_sub_path)



## === cell 7
if not TF_AVAILABLE or test_images_dir is None or not os.path.isdir(test_images_dir):
    submit["diagnosis"] = 0
    submit.to_csv("submission.csv", index=False)
    print("Wrote fallback submission.csv with shape:", submit.shape)
    print(submit.head())
else:
    resNet50 = getResNet50(weights=None)

    weights_path = _find_first_existing(
        [
            os.path.join(BASE_INPUT, "aptos2019-resnet50", "resNet50.h5"),
            os.path.join(BASE_DATA_FALLBACK, "aptos2019-resnet50", "resNet50.h5"),
        ]
    )
    if weights_path is not None and os.path.exists(weights_path):
        resNet50.load_weights(weights_path)
        print("Loaded weights:", weights_path)
    else:
        print("WARNING: weights not found at expected path(s).")
        print(
            "Proceeding with randomly initialized weights (submission will be valid but score may be low)."
        )

    ans = []
    for name in tqdm(submit["id_code"].astype(str).tolist(), total=len(submit)):
        img_path = os.path.join(test_images_dir, name + ".png")
        img = preprocess_image(img_path)
        if img is None:
            ans.append(0)
            continue
        img = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(img, axis=0)
        pre = resNet50.predict(x, verbose=0)
        ans.append(int(np.argmax(pre, axis=1)[0]))

    if len(ans) != len(submit):
        raise RuntimeError(
            f"Prediction length mismatch: len(ans)={len(ans)} vs len(submit)={len(submit)}"
        )

    submit["diagnosis"] = ans
    submit.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submit.shape)
    print(submit.head())
