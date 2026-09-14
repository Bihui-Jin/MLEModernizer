# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import glob
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



## === cell 1
DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

train_images_folder = f"{DATA_DIR}/train_images"
test_images_folder = f"{DATA_DIR}/test_images"

csv_train = pd.read_csv(TRAIN_CSV)
csv_test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print(csv_train.shape, csv_test.shape, sample_submission.shape)
print(sample_submission.columns.tolist())



## === cell 2
columns_in_test = [c for c in csv_test.columns if c != "prediction_id"]
columns_in_test.append("cancer")  # keep target

data_train = csv_train[columns_in_test].copy()

bad_ids = [1942326353]
data_train = data_train.drop(data_train[data_train["image_id"].isin(bad_ids)].index)

data_train = data_train.dropna(subset=["age"]).copy()
data_train["age"] = data_train["age"] / data_train["age"].max()

data_test = csv_test.copy()
data_test["age"] = data_test["age"].fillna(data_test["age"].mean())
data_test["age"] = data_test["age"] / data_train["age"].max()

print(data_train.shape, data_test.shape)



## === cell 3
cancer_train = data_train.loc[data_train.cancer == 1].copy()
nocancer_train = data_train.loc[data_train.cancer == 0].copy()

print("cancer:", len(cancer_train), "no cancer:", len(nocancer_train))



## === cell 4
import cv2
import pydicom


def _safe_pixel_array(ds):
    """
    Return pixel array if possible. If compressed transfer syntax cannot be decoded
    in this environment, return None so we can fall back safely.
    """
    try:
        arr = ds.pixel_array
        return arr
    except Exception:
        return None


def cut_empty_space(im, T=100, cutedge=10):
    impx = im[cutedge:-cutedge, cutedge:-cutedge]
    if impx.size == 0:
        return im

    _, bw = cv2.threshold(impx.astype(np.float32), float(T), 255.0, cv2.THRESH_BINARY)
    bw_u8 = np.clip(bw, 0, 255).astype(np.uint8)

    contours, _ = cv2.findContours(bw_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return impx

    boundary = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(boundary)
    if w <= 1 or h <= 1:
        return impx
    return impx[y : y + h, x : x + w]


def process_im(im, image_size=512):
    out = im.astype(np.float32)

    minval = float(np.min(out))
    maxval = float(np.max(out))
    if maxval <= minval:
        out = np.zeros((image_size, image_size), dtype=np.float32)
        return out

    threshold = maxval / 5.0
    out = cut_empty_space(out, T=threshold)

    minval = float(np.min(out))
    maxval = float(np.max(out))
    if maxval > minval:
        out = (out - minval) / (maxval - minval)
    else:
        out = np.zeros_like(out, dtype=np.float32)

    out = cv2.resize(out, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return out


def dcm_to_png(dcm_path, png_path, image_size=512):
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = _safe_pixel_array(ds)
    if arr is None:
        out = np.zeros((image_size, image_size), dtype=np.uint8)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, out)
        return False

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    out = process_im(arr, image_size=image_size)
    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    cv2.imwrite(png_path, out_u8)
    return True




## === cell 5
image_size = 512
workdir = "/kaggle/working/test/"
pngfolder = os.path.join(workdir, f"processed_{image_size}")
os.makedirs(pngfolder, exist_ok=True)

test_paths = [
    (
        os.path.join(test_images_folder, str(pid), f"{iid}.dcm"),
        os.path.join(pngfolder, str(pid), f"{iid}.png"),
    )
    for pid, iid in zip(data_test["patient_id"].values, data_test["image_id"].values)
]

start = time.time()
ok = 0
for dcm_path, png_path in test_paths:
    if os.path.exists(png_path):
        ok += 1
        continue
    success = dcm_to_png(dcm_path, png_path, image_size=image_size)
    ok += int(success)

print(
    f"Test PNGs prepared in {time.time()-start:.1f}s; successfully decoded {ok}/{len(test_paths)}"
)
gc.collect()




## === cell 6
def onehot(df, encode):
    temp = pd.get_dummies(df[encode])
    df = df.drop([encode], axis=1)
    df = pd.concat([df, temp], axis=1)
    return df


irrelevant_train = ["machine_id", "site_id"]
data_train_relevant = data_train.drop(irrelevant_train, axis=1).copy()

irrelevant_test = ["machine_id", "site_id", "prediction_id"]
data_test_relevant = data_test.drop(irrelevant_test, axis=1).copy()

onehotcols = ["laterality", "view"]
for col in onehotcols:
    data_test_relevant = onehot(data_test_relevant, col)
    data_train_relevant = onehot(data_train_relevant, col)

data_test_relevant["implant"] = data_test_relevant["implant"].fillna(0).astype("uint8")
data_train_relevant["implant"] = (
    data_train_relevant["implant"].fillna(0).astype("uint8")
)

for c in data_train_relevant.columns:
    if c not in data_test_relevant.columns and c != "cancer":
        data_test_relevant[c] = 0
        data_test_relevant[c] = data_test_relevant[c].astype("uint8")

tag_cols = ["age", "implant", "L", "R", "AT", "CC", "LM", "LMO", "ML", "MLO"]
for c in tag_cols:
    if c not in data_test_relevant.columns:
        data_test_relevant[c] = 0

data_test_relevant["file"] = data_test_relevant.apply(
    lambda x: os.path.join(
        pngfolder, str(int(x["patient_id"])), f"{int(x['image_id'])}.png"
    ),
    axis=1,
)

print(data_test_relevant[["patient_id", "image_id", "file"]].head())



## === cell 7
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Concatenate,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpu_devices = tf.config.list_physical_devices("GPU")
for device in gpu_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

ImageSize = 512
batch_size = 64
val_split = 0.0

imagegen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=val_split)


def test_from_dataframe(
    directory,
    generator,
    subset="training",
    batch_size=64,
    data=None,
    columns=None,
    seed=None,
):
    gendat = generator.flow_from_dataframe(
        data,
        directory=directory,
        shuffle=False,  # preserve alignment with dataframe order
        target_size=(ImageSize, ImageSize),
        subset=subset,
        batch_size=batch_size,
        x_col="file",
        y_col=columns,
        class_mode="raw",
        color_mode="grayscale",
        validate_filenames=False,
        seed=seed,
    )
    N = gendat.n
    i = 0
    while i < N:
        x_im, x_info = gendat.next()
        x_im = np.asarray(x_im, dtype=np.float32)
        x_info = np.asarray(x_info, dtype=np.float32)
        i += batch_size
        yield [x_im, x_info]
    return


test_gen = test_from_dataframe(
    None,
    imagegen,
    batch_size=batch_size,
    data=data_test_relevant,
    columns=tag_cols,
    subset="training",
)

with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    input_Image = Input(shape=(512, 512, 1))
    conv_Image = Conv2D(10, 13, activation="relu")(input_Image)
    pool_Image = MaxPooling2D(pool_size=10)(conv_Image)
    conv_Image = Conv2D(10, 5, activation="relu")(pool_Image)
    pool_Image = MaxPooling2D(pool_size=4)(conv_Image)
    flatten_Image = Flatten()(pool_Image)

    input_Tags = Input(shape=(10,))
    input_all = Concatenate()([flatten_Image, input_Tags])

    CNN = Dense(10, activation="relu")(input_all)
    CNN = Dense(10, activation="relu")(CNN)
    CNN = Dense(10, activation="relu")(CNN)
    CNN = Dense(10, activation="relu")(CNN)
    CNN = Dense(1, activation="sigmoid")(CNN)

    ML = Model(inputs=[input_Image, input_Tags], outputs=CNN)

ML.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

print(ML.input_shape, ML.output_shape)



## === cell 8
model_import_path = "/kaggle/input/breastcancerpredictor/breastcancerpredictor.h5"
if os.path.exists(model_import_path):
    try:
        ML.load_weights(model_import_path)
        print("Loaded weights:", model_import_path)
    except Exception as e:
        print("Could not load weights, proceeding without them. Error:", repr(e))
else:
    print("Weights file not found, proceeding without them:", model_import_path)



## === cell 9
steps = int(np.ceil(len(data_test_relevant) / batch_size))
test_pred = ML.predict(test_gen, steps=steps, verbose=1)
test_pred = test_pred.reshape(-1)[: len(data_test_relevant)]

predict_test = data_test[["patient_id", "laterality", "prediction_id"]].copy()
predict_test["cancer"] = test_pred.astype(float)

print(predict_test.head(), predict_test.shape)



## === cell 10
pred_agg = predict_test.groupby("prediction_id", as_index=False)["cancer"].max()

sub = sample_submission[["prediction_id"]].merge(
    pred_agg, on="prediction_id", how="left"
)
sub["cancer"] = sub["cancer"].fillna(0.0).clip(0.0, 1.0)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print(sub.shape)
print("Missing preds filled:", int(sub["cancer"].isna().sum()))
