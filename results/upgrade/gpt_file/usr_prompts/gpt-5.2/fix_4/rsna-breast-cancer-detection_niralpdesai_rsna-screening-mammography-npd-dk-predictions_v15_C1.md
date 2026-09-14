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
import glob
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

os.environ.setdefault("PYTHONHASHSEED", "0")




## === cell 1
DATA_DIR = "/kaggle/input/rsna-breast-cancer-detection"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"

csv_train = pd.read_csv(TRAIN_CSV)
csv_test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_SUB)

print(csv_train.shape, csv_test.shape, sample_submission.shape)
print(sample_submission.columns.tolist())




## === cell 2
print(csv_train.groupby("cancer").cancer.count())




## === cell 3
testdir = TEST_IMG_DIR
some_paths = []
for i, p in enumerate(glob.iglob(f"{testdir}/**/*.dcm", recursive=True)):
    some_paths.append(p)
    if i >= 3:
        break
print("Example test DICOM paths:", some_paths)




## === cell 4
columns_in_test = [c for c in csv_test.columns if c != "prediction_id"]
columns_in_test.append("cancer")

data_train = csv_train[columns_in_test].copy()

bad_ids = [1942326353]
data_train = data_train.drop(data_train[data_train["image_id"].isin(bad_ids)].index)

data_train = data_train.dropna(subset=["age"]).copy()
data_train["age"] = data_train["age"] / data_train["age"].max()

print("data_train:", data_train.shape)




## === cell 5
data_test = csv_test.copy()
data_test["age"] = data_test["age"].fillna(data_test["age"].mean())
data_test["age"] = data_test["age"] / data_train["age"].max()

print("data_test:", data_test.shape)




## === cell 6
cancer_train = data_train.loc[data_train.cancer == 1]
nocancer_train = data_train.loc[data_train.cancer == 0]
print("cancer rows:", len(cancer_train), "non-cancer rows:", len(nocancer_train))
cancer_train.head()




## === cell 7
print("Unique patients:", len(data_train["patient_id"].unique()))
print(
    "Duplicates check (image_id unique - rows):",
    len(data_train["image_id"].unique()) - len(data_train),
)




## === cell 8
import pydicom


_DICOM_TAGS = [
    "PixelData",
    "PhotometricInterpretation",
    "BitsStored",
    "BitsAllocated",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "RescaleIntercept",
    "RescaleSlope",
    "PixelRepresentation",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "TransferSyntaxUID",
]


def load_dicom_array(dcm_path: str):
    """Return float32 image in [0,1]. If decoding fails, return None."""
    try:
        ds = pydicom.dcmread(dcm_path, force=True, specific_tags=_DICOM_TAGS)
        arr = ds.pixel_array.astype(np.float32)
        photometric = getattr(ds, "PhotometricInterpretation", None)
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return arr
    except Exception:
        return None




## === cell 9
test_im_dir = TEST_IMG_DIR
patient_str = data_test["patient_id"].astype(np.int64).astype(str).to_numpy()
image_str = data_test["image_id"].astype(np.int64).astype(str).to_numpy()
path = (test_im_dir + "/" + patient_str + "/" + image_str + ".dcm").tolist()
print("Num test paths:", len(path), "first:", path[0])




## === cell 10
import cv2


def cut_empty_space_ray(im, T=100, cutedge=10, show=False):
    if im.shape[0] <= 2 * cutedge or im.shape[1] <= 2 * cutedge:
        return im
    impx_cv2_raw = im[cutedge:-cutedge, cutedge:-cutedge]

    _, impx_cv2 = cv2.threshold(
        impx_cv2_raw.astype(np.float32), T, 255, cv2.THRESH_BINARY
    )
    impx_cv2_u8 = impx_cv2.astype(np.uint8)

    contours, _ = cv2.findContours(
        impx_cv2_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return impx_cv2_raw

    boundary = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(boundary)
    out = impx_cv2_raw[y : y + h, x : x + w]
    if show:
        plt.imshow(out, cmap="gray")
        plt.show()
        plt.close()
    return out


def process_im_ray_from_array(out, image_size=256, show=False):
    if out is None:
        return None
    if show:
        plt.imshow(out, cmap="gray")
        plt.show()
        plt.close()

    maxval = float(out.max())
    Threshold = maxval / 5.0
    out = cut_empty_space_ray(out, T=Threshold, show=show)

    minval2 = float(out.min())
    maxval2 = float(out.max())
    if maxval2 > minval2:
        out = (out - minval2) / (maxval2 - minval2)
    else:
        out = np.zeros_like(out, dtype=np.float32)

    out = cv2.resize(out, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if show:
        plt.imshow(out, cmap="gray")
        plt.show()
        plt.close()
    return out.astype(np.float32)




## === cell 11
from concurrent.futures import ThreadPoolExecutor

start = time.time()
image_size = 512

run_dcm_to_png = True
max_to_process = None  # set to an int for debugging; keep None for full run

if run_dcm_to_png:
    workdir = "/kaggle/working/test/"
    save_dir = workdir + f"processed_{image_size}"
    os.makedirs(save_dir, exist_ok=True)

    iterable = path if max_to_process is None else path[:max_to_process]

    if max_to_process is None:
        patient_ids_to_make = np.unique(patient_str)
    else:
        patient_ids_to_make = np.unique(patient_str[:max_to_process])

    for pid in patient_ids_to_make.tolist():
        os.makedirs(os.path.join(save_dir, pid), exist_ok=True)

    out_paths = []
    for imagepath in iterable:
        base = os.path.basename(imagepath)
        image_id = os.path.splitext(base)[0]
        patient_id = os.path.basename(os.path.dirname(imagepath))
        out_paths.append(os.path.join(save_dir, patient_id, image_id + ".png"))

    n_exist = sum(os.path.exists(p) for p in out_paths)
    if n_exist == len(out_paths):
        print(
            f"All processed PNGs already exist ({n_exist}/{len(out_paths)}). Skipping conversion."
        )
    else:
        def _convert_one(args):
            imagepath, out_path = args
            if os.path.exists(out_path):
                return (1, 0)

            arr = load_dicom_array(imagepath)
            proc = process_im_ray_from_array(arr, image_size=image_size, show=False)
            if proc is None:
                image_save = np.zeros((image_size, image_size), dtype=np.uint8)
                cv2.imwrite(out_path, image_save)
                return (0, 1)

            image_save = (proc * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(out_path, image_save)
            return (1, 0)

        n_workers = min(32, (os.cpu_count() or 8))
        n_ok = 0
        n_fail = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for ok, fail in tqdm(
                ex.map(_convert_one, zip(iterable, out_paths)), total=len(iterable)
            ):
                n_ok += ok
                n_fail += fail

        print(f"Converted: ok={n_ok}, fail={n_fail}, dir={save_dir}")

print("Elapsed seconds:", time.time() - start)




## === cell 12
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




## === cell 13
for c in data_train_relevant.columns:
    if c not in data_test_relevant.columns and c != "cancer":
        data_test_relevant[c] = 0
        data_test_relevant[c] = data_test_relevant[c].astype("uint8")

for c in data_test_relevant.columns:
    if c not in data_train_relevant.columns and c != "cancer":
        data_train_relevant[c] = 0

print("train relevant cols:", data_train_relevant.columns.tolist())
print("test relevant cols:", data_test_relevant.columns.tolist())




## === cell 14
pngfolder = f"/kaggle/working/test/processed_{image_size}"
pids = data_test_relevant["patient_id"].astype(np.int64).astype(str).to_numpy()
iids = data_test_relevant["image_id"].astype(np.int64).astype(str).to_numpy()
data_test_relevant["file"] = pngfolder + "/" + pids + "/" + iids + ".png"

missing_files = (~data_test_relevant["file"].map(os.path.exists)).sum()
print("Missing processed PNG files:", int(missing_files))




## === cell 15
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Concatenate,
)
from tensorflow.keras.models import Model

tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

gpu_devices = tf.config.list_physical_devices("GPU")
for device in gpu_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

ImageSize = 512
batch_size = 64

cols = ["age", "implant", "L", "R", "AT", "CC", "LM", "LMO", "ML", "MLO"]

for c in cols:
    if c not in data_test_relevant.columns:
        data_test_relevant[c] = 0.0
data_test_relevant[cols] = data_test_relevant[cols].astype(np.float32)


def make_tf_dataset(df, batch_size=64, image_size=512, tag_cols=None, shuffle=False):
    paths = df["file"].values
    tags = df[tag_cols].values.astype(np.float32)

    def _load(path, tag):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img.set_shape([image_size, image_size, 1])
        return (img, tag)

    ds = tf.data.Dataset.from_tensor_slices((paths, tags))
    opt = tf.data.Options()
    opt.experimental_deterministic = True
    ds = ds.with_options(opt)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=False)

    ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_tf_dataset(
    data_test_relevant,
    batch_size=batch_size,
    image_size=ImageSize,
    tag_cols=cols,
    shuffle=False,
)




## === cell 16
data_test_relevant.head()




## === cell 17
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    input_Image = Input(shape=(512, 512, 1))
    conv_Image = Conv2D(10, 13, activation="relu", input_shape=(512, 512, 1))(
        input_Image
    )
    pool_Image = MaxPooling2D(pool_size=10)(conv_Image)
    conv_Image2 = Conv2D(10, 5, activation="relu", input_shape=(50, 50, 10))(pool_Image)
    pool_Image2 = MaxPooling2D(pool_size=4)(conv_Image2)
    flatten_Image = Flatten()(pool_Image2)

    input_Tags = Input(shape=(10,))
    input_all = Concatenate()([flatten_Image, input_Tags])

    CNN = Dense(10, activation="relu")(input_all)
    CNN = Dense(10, activation="relu")(CNN)
    CNN = Dense(10, activation="relu")(CNN)
    CNN = Dense(10, activation="relu")(CNN)
    output = Dense(1, activation="sigmoid")(CNN)

    ML = Model(inputs=[input_Image, input_Tags], outputs=output)
    ML.summary()

    ML.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

print(
    "Model built. Input image shape:",
    input_Image.shape,
    "Input tags shape:",
    input_Tags.shape,
)




## === cell 18
model_import_path = "/kaggle/input/breastcancerpredictor/breastcancerpredictor.h5"
if os.path.exists(model_import_path):
    ML.load_weights(model_import_path)
    print("Loaded weights:", model_import_path)
else:
    print(
        "Weights file not found, using randomly initialized model:", model_import_path
    )




## === cell 19
test_pred = ML.predict(test_ds, verbose=1)
print(
    "Pred shape:",
    test_pred.shape,
    "min/max:",
    float(test_pred.min()),
    float(test_pred.max()),
)




## === cell 20
test_pred_series = pd.Series(test_pred.reshape(-1))
predict_test = data_test.copy()
predict_test["cancer"] = test_pred_series.values
print(predict_test[["patient_id", "laterality", "prediction_id", "cancer"]].head())




## === cell 21
def sigmoid(x, m=10):
    y = 2 * m * x - m
    return 1 / (1 + np.exp(-y))


predict_merge = predict_test.groupby("prediction_id", as_index=False)["cancer"].mean()
predict_merge["cancer"] = sigmoid(predict_merge["cancer"].values)

sub = sample_submission[["prediction_id"]].merge(
    predict_merge, on="prediction_id", how="left"
)
sub["cancer"] = sub["cancer"].fillna(
    sub["cancer"].mean() if sub["cancer"].notna().any() else 0.0
)
sub["cancer"] = sub["cancer"].clip(0.0, 1.0)

print(sub.head())
print("Submission shape:", sub.shape, "missing preds:", int(sub["cancer"].isna().sum()))




## === cell 22
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", sub.columns.tolist())
print(pd.read_csv("submission.csv").head())
