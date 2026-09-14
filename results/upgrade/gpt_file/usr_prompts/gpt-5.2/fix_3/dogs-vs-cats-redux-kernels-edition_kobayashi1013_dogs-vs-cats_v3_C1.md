# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

print("Input root exists:", os.path.exists("/kaggle/input"))
print("Input top-level entries:", sorted(os.listdir("/kaggle/input"))[:50])



## === cell 1
import zipfile  # zipファイルの解凍に必要
import tensorflow as tf  # 機械学習に必要
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # データの正規化に必要
from tensorflow.keras import layers, models  # レイヤークラス, 学習モデル
from tensorflow.keras.preprocessing import image  # 画像の前処理に必要
import shutil  # クラス分けに必要
import random
import re
from pathlib import Path

print("Python:", os.sys.version)
print("TensorFlow:", tf.__version__)




## === cell 2
def find_dir(root, target_name):
    root = Path(root)
    for p in root.rglob(target_name):
        if p.is_dir() and p.name == target_name:
            return str(p)
    return None


def find_flat_train_images_dir(root):
    root = Path(root)
    candidates = [
        root / "train",
        root / "dogs-vs-cats-redux-kernels-edition" / "train",
        root
        / "dogs-vs-cats-redux-kernels-edition"
        / "dogs-vs-cats-redux-kernels-edition"
        / "train",
    ]
    for c in candidates:
        if c.is_dir():
            try:
                files = [p.name.lower() for p in c.iterdir() if p.is_file()]
            except Exception:
                continue
            has_cat = any(fn.startswith("cat.") and fn.endswith(".jpg") for fn in files)
            has_dog = any(fn.startswith("dog.") and fn.endswith(".jpg") for fn in files)
            if has_cat and has_dog:
                return str(c)

    for dirpath, _, filenames in os.walk(str(root)):
        filenames_l = [f.lower() for f in filenames]
        if any(
            f.startswith("cat.") and f.endswith(".jpg") for f in filenames_l
        ) and any(f.startswith("dog.") and f.endswith(".jpg") for f in filenames_l):
            return dirpath
    return None


def find_flat_test_images_dir(root):
    root = Path(root)
    candidates = [
        root / "test",
        root / "dogs-vs-cats-redux-kernels-edition" / "test",
        root
        / "dogs-vs-cats-redux-kernels-edition"
        / "dogs-vs-cats-redux-kernels-edition"
        / "test",
    ]
    for c in candidates:
        if c.is_dir():
            try:
                for p in c.iterdir():
                    if (
                        p.is_file()
                        and p.suffix.lower() == ".jpg"
                        and re.fullmatch(r"\d+", p.stem)
                    ):
                        return str(c)
            except Exception:
                pass

    for dirpath, _, filenames in os.walk(str(root)):
        numeric_jpgs = 0
        for f in filenames:
            if f.lower().endswith(".jpg") and re.fullmatch(r"\d+\.jpg", f):
                numeric_jpgs += 1
                if numeric_jpgs >= 20:
                    return dirpath
    return None




## === cell 3
hyper_epochs = 10
hyper_split_rate = 0.8
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



## === cell 4
model = models.Sequential(
    [
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),  # 画像化から特徴を抽出する
        layers.MaxPooling2D(2, 2),  # 重要な情報を残しつつ、画像サイズを圧縮
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),  # 畳み込み層の出力を、1次元のベクトルに変換
        layers.Dense(512, activation="relu"),  # 中間特徴を学習し、分析精度を向上させる
        layers.Dense(1, activation="sigmoid"),  # sigmoidにより0-1の確率を算出
    ]
)



## === cell 5
model.compile(
    loss="binary_crossentropy",  # 損失関数
    optimizer="adam",  # 最適化アルゴリズム
    metrics=["accuracy"],  # 評価指標
)



## === cell 6
zip_file = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
out_dir = "/kaggle/working"
sentinel = os.path.join(out_dir, ".train_zip_extracted.ok")

if not os.path.exists(sentinel):
    with zipfile.ZipFile(zip_file, "r") as file:
        file.extractall(out_dir)
    with open(sentinel, "w") as f:
        f.write("ok\n")

extracted_train_images_dir = find_flat_train_images_dir(out_dir)
if extracted_train_images_dir is None:
    raise FileNotFoundError(
        "Could not find extracted train images directory containing cat.*.jpg and dog.*.jpg under /kaggle/working"
    )

print("Extracted train images dir:", extracted_train_images_dir)




## === cell 7
def _safe_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


learn_dir = extracted_train_images_dir
learn_file_names = [
    f
    for f in os.listdir(learn_dir)
    if os.path.isfile(os.path.join(learn_dir, f)) and f.lower().endswith(".jpg")
]
if len(learn_file_names) == 0:
    raise ValueError(f"No training images found in {learn_dir}")

random.shuffle(learn_file_names)

split_point = int(len(learn_file_names) * hyper_split_rate)
train_file_names = learn_file_names[:split_point]
val_file_names = learn_file_names[split_point:]

base_split_dir = "/kaggle/working/split_data"
train_dir = os.path.join(base_split_dir, "train")
val_dir = os.path.join(base_split_dir, "val")

train_cats = os.path.join(train_dir, "cats")
train_dogs = os.path.join(train_dir, "dogs")
val_cats = os.path.join(val_dir, "cats")
val_dogs = os.path.join(val_dir, "dogs")
for d in (train_cats, train_dogs, val_cats, val_dogs):
    os.makedirs(d, exist_ok=True)


def _class_dst_dir(fname, is_train):
    f = fname.lower()
    if f.startswith("cat"):
        return train_cats if is_train else val_cats
    elif f.startswith("dog"):
        return train_dogs if is_train else val_dogs
    else:
        return None


for file_name in train_file_names:
    dst_base = _class_dst_dir(file_name, True)
    if dst_base is None:
        continue
    src = os.path.join(learn_dir, file_name)
    dst = os.path.join(dst_base, file_name)
    _safe_link(src, dst)

for file_name in val_file_names:
    dst_base = _class_dst_dir(file_name, False)
    if dst_base is None:
        continue
    src = os.path.join(learn_dir, file_name)
    dst = os.path.join(dst_base, file_name)
    _safe_link(src, dst)

print("Split counts:", len(train_file_names), len(val_file_names))
print("Train dirs:", train_dir)
print("Val dirs:", val_dir)




## === cell 8
def devide_class(any_dir):
    return


devide_class(train_dir)
devide_class(val_dir)



## === cell 9
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode="binary", shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode="binary", shuffle=False
)

if train_generator.samples == 0 or val_generator.samples == 0:
    raise ValueError(
        f"Empty generator: train samples={train_generator.samples}, val samples={val_generator.samples}"
    )

print("Class indices:", train_generator.class_indices)



## === cell 10
history = model.fit(train_generator, epochs=hyper_epochs, validation_data=val_generator)



## === cell 11
model_file = "/kaggle/working/cnn.h5"
model.save(model_file)
print("Saved model to:", model_file)



## === cell 12
zip_file = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
out_dir = "/kaggle/working"
sentinel = os.path.join(out_dir, ".test_zip_extracted.ok")

if not os.path.exists(sentinel):
    with zipfile.ZipFile(zip_file, "r") as file:
        file.extractall(out_dir)
    with open(sentinel, "w") as f:
        f.write("ok\n")

extracted_test_images_dir = find_flat_test_images_dir(out_dir)
if extracted_test_images_dir is None:
    raise FileNotFoundError(
        "Could not find extracted test images directory containing numeric *.jpg under /kaggle/working"
    )

print("Extracted test images dir:", extracted_test_images_dir)



## === cell 13
test_dir = extracted_test_images_dir
test_files = [
    f
    for f in os.listdir(test_dir)
    if os.path.isfile(os.path.join(test_dir, f))
    and f.lower().endswith(".jpg")
    and re.fullmatch(r"\d+\.jpg", f.lower())
]

test_ids = np.array([int(os.path.splitext(f)[0]) for f in test_files], dtype=np.int64)
order = np.argsort(test_ids)
sorted_files = [test_files[i] for i in order]
sorted_ids = test_ids[order].tolist()
image_path_list = [os.path.join(test_dir, f) for f in sorted_files]

if len(image_path_list) == 0:
    raise ValueError(f"No test images found in {test_dir}")

print("Num test images:", len(image_path_list))
print("First/last test file:", sorted_files[0], sorted_files[-1])

df_test = pd.DataFrame({"filename": image_path_list})

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col="filename",
    y_col=None,
    target_size=(150, 150),
    class_mode=None,
    batch_size=64,
    shuffle=False,
)

preds = model.predict(test_generator, verbose=0).reshape(-1)

labels = np.clip(preds.astype(np.float64), 1e-6, 1 - 1e-6)

df = pd.DataFrame({"id": sorted_ids, "label": labels})
df = df.sort_values("id").reset_index(drop=True)

sub_path = "/kaggle/working/submission.csv"
df.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)
print(df.head())
print(df.tail())
