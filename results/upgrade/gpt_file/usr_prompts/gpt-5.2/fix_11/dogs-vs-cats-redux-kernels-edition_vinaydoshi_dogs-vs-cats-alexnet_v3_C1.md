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

3.8

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

3.2273404955968377

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.11761) has done: 'The timeout is dominated by Python-side image I/O and preprocessing inside the generators (multiple redundant `cv2.imread` passes per batch), plus per-image crop assembly loops during 10-crop TTA. I keep the same model, augmentation, mean subtraction, patch/crop logic, and training/prediction semantics, but remove redundant reads, preallocate correctly, and vectorize the 10-crop batch assembly to reduce Python overhead. I also enable `tf.data` prefetching behavior via Keras generator options (no change in data/labels) and tune OpenCV threading to avoid oversubscription stalls. These changes are provably equivalent in outputs (same images, same preprocessors, same augmentations) while cutting constant-factor runtime substantially.'
- What this solution (achieved 0.79706) has done: 'Main bottlenecks are Python-side image decoding/augmentation inside the generator and extra overhead from legacy Keras `ImageDataGenerator` per-image transforms, plus redundant disk scans. I keep the exact model, loss, and training loop (still `model.fit(..., epochs=1)` with the same preprocessing order), but speed up the input pipeline by switching the training/validation generators to a deterministic `tf.data` pipeline that uses OpenCV decoding via `tf.numpy_function`, parallel map, prefetch, and fixed shuffling. I also remove unused HDF5 path variables (not used anywhere), reduce per-batch Python overhead, and ensure determinism is preserved by seeding both TF and NumPy and using deterministic dataset options. Ten-crop test-time augmentation remains the same core logic, but I reduce overhead by batching predictions more efficiently and avoiding repeated small allocations.'

# 9. Code solution

## === cell 0
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    for rootDir, dirNames, filenames in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind(".") :].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def resize(image, width=None, height=None, inter=None):
    if inter is None:
        inter = cv2.INTER_AREA

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized




## === cell 1
train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/"
final_test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown/"

train_img_paths = sorted(list(list_images(train_path)))
final_test_img_paths = sorted(list(list_images(final_test_path)))

len(train_img_paths), len(final_test_img_paths), train_img_paths[
    :2
], final_test_img_paths[:2]




## === cell 2
trainLabels = []
train_img_paths_filtered = []
rej = 0

for p in train_img_paths:
    fname = os.path.basename(p).lower()
    if not fname.endswith(".jpg"):
        rej += 1
        continue
    parent = os.path.basename(os.path.dirname(p)).lower()
    if parent in ("cat", "dog"):
        train_img_paths_filtered.append(p)
        trainLabels.append(parent)
    else:
        rej += 1

train_img_paths = train_img_paths_filtered
trainLabels = np.array(trainLabels)

print(
    "num images:",
    len(trainLabels),
    "rejected:",
    rej,
    "classes:",
    np.unique(trainLabels, return_counts=True),
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3886608737.py in <cell line: 0>()
     16 
     17 train_img_paths = train_img_paths_filtered
---> 18 trainLabels = np.array(trainLabels)
     19 
     20 print(

NameError: name 'np' is not defined

## === cell 3
le = LabelEncoder()
trainLabels_enc = le.fit_transform(trainLabels)
class_to_index = {c: int(i) for i, c in enumerate(le.classes_)}
dog_index = class_to_index.get("dog", 1)
print("LabelEncoder classes_:", le.classes_, "dog_index:", dog_index)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3555635125.py in <cell line: 0>()
----> 1 le = LabelEncoder()
      2 trainLabels_enc = le.fit_transform(trainLabels)
      3 class_to_index = {c: int(i) for i, c in enumerate(le.classes_)}
      4 dog_index = class_to_index.get("dog", 1)
      5 print("LabelEncoder classes_:", le.classes_, "dog_index:", dog_index)

NameError: name 'LabelEncoder' is not defined

## === cell 4
plt.figure(figsize=(4, 3))
sns.countplot(x=trainLabels)
plt.title("Class counts")
plt.tight_layout()
plt.show()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/425927876.py in <cell line: 0>()
----> 1 plt.figure(figsize=(4, 3))
      2 sns.countplot(x=trainLabels)
      3 plt.title("Class counts")
      4 plt.tight_layout()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 5
NUM_CLASSES = 2

NUM_VAL_IMAGES = 1250 * NUM_CLASSES  # 2500
NUM_TEST_IMAGES = 1250 * NUM_CLASSES  # 2500

MODEL_PATH = "/kaggle/working/alexnet_dogs_vs_cats.keras"

dataset_mean = "/kaggle/working/dogs_vs_cats_mean.json"
output_path = "/kaggle/working/"




## === cell 6
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4028584586.py in <cell line: 0>()
----> 1 class SimplePreprocessor:
      2     def __init__(self, width, height, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.inter = inter

/tmp/ipykernel_11/4028584586.py in SimplePreprocessor()
      1 class SimplePreprocessor:
----> 2     def __init__(self, width, height, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.inter = inter

NameError: name 'cv2' is not defined

## === cell 7
class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dH, dW = 0, 0

        if w < h:
            image = resize(image, width=self.width, inter=self.inter)
            dH = (image.shape[0] - self.height) // 2
        else:
            image = resize(image, height=self.height, inter=self.inter)
            dW = (image.shape[1] - self.width) // 2

        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2987769447.py in <cell line: 0>()
----> 1 class AspectAwarePreprocessor:
      2     def __init__(self, width, height, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.inter = inter

/tmp/ipykernel_11/2987769447.py in AspectAwarePreprocessor()
      1 class AspectAwarePreprocessor:
----> 2     def __init__(self, width, height, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.inter = inter

NameError: name 'cv2' is not defined

## === cell 8
n_total = len(train_img_paths)
test_size = min(NUM_TEST_IMAGES, n_total // 5)  # keep <= 20% if dataset smaller
val_size = min(NUM_VAL_IMAGES, n_total // 5)

train_paths_tmp, test_img_paths, y_train_tmp, y_test = train_test_split(
    train_img_paths,
    trainLabels_enc,
    test_size=test_size,
    random_state=42,
    stratify=trainLabels_enc,
)
train_img_paths, val_img_paths, y_train, y_val = train_test_split(
    train_paths_tmp,
    y_train_tmp,
    test_size=val_size,
    random_state=42,
    stratify=y_train_tmp,
)

len(train_img_paths), len(val_img_paths), len(test_img_paths)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3573179060.py in <cell line: 0>()
      3 val_size = min(NUM_VAL_IMAGES, n_total // 5)
      4 
----> 5 train_paths_tmp, test_img_paths, y_train_tmp, y_test = train_test_split(
      6     train_img_paths,
      7     trainLabels_enc,

NameError: name 'train_test_split' is not defined

## === cell 9
aap = AspectAwarePreprocessor(227, 227)
R, G, B = [], [], []

sample_for_mean = min(5000, len(train_img_paths))
for path in tqdm(
    train_img_paths[:sample_for_mean], desc="Computing dataset mean (subset)"
):
    image = cv2.imread(path)
    if image is None:
        continue
    image = aap.preprocess(image)
    (b, g, r) = cv2.mean(image)[:3]
    R.append(r)
    G.append(g)
    B.append(b)

B_mean = float(np.mean(B)) if len(B) else 0.0
G_mean = float(np.mean(G)) if len(G) else 0.0
R_mean = float(np.mean(R)) if len(R) else 0.0
B_mean, G_mean, R_mean




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2489804698.py in <cell line: 0>()
----> 1 aap = AspectAwarePreprocessor(227, 227)
      2 R, G, B = [], [], []
      3 
      4 sample_for_mean = min(5000, len(train_img_paths))
      5 for path in tqdm(

NameError: name 'AspectAwarePreprocessor' is not defined

## === cell 10
with open(dataset_mean, "w") as f:
    json.dump({"R": R_mean, "G": G_mean, "B": B_mean}, f)


class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = float(rMean)
        self.gMean = float(gMean)
        self.bMean = float(bMean)
        self._mean_vec = np.array(
            [self.bMean, self.gMean, self.rMean], dtype=np.float32
        )

    def preprocess(self, image):
        img = image.astype(np.float32, copy=False)
        return img - self._mean_vec




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/609883580.py in <cell line: 0>()
      1 with open(dataset_mean, "w") as f:
----> 2     json.dump({"R": R_mean, "G": G_mean, "B": B_mean}, f)
      3 
      4 
      5 class MeanPreprocessor:

NameError: name 'json' is not defined

## === cell 11
class PatchPreprocessor:
    def __init__(self, width, height, rng=None):
        self.width = width
        self.height = height
        self.rng = rng if rng is not None else np.random

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        if h <= self.height or w <= self.width:
            image = aap.preprocess(image)
            (h, w) = image.shape[:2]

        max_y = h - self.height
        max_x = w - self.width
        if max_y <= 0 or max_x <= 0:
            return cv2.resize(
                image, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        y = int(self.rng.randint(0, max_y + 1))
        x = int(self.rng.randint(0, max_x + 1))
        return image[y : y + self.height, x : x + self.width]




## === cell 12
class CropPreprocessor:
    def __init__(self, height, width, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        crops = []
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h],
        ]
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        for startX, startY, endX, endY in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3906606301.py in <cell line: 0>()
----> 1 class CropPreprocessor:
      2     def __init__(self, height, width, horiz=True, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.horiz = horiz

/tmp/ipykernel_11/3906606301.py in CropPreprocessor()
      1 class CropPreprocessor:
----> 2     def __init__(self, height, width, horiz=True, inter=cv2.INTER_AREA):
      3         self.width = width
      4         self.height = height
      5         self.horiz = horiz

NameError: name 'cv2' is not defined

## === cell 13
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array


class imageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)


aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(R_mean, G_mean, B_mean)
iap = imageToArrayPreprocessor()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
_ONE_HOT = np.eye(NUM_CLASSES, dtype=np.float32)


def _cv2_read(path_bytes):
    if isinstance(path_bytes, (np.ndarray,)):
        path_bytes = path_bytes.item()
    if isinstance(path_bytes, (bytes, bytearray)):
        path = path_bytes.decode("utf-8")
    else:
        path = str(path_bytes)

    img = cv2.imread(path)
    if img is None:
        img = np.zeros((227, 227, 3), dtype=np.uint8)
    return img


def _train_preprocess_np(path_bytes, label_int):
    img = _cv2_read(path_bytes)
    img = pp.preprocess(img)
    img = mp.preprocess(img)
    img = iap.preprocess(img)

    img = img.astype(np.float32, copy=False)
    img = aug.random_transform(img)
    img = aug.standardize(img)
    img = img.astype(np.float32, copy=False)

    y = _ONE_HOT[int(label_int)]
    return img, y


def _val_preprocess_np(path_bytes, label_int):
    img = _cv2_read(path_bytes)
    img = sp.preprocess(img)
    img = mp.preprocess(img)
    img = iap.preprocess(img)
    img = img.astype(np.float32, copy=False)

    y = _ONE_HOT[int(label_int)]
    return img, y


def make_tf_dataset(paths, labels, batch_size, training):
    paths = np.asarray(paths, dtype=object)
    labels = np.asarray(labels, dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        shuffle_buf = min(len(labels), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=False
        )

        def _map_fn(p, y):
            img, yy = tf.numpy_function(
                func=_train_preprocess_np,
                inp=[p, y],
                Tout=(tf.float32, tf.float32),
            )
            img.set_shape((227, 227, 3))
            yy.set_shape((NUM_CLASSES,))
            return img, yy

    else:

        def _map_fn(p, y):
            img, yy = tf.numpy_function(
                func=_val_preprocess_np,
                inp=[p, y],
                Tout=(tf.float32, tf.float32),
            )
            img.set_shape((227, 227, 3))
            yy.set_shape((NUM_CLASSES,))
            return img, yy

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if not training:
        ds = ds.cache()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/50114559.py in <cell line: 0>()
----> 1 _ONE_HOT = np.eye(NUM_CLASSES, dtype=np.float32)
      2 
      3 
      4 def _cv2_read(path_bytes):
      5     if isinstance(path_bytes, (np.ndarray,)):

NameError: name 'np' is not defined

## === cell 15
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
from tensorflow.keras.regularizers import l2


class Alexnet:
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(
            Conv2D(
                96,
                (11, 11),
                strides=(4, 4),
                input_shape=inputShape,
                padding="same",
                kernel_regularizer=l2(reg),
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                256, (5, 5), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(
            Conv2D(
                384, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(
            Conv2D(
                256, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(reg)
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation="softmax", kernel_regularizer=l2(reg)))
        return model




## === cell 16
model = Alexnet.build(227, 227, 3, NUM_CLASSES, reg=0.0002)
opt = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

BATCH_SIZE = 64
train_ds = make_tf_dataset(
    train_img_paths, y_train, batch_size=BATCH_SIZE, training=True
)
val_ds = make_tf_dataset(val_img_paths, y_val, batch_size=BATCH_SIZE, training=False)

steps_per_epoch = int(np.ceil(len(y_train) / BATCH_SIZE))
val_steps = int(np.ceil(len(y_val) / BATCH_SIZE))

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=1,
    verbose=1,
)

model.save(MODEL_PATH)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/995346875.py in <cell line: 0>()
----> 1 model = Alexnet.build(227, 227, 3, NUM_CLASSES, reg=0.0002)
      2 opt = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
      3 model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
      4 
      5 BATCH_SIZE = 64

/tmp/ipykernel_11/2773683921.py in build(width, height, depth, classes, reg)
     18         chanDim = -1
     19 
---> 20         if K.image_data_format() == "channels_first":
     21             inputShape = (depth, height, width)
     22             chanDim = 1

NameError: name 'K' is not defined

## === cell 17
cp = CropPreprocessor(227, 227)
aap2 = AspectAwarePreprocessor(256, 256)


def _ten_crop_batch(images_256, mp_obj, out_size=227):
    imgs = images_256.astype(np.float32, copy=False)
    imgs = mp_obj.preprocess(imgs)  # broadcast mean subtraction

    n, h, w, c = imgs.shape
    assert h == 256 and w == 256 and out_size == 227

    dW = (w - out_size) // 2  # 14
    dH = (h - out_size) // 2  # 14

    c0 = imgs[:, 0:out_size, 0:out_size, :]
    c1 = imgs[:, 0:out_size, w - out_size : w, :]
    c2 = imgs[:, h - out_size : h, w - out_size : w, :]
    c3 = imgs[:, h - out_size : h, 0:out_size, :]
    c4 = imgs[:, dH : h - dH, dW : w - dW, :]

    crops5 = np.stack([c0, c1, c2, c3, c4], axis=1)  # (n,5,227,227,3)
    mirrors5 = crops5[:, :, :, ::-1, :]  # horizontal flip (x-axis)

    crops10 = np.concatenate([crops5, mirrors5], axis=1)  # (n,10,227,227,3)
    return crops10.reshape(n * 10, out_size, out_size, c)


def test_data_generator(directory_list, bs=128, preprocessors=None, passes=1):
    for _ in range(passes):
        for i in range(0, len(directory_list), bs):
            imagePaths = directory_list[i : i + bs]
            images = []
            for path in imagePaths:
                img = cv2.imread(path)
                if img is None:
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                if preprocessors is not None:
                    for p in preprocessors:
                        img = p.preprocess(img)
                images.append(img)
            yield np.stack(images, axis=0)


def predict_with_crops(paths, batch_size=64):
    out = []
    gen = test_data_generator(paths, bs=batch_size, preprocessors=[aap2], passes=1)
    total = int(np.ceil(len(paths) / batch_size))

    for imgs in tqdm(gen, total=total, desc="Predicting with 10-crop TTA"):
        n = imgs.shape[0]
        crops_batch = _ten_crop_batch(imgs, mp, out_size=227)  # (n*10,227,227,3)

        preds = model.predict(
            crops_batch, batch_size=batch_size * 10, verbose=0
        )  # (n*10, 2)
        preds = preds.reshape(n, 10, preds.shape[1]).mean(axis=1)  # (n, 2)
        out.append(preds)

    return np.vstack(out) if len(out) else np.zeros((0, NUM_CLASSES), dtype=np.float32)


final_predict = predict_with_crops(final_test_img_paths, batch_size=64)
final_predict.shape




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2678186456.py in <cell line: 0>()
----> 1 cp = CropPreprocessor(227, 227)
      2 aap2 = AspectAwarePreprocessor(256, 256)
      3 
      4 
      5 # CHANGE (timeout): vectorize 10-crop extraction to remove per-crop cv2.resize and cv2.flip loops.

NameError: name 'CropPreprocessor' is not defined

## === cell 18
def extract_test_id(path):
    base = os.path.basename(path)
    m = re.match(r"^(\d+)\.jpg$", base)
    if m:
        return int(m.group(1))
    stem = os.path.splitext(base)[0]
    return int(stem)


final_ids = np.array([extract_test_id(p) for p in final_test_img_paths], dtype=np.int64)

dog_probs = final_predict[:, dog_index].astype(np.float64)

eps = 1e-7
dog_probs = np.clip(dog_probs, eps, 1.0 - eps)

submission = pd.DataFrame({"id": final_ids, "label": dog_probs})
submission = submission.sort_values("id").reset_index(drop=True)

submission.head(), submission.shape




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3943467155.py in <cell line: 0>()
      8 
      9 
---> 10 final_ids = np.array([extract_test_id(p) for p in final_test_img_paths], dtype=np.int64)
     11 
     12 dog_probs = final_predict[:, dog_index].astype(np.float64)

NameError: name 'np' is not defined

## === cell 19
sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.describe(include="all"))
print("Min/Max label:", submission["label"].min(), submission["label"].max())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2479804120.py in <cell line: 0>()
      1 sub_path = "/kaggle/working/submission.csv"
----> 2 submission.to_csv(sub_path, index=False)
      3 
      4 print("Wrote:", sub_path)
      5 print(submission.describe(include="all"))

NameError: name 'submission' is not defined
