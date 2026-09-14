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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.88512

# 6. Current score

4.78749

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 4.78749) has done: 'The timeout is dominated by (1) training for up to 100 epochs on the full 9199-image dataset and (2) repeatedly decoding/resizing images every epoch. To keep the exact same model, loss, optimizer, and training semantics, the main speedups are: cache the decoded+resized `tf.data` pipelines to disk (so each image is processed once), enable aggressive pipelining (AUTOTUNE map/prefetch), and avoid any redundant TFHub loads/builds. I also add `steps_per_execution` in `compile()` to reduce Python↔TF overhead without changing the math, and I ensure the TFHub module is loaded as a KerasLayer (same underlying module, same outputs) to reduce overhead. These changes are correctness-preserving (same inputs → same tensors; same training loop/epochs/early-stopping criteria) but drastically cut wall time.'

# 9. Code solution

## === cell 0
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from IPython.display import Image
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels_csv.head()



## === cell 2
print("labels_csv shape:", labels_csv.shape)
print(labels_csv["breed"].nunique(), "unique breeds")



## === cell 3
print("Skipped breed distribution plot for performance.")



## === cell 4
filenames = [
    "../input/dog-breed-identification/train/" + fname + ".jpg"
    for fname in labels_csv["id"]
]
filenames[:5]



## === cell 5
print("Skipped filesystem count check for performance.")



## === cell 6
print("Skipped sample image display for performance.")



## === cell 7
labels = labels_csv["breed"].to_numpy()
labels[:10]



## === cell 8
if len(labels) == len(filenames):
    print("Number of labels matches number of filenames!")
else:
    print(
        "Number of labels does not match number of filenames, check data directories."
    )



## === cell 9
unique_breeds = np.unique(labels)
len(unique_breeds), unique_breeds[:5]



## === cell 10
breed_to_index = {b: i for i, b in enumerate(unique_breeds)}
label_indices = labels_csv["breed"].map(breed_to_index).astype(np.int32).to_numpy()

boolean_labels = tf.one_hot(label_indices, depth=len(unique_breeds), dtype=tf.float32)
boolean_labels[:2]



## === cell 11
X = filenames
y = boolean_labels



## === cell 12
NUM_IMAGES = 1000

X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=42
)

len(X_train), len(y_train), len(X_val), len(y_val)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3558553410.py in <cell line: 0>()
      1 NUM_IMAGES = 1000
      2 
----> 3 X_train, X_val, y_train, y_val = train_test_split(
      4     X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=42
      5 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2583         train, test = next(cv.split(X=arrays[0], y=stratify))
   2584 
-> 2585     return list(
   2586         chain.from_iterable(
   2587             (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in <genexpr>(.0)
   2585     return list(
   2586         chain.from_iterable(
-> 2587             (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
   2588         )
   2589     )

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in _safe_indexing(X, indices, axis)
    354         return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    355     elif hasattr(X, "shape"):
--> 356         return _array_indexing(X, indices, indices_dtype, axis=axis)
    357     else:
    358         return _list_indexing(X, indices, indices_dtype)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in _array_indexing(array, key, key_dtype, axis)
    183     if isinstance(key, tuple):
    184         key = list(key)
--> 185     return array[key] if axis == 0 else array[:, key]
    186 
    187 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_getitem_override.py in _check_index(idx)
     60     # TODO(slebedev): IndexError seems more appropriate here, but it
     61     # will break `_slice_helper` contract.
---> 62     raise TypeError(_SLICE_TYPE_ERROR + ", got {!r}".format(idx))
     63 
     64 

TypeError: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices, got array([ 29, 535, 695, 557, 836, 596, 165, 918, 495, 824,  65, 141, 925,
       827, 655, 331, 664, 249, 907, 708, 305, 734, 975,  49, 896,   2,
       544, 350, 904, 536, 344, 994, 481, 575,  33,  31, 231, 963, 192,
       333,   3, 204, 514, 799, 306, 109, 430,  77,  84, 286,  82, 991,
       789, 894, 398, 323, 519, 916, 922,   5, 731, 465,  97, 266, 357,
       868, 798, 380, 631, 381, 490, 118, 900, 250, 523,   9, 196, 603,
        81, 783, 587, 797, 239, 290, 211, 717, 359, 449, 227, 950, 946,
       796, 501, 464, 362, 468, 935, 428,   7, 155, 541, 440, 482, 422,
       778, 949, 334, 576, 934, 567, 594, 530, 581, 707, 448, 453, 228,
       352, 728, 212,  79, 148, 302, 628, 777, 506, 342, 485, 711, 133,
       703, 311, 722, 629,   0, 316, 706, 547, 872, 532, 477, 404, 172,
       125, 394, 420, 552, 903,  90, 939, 181, 274, 895,  69, 291, 131,
       300, 424, 326, 144, 423, 580, 135, 450, 164,  28, 773, 193, 388,
       852, 169, 705, 140, 173,   6, 745, 478,  73, 910, 813, 238, 145,
       792, 234, 220, 923, 500, 132, 990, 774, 185,  41, 696, 108, 588,
        56, 405, 442, 757, 997,  24, 467, 539, 531, 618, 694, 926, 338,
        51, 507, 516, 920, 781, 264, 817, 710, 682, 832, 518, 447,  18,
       715, 483, 568, 433, 367,  83,  61, 638, 272, 285, 360, 354, 456,
       278,  12, 182, 368, 881, 615, 223, 572, 970, 653, 545, 582, 633,
       176, 665, 673, 585, 873, 393, 163, 248, 634, 885, 669, 375, 412,
        74, 113, 598, 961, 390, 104, 114, 417, 525, 457, 409,  92, 930,
        89, 336, 988, 921, 933, 605, 593, 611,  94,  11, 396, 533,  43,
        42, 329, 167, 497, 876, 597, 756, 100, 426, 178, 444, 416, 870,
       882, 680, 177, 395, 911, 793, 960, 684, 383, 956, 751, 257, 538,
       335,  15, 324, 758, 222, 179, 983,  22, 356, 666, 861, 340, 431,
       551, 833, 203, 630,  93, 558,  68, 622, 284, 844, 434, 153,  75,
       730, 446, 188, 271, 236, 487, 117, 943, 512, 825, 591, 126, 116,
       473, 693,  57, 863, 912, 369, 268,  46, 349, 195, 999, 834, 736,
       263, 443, 675, 304, 341, 966, 149, 124, 786,  50, 353, 927, 142,
       470, 399, 625, 320,  19, 809, 790, 808, 407, 537, 620,  38, 175,
       245, 828, 667, 754, 858, 154, 287, 602, 569, 743,  17, 127, 322,
       255, 657, 964, 190, 115, 616, 606, 180, 301, 759, 712, 723, 685,
       979, 517, 984,  45, 909, 157, 851, 171,  16, 511,  48, 971, 940,
       515, 952, 480, 283, 718, 877, 225,  26, 954, 437, 951, 364, 229,
        37, 965, 374, 469, 967, 850, 704, 841, 194, 854, 864, 503, 969,
       830, 579, 968, 162, 908, 152, 801, 993, 755, 111, 226, 688, 103,
       421, 419, 750, 586, 780, 672, 119,  53, 151, 403, 945, 207, 658,
       843, 762,   8, 807,  36, 452, 651, 253, 303, 746, 571, 623, 732,
       891, 262, 610, 297, 414, 150, 788, 640, 889, 550, 886, 488, 147,
       146, 720, 931, 739, 659, 348, 463, 325, 186, 123, 853, 608, 143,
       958, 197, 609, 279, 293, 400, 122, 183, 202, 438, 246, 415, 932,
       765, 906, 835, 887, 129, 637, 402, 784, 770, 735, 913, 219, 641,
       915, 752, 806, 919, 624, 874, 760, 386, 972, 509, 267, 819, 441,
       496, 112, 691, 232, 869, 607, 671, 373, 981, 842, 233, 785, 676,
       317, 648, 410, 898, 709, 358, 258, 744, 627, 632, 282, 376, 384,
       224, 953, 814, 472, 347, 505, 639, 987, 928, 905, 619, 855, 803,
       645, 846, 556, 957, 577, 795,  85, 242, 698, 159, 524,  35, 540,
       170, 654, 890, 857, 847, 944, 733,  95, 563, 240, 742, 574, 690,
       460, 553, 888, 206, 392, 794, 397, 766, 848, 217,   4, 768, 642,
       929, 612, 738, 546, 725, 683,  98, 804, 727, 573, 406, 502,  47,
        32, 779, 839, 200, 134,  27, 880, 230, 489, 772, 378, 288, 418,
       674, 391, 592, 498, 138,  62, 471, 647, 128, 976, 520, 838, 962,
        64, 812,  14, 156,  40, 492, 379, 187, 763, 216, 791,  52, 878,
       337, 748, 719, 724, 295, 701, 251, 726, 461, 455, 996, 815, 862,
       269, 201, 161, 555, 729, 401, 702, 476, 821, 771, 105, 565, 389,
         1, 937, 982, 561,  80, 205,  34, 775, 508, 427, 454, 366,  91,
       339, 897, 564, 345, 776, 241,  13, 315, 600, 387, 273, 166, 840,
       992, 646, 818, 484, 980, 504, 831, 243, 566, 875, 562, 686, 189,
       782, 699, 475, 681, 510,  58, 474, 560, 856, 747, 252,  21, 313,
       459, 160, 276, 955, 191, 385, 805, 413, 491, 343, 769, 308, 661,
       130, 663, 871,  99, 372,  87, 458, 330, 214, 466, 121, 614,  20,
       700,  71, 106, 270, 860, 435, 102])

## === cell 13
IMG_SIZE = 224


@tf.function
def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image




## === cell 14
@tf.function
def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image(image_path)
    return image, label




## === cell 15
BATCH_SIZE = 32


def create_data_batches(
    x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False, cache_name=None
):
    """
    Creates batches of data out of image (x) and label (y) pairs.

    Runtime fix (correctness-preserving):
    - Cache post-map (after decode/resize) so preprocessing happens once per image; values identical.
    - Use AUTOTUNE parallel mapping and prefetch to overlap CPU input with GPU/TF compute.
    - Keep deterministic pipeline and shuffle behavior unchanged (same seed, reshuffle_each_iteration=True).
    """
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass

    def maybe_cache(ds, cache_suffix):
        if cache_name is None:
            return ds
        os.makedirs("./tfdata_cache", exist_ok=True)
        return ds.cache(os.path.join("./tfdata_cache", f"{cache_name}_{cache_suffix}"))

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(x).with_options(options)
        data = data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        data = maybe_cache(data, "mapped")
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch

    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((x, y)).with_options(options)
        data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        data = maybe_cache(data, "mapped")
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((x, y)).with_options(options)

        buffer = min(len(x), 2048)
        data = data.shuffle(
            buffer_size=buffer, seed=SEED, reshuffle_each_iteration=True
        )

        data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
        data = maybe_cache(data, "mapped")
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch




## === cell 16
train_data = create_data_batches(X_train, y_train, cache_name="train_224_cache_1k")
val_data = create_data_batches(
    X_val, y_val, valid_data=True, cache_name="val_224_cache_1k"
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3144251792.py in <cell line: 0>()
----> 1 train_data = create_data_batches(X_train, y_train, cache_name="train_224_cache_1k")
      2 val_data = create_data_batches(
      3     X_val, y_val, valid_data=True, cache_name="val_224_cache_1k"
      4 )
      5 

NameError: name 'X_train' is not defined

## === cell 17
train_data.element_spec, val_data.element_spec



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1061206888.py in <cell line: 0>()
----> 1 train_data.element_spec, val_data.element_spec
      2 

NameError: name 'train_data' is not defined

## === cell 18
print("Skipped training batch visualization for performance.")



## === cell 19
print("Skipped fetching a batch for plotting.")



## === cell 20
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]  # batch, height, width, colour channel
OUTPUT_SHAPE = len(unique_breeds)  # number of unique labels
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4"




## === cell 21
def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with:", model_url)

    feature_extractor = hub.KerasLayer(
        model_url, trainable=False, name="hub_module", dtype=tf.float32
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape[1:]),
            feature_extractor,
            tf.keras.layers.Dense(units=output_shape, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
        steps_per_execution=16,
    )
    return model




## === cell 22
print("Skipped standalone model.summary() build to avoid redundant TFHub load.")



## === cell 23
print("TensorBoard extension skipped (script-safe).")




## === cell 24
def create_tensorboard_callback():
    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir, write_graph=False, profile_batch=0)




## === cell 25
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)



## === cell 26
print(
    "GPU",
    (
        "available (YESS!!!!)"
        if tf.config.list_physical_devices("GPU")
        else "not available :("
    ),
)



## === cell 27
NUM_EPOCHS = 100


def train_model():
    """
    Trains a given model and returns the trained version.
    """
    model = create_model()
    tensorboard = None

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(X_val) / BATCH_SIZE))

    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        validation_freq=1,
        callbacks=(
            [early_stopping] if tensorboard is None else [tensorboard, early_stopping]
        ),
        verbose=2,
    )
    return model




## === cell 28
print("Skipped initial small training run to avoid redundant compute.")



## === cell 29
print("TensorBoard magic skipped (script-safe). Logs (if any) are in ./logs")



## === cell 30
print("Skipped validation predictions for performance.")



## === cell 31
print("Skipped.")



## === cell 32
print("Skipped.")




## === cell 33
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


print("Skipped prediction label demo.")




## === cell 34
def unbatchify(data):
    """
    Takes a batched dataset of (image, label) Tensors and returns separate arrays
    of images and labels.
    """
    images = []
    labels_out = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_out.append(unique_breeds[np.argmax(label)])
    return images, labels_out


print("Skipped unbatchify/validation image extraction.")




## === cell 35
def plot_pred(prediction_probabilities, labels, images, n=1):
    """
    View the prediction, ground truth label and image for sample n.
    """
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]

    pred_label = get_pred_label(pred_prob)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    color = "green" if pred_label == true_label else "red"

    plt.title(
        "{} {:2.0f}% ({})".format(pred_label, np.max(pred_prob) * 100, true_label),
        color=color,
    )




## === cell 36
print("Skipped plot_pred for performance.")




## === cell 37
def plot_pred_conf(prediction_probabilities, labels, n=1):
    """
    Plots the top 10 highest prediction confidences along with the truth label for sample n.
    """
    pred_prob, true_label = prediction_probabilities[n], labels[n]
    pred_label = get_pred_label(pred_prob)

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    top_plot = plt.bar(
        np.arange(len(top_10_pred_labels)), top_10_pred_values, color="grey"
    )
    plt.xticks(
        np.arange(len(top_10_pred_labels)),
        labels=top_10_pred_labels,
        rotation="vertical",
    )

    if np.isin(true_label, top_10_pred_labels):
        top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")




## === cell 38
print("Skipped plot_pred_conf for performance.")



## === cell 39
print("Skipped combined prediction plots for performance.")




## === cell 40
def create_full_data_batches(x, y, batch_size=BATCH_SIZE, cache_name="full_224_cache"):
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass

    data = tf.data.Dataset.from_tensor_slices((x, y)).with_options(options)
    data = data.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_name is not None:
        os.makedirs("./tfdata_cache", exist_ok=True)
        data = data.cache(os.path.join("./tfdata_cache", f"{cache_name}_mapped"))

    data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return data


full_data = create_full_data_batches(X, y, cache_name="full_224_cache")



## === cell 41
full_model = create_model()
print("Model built for full-data training.")



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2759876386.py in <cell line: 0>()
----> 1 full_model = create_model()
      2 print("Model built for full-data training.")
      3 

/tmp/ipykernel_11/3295694361.py in create_model(input_shape, output_shape, model_url)
     12     )
     13 
---> 14     model = tf.keras.Sequential(
     15         [
     16             tf.keras.layers.InputLayer(input_shape=input_shape[1:]),

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in __init__(self, layers, trainable, name)
     73         if layers:
     74             for layer in layers:
---> 75                 self.add(layer, rebuild=False)
     76             self._maybe_rebuild()
     77 

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
     95                 layer = origin_layer
     96         if not isinstance(layer, Layer):
---> 97             raise ValueError(
     98                 "Only instances of `keras.Layer` can be "
     99                 f"added to a Sequential model. Received: {layer} "

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f2ffcb05790> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 42
full_model_tensorboard = None
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=3, restore_best_weights=True
)



## === cell 43
print("TensorBoard magic skipped (script-safe).")



## === cell 44
full_steps_per_epoch = int(np.ceil(len(X) / BATCH_SIZE))

full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    steps_per_epoch=full_steps_per_epoch,
    callbacks=(
        [full_model_early_stopping]
        if full_model_tensorboard is None
        else [full_model_tensorboard, full_model_early_stopping]
    ),
    verbose=2,
)




## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1916483404.py in <cell line: 0>()
      1 full_steps_per_epoch = int(np.ceil(len(X) / BATCH_SIZE))
      2 
----> 3 full_model.fit(
      4     x=full_data,
      5     epochs=NUM_EPOCHS,

NameError: name 'full_model' is not defined

## === cell 45
def save_model(model, suffix=None):
    """
    Saves a given model in a models directory and appends a suffix (str) for clarity and reuse.
    """
    modeldir = os.path.join("./", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = modeldir + "-" + (suffix if suffix else "model") + ".h5"
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path




## === cell 46
def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    return model




## === cell 47
loaded_full_model = full_model
print("Skipped model save/load; using trained full_model directly.")



## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4240118443.py in <cell line: 0>()
----> 1 loaded_full_model = full_model
      2 print("Skipped model save/load; using trained full_model directly.")
      3 

NameError: name 'full_model' is not defined

## === cell 48
test_path = "../input/dog-breed-identification/test/"

test_filenames = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.lower().endswith(".jpg")
    ]
)
test_filenames[:5]



## === cell 49
len(test_filenames)



## === cell 50
print("Skipped unordered test batching to avoid duplicate prediction work.")



## === cell 51
print("Skipped unordered test predictions.")



## === cell 52
print("Skipped.")



## === cell 53
sample_sub = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
breed_cols = list(sample_sub.columns[1:])  # exact breed column order required by Kaggle

unique_breeds_list = list(unique_breeds)
breed_to_pos = {b: i for i, b in enumerate(unique_breeds_list)}
idx_map = np.asarray([breed_to_pos[b] for b in breed_cols], dtype=np.int32)

preds_df = sample_sub.copy()



## === cell 54
test_ids = preds_df["id"].tolist()
ordered_test_filenames = [os.path.join(test_path, f"{id_}.jpg") for id_ in test_ids]

missing = [p for p in ordered_test_filenames if not tf.io.gfile.exists(p)]
print("Missing test files:", len(missing))

ordered_test_data = create_data_batches(
    ordered_test_filenames, test_data=True, cache_name="test_224_cache_ordered"
)



## === cell 55
pred_steps = int(np.ceil(len(ordered_test_filenames) / BATCH_SIZE))
ordered_test_predictions = loaded_full_model.predict(
    ordered_test_data, steps=pred_steps, verbose=1
)

ordered_test_predictions = ordered_test_predictions[:, idx_map]

preds_df[breed_cols] = ordered_test_predictions
preds_df.head()



## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1496800536.py in <cell line: 0>()
      1 pred_steps = int(np.ceil(len(ordered_test_filenames) / BATCH_SIZE))
----> 2 ordered_test_predictions = loaded_full_model.predict(
      3     ordered_test_data, steps=pred_steps, verbose=1
      4 )
      5 

NameError: name 'loaded_full_model' is not defined

## === cell 56
submission_path = "./full_submission_1_mobilienetV2_adam.csv"
preds_df.to_csv(submission_path, index=False)
print("Wrote submission:", submission_path, "shape:", preds_df.shape)
print(
    "Columns match sample_submission:",
    list(preds_df.columns) == list(sample_sub.columns),
)
