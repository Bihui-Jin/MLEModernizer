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
Label images of animals with their species.

## Metric
Macro F1 score

## Submission Format
```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Dataset
The training set contains 196,157 images from 138 different locations in Southern California. 

The test set contains 153,730 images from 100 locations in Idaho.

The task is to label each image with one of the following label ids:

```
name, id
empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22
```

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
            train/
                train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
        input/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
                    test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
        working/
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
```

-> data/iwildcam-2019-fgvc6/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/iwildcam-2019-fgvc6/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/iwildcam-2019-fgvc6/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> (stopped after 10 files for performance)

# 5. Target score

0.0390505687738178

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/iwildcam-2019-fgvc6/train.csv")
test = pd.read_csv("../input/iwildcam-2019-fgvc6/test.csv")
sample_submission = pd.read_csv("../input/iwildcam-2019-fgvc6/sample_submission.csv")

print("test.shape:", test.shape)
print("sample_submmission.shape:", sample_submission.shape)
print("train.shape:", train.shape)
print("sample_submission columns:", sample_submission.columns.tolist())



## === cell 2
import cv2
import matplotlib.pyplot as plt
import tqdm
import math
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

print("PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 3
train_id = train["file_name"]
labels = train["category_id"]
test_id = sample_submission["Id"]

example_path = os.path.join(
    "../input/iwildcam-2019-fgvc6/train_images", train["file_name"].iloc[0]
)
img = plt.imread(example_path)
print("Example image shape:", img.shape)



## === cell 4
TRAIN_IMAGES_DIR = "../input/iwildcam-2019-fgvc6/train_images"
TEST_IMAGES_DIR = "../input/iwildcam-2019-fgvc6/test_images"


def _read_resize_rgb(path, size=(32, 32)):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
    return im


from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = "../working/cache_iwildcam32"
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_TRAIN_IMAGES = 20000  # keep user's existing cap (core logic expectation)
train_subset = train.iloc[:MAX_TRAIN_IMAGES].copy().reset_index(drop=True)

num_classes = 23
y_trains1 = train_subset["category_id"].values.astype(np.int64)
y_trains1_oh = np.eye(num_classes, dtype=np.float32)[y_trains1]

if len(test) != len(sample_submission):
    raise RuntimeError(
        f"Row count mismatch: test has {len(test)} rows but sample_submission has {len(sample_submission)} rows"
    )

test_ids = test["id"].astype(str).values
test_file_names = test["file_name"].values


def _cache_path(name):
    return os.path.join(CACHE_DIR, name)


def _load_or_build_cache_uint8(cache_path, paths, size=(32, 32), max_workers=8):
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        return arr

    n = len(paths)
    out = np.zeros((n, size[1], size[0], 3), dtype=np.uint8)

    def _worker(i_path):
        i, p = i_path
        return i, _read_resize_rgb(p, size=size)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, im in tqdm.tqdm(ex.map(_worker, enumerate(paths)), total=n):
            out[i] = im

    np.save(cache_path, out)
    return out


train_paths = [
    os.path.join(TRAIN_IMAGES_DIR, fn) for fn in train_subset["file_name"].values
]
test_paths = [os.path.join(TEST_IMAGES_DIR, fn) for fn in test_file_names]

x_trains1 = _load_or_build_cache_uint8(
    _cache_path(f"train_uint8_{len(train_subset)}.npy"),
    train_paths,
    size=(32, 32),
    max_workers=min(8, (os.cpu_count() or 4)),
)
x_test_arr = _load_or_build_cache_uint8(
    _cache_path(f"test_uint8_{len(test_paths)}.npy"),
    test_paths,
    size=(32, 32),
    max_workers=min(8, (os.cpu_count() or 4)),
)

print("x_trains1.shape:", x_trains1.shape)
print("y_trains1_oh.shape:", y_trains1_oh.shape)
print("x_test_arr shape:", x_test_arr.shape)
print(
    "Expected test rows:",
    len(test),
    "Actual cached test rows:",
    len(x_test_arr),
)



## === cell 5
x_test = x_test_arr.astype("float32") / 255.0
x_trains1_f = x_trains1.astype("float32") / 255.0

print("x_test.shape:", x_test.shape)
print("x_trains1_f.shape:", x_trains1_f.shape)



## === cell 6
y_trains1 = y_trains1_oh
print("x_trains1.shape:", x_trains1_f.shape)
print("y_trains1.shape:", y_trains1.shape)



## === cell 7
x_train, x_dev, y_train, y_dev = train_test_split(
    x_trains1_f,
    y_trains1,
    test_size=0.10,
    random_state=32,
    stratify=np.argmax(y_trains1, axis=1),
)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_dev.shape:", x_dev.shape)
print("y_dev.shape:", y_dev.shape)

print("Prediction will be batched over full x_test.")
print("x_test.shape:", x_test.shape)



## === cell 8
pass




## === cell 9
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=1, padding=4 // 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=2 // 2)
        self.fc = nn.Linear(32 * 1 * 1, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=8, stride=8, ceil_mode=True)  # 32->4
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=8, stride=8, ceil_mode=True)  # 4->1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :
        ]
        mini_batch_Y = shuffled_Y[
            k * mini_batch_size : k * mini_batch_size + mini_batch_size, :
        ]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[
            num_complete_minibatches * mini_batch_size : m, :, :, :
        ]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches




## === cell 10
def model(
    X_train,
    Y_train,
    X_dev,
    Y_dev,
    X_test_full,
    test_ids_full,
    learning_rate=0.009,
    num_epochs=20,
    minibatch_size=64,
    print_cost=True,
):
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    m = X_train.shape[0]
    costs = []

    net = SimpleCNN(num_classes=Y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    mse = nn.MSELoss(reduction="mean")

    def _to_torch_images(x_nhwc):
        return torch.from_numpy(np.transpose(x_nhwc, (0, 3, 1, 2))).float()

    def _accuracy(X_np, Y_np, max_eval=25000):
        net.eval()
        n = X_np.shape[0]
        if n > max_eval:
            X_np = X_np[:max_eval]
            Y_np = Y_np[:max_eval]
        with torch.no_grad():
            xb = _to_torch_images(X_np).to(device)
            logits = net(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            true = np.argmax(Y_np, axis=1)
            return float((pred == true).mean()), X_np.shape[0]

    def _predict_in_batches(X_np, batch_size=2048):
        net.eval()
        n = X_np.shape[0]
        preds = np.empty((n,), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = _to_torch_images(X_np[start:end]).to(device)
                logits = net(xb)
                preds[start:end] = (
                    torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
                )
        return preds

    for epoch in range(num_epochs):
        net.train()
        minibatch_cost = 0.0
        num_minibatches = int(np.ceil(m / minibatch_size))

        seed_for_epoch = 3 + epoch + 1
        minibatches = random_mini_batches(
            X_train, Y_train, minibatch_size, seed=seed_for_epoch
        )

        for minibatch_X, minibatch_Y in minibatches:
            xb = _to_torch_images(minibatch_X).to(device)
            yb = torch.from_numpy(minibatch_Y).float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = mse(logits, yb)  # keep original objective semantics
            loss.backward()
            optimizer.step()

            minibatch_cost += float(loss.detach().cpu().item()) / num_minibatches

        if print_cost and epoch % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost:
            costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    train_accuracy, number_for_train_accuracy = _accuracy(
        X_train, Y_train, max_eval=25000
    )
    dev_accuracy, number_for_dev_accuracy = _accuracy(X_dev, Y_dev, max_eval=25000)

    print("Train Accuracy:", train_accuracy)
    print("Dev Accuracy:", dev_accuracy)

    test_results = _predict_in_batches(X_test_full, batch_size=2048)

    if len(test_results) != len(test_ids_full):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(test_results)} vs expected {len(test_ids_full)}"
        )

    submission = pd.DataFrame(
        {
            "Id": np.asarray(test_ids_full, dtype=str),
            "Category": np.array(test_results, dtype=np.int64),
        }
    )
    out_path = "submission_got_it10.csv"
    submission.to_csv(out_path, index=False)

    submission_debug = submission.rename(columns={"Category": "Predicted"})
    out_path_debug = "submission_got_it10_debug_predicted.csv"
    submission_debug.to_csv(out_path_debug, index=False)

    print("Used in training set:", X_train.shape[0], "elements")
    print("Used in validation set:", X_dev.shape[0], "elements")
    print("Used in prediction set:", len(test_results), "elements")
    print("Used for train accuracy:", number_for_train_accuracy, "elements")
    print("Used for dev accuracy:", number_for_dev_accuracy, "elements")
    print(submission.head())
    print("Summary of predictions:")
    print(submission.Category.value_counts())

    parameters = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return train_accuracy, dev_accuracy, parameters, out_path




## === cell 11
_, _, parameters, out_path = model(
    x_train,
    y_train,
    x_dev,
    y_dev,
    x_test,
    test_ids,
    learning_rate=0.0050,
    num_epochs=150,
    minibatch_size=64,
    print_cost=True,
)

print("Wrote submission to:", out_path)



## === cell 12
print(os.listdir("../working"))
print("Expected submission path:", os.path.abspath(out_path))
if os.path.exists(out_path):
    sub = pd.read_csv(out_path)
    print(sub.head())
    print(sub.columns.tolist())
    print("submission rows:", len(sub))
else:
    print("Submission file not found. Files in CWD:", os.listdir("."))
    raise FileNotFoundError(out_path)
