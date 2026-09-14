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
Predict whether an image contains a ship or an iceberg.

## Metric
Log loss.

## Submission Format
For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Dataset
The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

sample_submission.csv: The submission file in the correct format:

# 2. Python version

3.6

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 5. Target score

0.59897

# 6. Current score

0.83418

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.65108) has done: 'The fix removes hard‑coded reshapes, correctly normalises the bands, builds a single feature matrix (both bands), uses a simple NumPy split instead of sklearn, and generates predictions on the test set with the proper probability column so a valid `submission.csv` is written.'
- What this solution (achieved 2.61417) has done: 'I add simple statistical features (mean, std of each band) and the numeric incidence angle (filled with the training median) to the feature set, and keep the same logistic‑regression pipeline. This modest enrichment usually lowers log‑loss toward the target without altering the core model code, and it also fixes any missing‑value handling for `inc_angle`. The script now writes a proper `submission.csv` with the required probability column.'
- What this solution (achieved 0.80196) has done: 'I improve the preprocessing by standardising each feature (zero‑mean, unit‑variance) instead of a single global max‑scaling, and I use the same statistics for the test set. This better‑conditioned data lets the same logistic‑regression core converge to more useful probabilities, moving the log‑loss far closer to the target. I also increase the optimisation iterations modestly and lower the learning rate for a steadier descent. The rest of the logic—including the model, split and submission writing—remains unchanged.'
- What this solution (achieved 0.71634) has done: 'I add a small L2 regularisation term to the logistic‑regression gradients and cost, and train a bit longer with a slightly smaller learning rate. This keeps the model architecture identical while reducing over‑fitting, which should lower the log‑loss toward the target score.'
- What this solution (achieved 0.71651) has done: 'I keep the overall pipeline unchanged but tune the logistic‑regression hyper‑parameters to reduce over‑regularisation and give the optimizer more steps to converge, which should lower the log‑loss toward the target. Specifically, I lower the L2 strength, halve the learning rate, and double the iteration count in the model call.'
- What this solution (achieved 0.82133) has done: 'I add a simple interaction feature (mean of the element‑wise product of the two bands) to give the logistic model a bit more signal, and then relax the L2 regularisation while increasing the number of training iterations and the learning rate so the model can converge better. These changes keep the overall architecture and training loop identical, but they are expected to lower the log‑loss toward the target.'
- What this solution (achieved 0.81977) has done: 'I re‑enable a modest L2 regularisation (lambda = 0.1) when training the logistic regression model, because removing regularisation caused the validation loss to rise. Adding this small penalty should reduce over‑fitting and move the log‑loss closer to the target while keeping all other logic unchanged.'
- What this solution (achieved 0.83418) has done: 'The update keeps the exact logistic‑regression algorithm but cuts unnecessary work: `propagate` now computes the costly log‑loss only when requested, and the training loop calls it with this flag only every 100 steps. The number of gradient‑descent iterations is reduced to 5 000 (still enough to converge on the small dataset) and the learning‑rate is raised slightly to keep convergence speed. These changes dramatically lower runtime while preserving the model, loss, and prediction logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
path = "../input/"




## === cell 1
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s




## === cell 2
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    assert w.shape == (dim, 1)
    assert isinstance(b, float) or isinstance(b, int)
    return w, b




## === cell 3
def propagate(w, b, X, Y, lambda_reg=0.0, compute_cost=True):
    """
    Compute gradients (and optionally cost) for logistic regression with L2 regularisation.
    Setting compute_cost=False skips the expensive log‑loss calculation.
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)

    if compute_cost:
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        cost += (lambda_reg / (2 * m)) * np.sum(np.square(w))
        cost = np.squeeze(cost)
    else:
        cost = None

    dw = (1 / m) * np.dot(X, (A - Y).T) + (lambda_reg / m) * w
    db = (1 / m) * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    return grads, cost




## === cell 4
def optimize(
    w, b, X, Y, num_iterations, learning_rate, lambda_reg=0.0, print_cost=False
):
    costs = []
    for i in range(num_iterations):
        compute_cost = i % 100 == 0
        grads, cost = propagate(w, b, X, Y, lambda_reg, compute_cost)

        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

        if compute_cost:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")
    params = {"w": w, "b": b}
    return params, grads, costs




## === cell 5
def predict(w, b, X):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    Y_pred = (A > 0.5).astype(int)
    assert Y_pred.shape == (1, m)
    return Y_pred, A




## === cell 6
def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations=20000,
    learning_rate=0.001,
    lambda_reg=0.1,
    print_cost=False,
):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(
        w,
        b,
        X_train,
        Y_train,
        num_iterations,
        learning_rate,
        lambda_reg,
        print_cost,
    )
    w, b = params["w"], params["b"]
    Y_pred_test, A_test = predict(w, b, X_test)
    Y_pred_train, A_train = predict(w, b, X_train)

    print(
        "train accuracy: {:.2f} %".format(
            100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        )
    )
    print(
        "test accuracy:  {:.2f} %".format(
            100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100
        )
    )

    return {
        "w": w,
        "b": b,
        "train_with_prob": A_train,
        "test_with_prob": A_test,
        "costs": costs,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "lambda_reg": lambda_reg,
    }




## === cell 7
def train_val_split(X, Y, split_perc=0.25, seed=42):
    np.random.seed(seed)
    m = X.shape[1]
    indices = np.random.permutation(m)
    split = int(m * split_perc)
    val_idx, train_idx = indices[:split], indices[split:]
    return X[:, train_idx], X[:, val_idx], Y[:, train_idx], Y[:, val_idx]




## === cell 8
def JSON_to_array(file="train.json", split_perc=0.25):
    train_set = pd.read_json(path + file)

    inc_angle_raw = pd.to_numeric(train_set["inc_angle"], errors="coerce")
    median_angle = inc_angle_raw.median()
    inc_angle = inc_angle_raw.fillna(median_angle).values.reshape(-1, 1)

    band_1 = np.array(train_set["band_1"].tolist())
    band_2 = np.array(train_set["band_2"].tolist())

    b1_mean = band_1.mean(axis=1, keepdims=True)
    b1_std = band_1.std(axis=1, keepdims=True)
    b2_mean = band_2.mean(axis=1, keepdims=True)
    b2_std = band_2.std(axis=1, keepdims=True)

    diff_mean = (band_1 - band_2).mean(axis=1, keepdims=True)

    prod_mean = (band_1 * band_2).mean(axis=1, keepdims=True)

    X = np.concatenate(
        [
            band_1,
            band_2,
            b1_mean,
            b1_std,
            b2_mean,
            b2_std,
            diff_mean,
            prod_mean,
            inc_angle,
        ],
        axis=1,
    )

    feature_means = X.mean(axis=0, keepdims=True)
    feature_stds = X.std(axis=0, keepdims=True)
    feature_stds[feature_stds == 0] = 1.0  # avoid division by zero
    X = (X - feature_means) / feature_stds

    X = X.T  # (features, n_samples)

    Y = np.array(train_set["is_iceberg"]).reshape(1, -1)

    X_train, X_val, Y_train, Y_val = train_val_split(X, Y, split_perc)
    return (
        X_train,
        X_val,
        Y_train,
        Y_val,
        feature_means,
        feature_stds,
        median_angle,
    )




## === cell 9
X_tr, X_va, y_tr, y_va, f_means, f_stds, median_angle = JSON_to_array(
    file="train.json", split_perc=0.25
)

model_dict = model(
    X_tr,
    y_tr,
    X_va,
    y_va,
    num_iterations=5000,  # reduced iterations for speed
    learning_rate=0.01,  # slightly higher LR to keep convergence
    lambda_reg=0.01,
    print_cost=False,
)



## === cell 10
test_set = pd.read_json(path + "test.json")

inc_angle_test_raw = pd.to_numeric(test_set["inc_angle"], errors="coerce")
inc_angle_test = inc_angle_test_raw.fillna(median_angle).values.reshape(-1, 1)

band_1_test = np.array(test_set["band_1"].tolist())
band_2_test = np.array(test_set["band_2"].tolist())

b1_mean_test = band_1_test.mean(axis=1, keepdims=True)
b1_std_test = band_1_test.std(axis=1, keepdims=True)
b2_mean_test = band_2_test.mean(axis=1, keepdims=True)
b2_std_test = band_2_test.std(axis=1, keepdims=True)

diff_mean_test = (band_1_test - band_2_test).mean(axis=1, keepdims=True)

prod_mean_test = (band_1_test * band_2_test).mean(axis=1, keepdims=True)

X_test = np.concatenate(
    [
        band_1_test,
        band_2_test,
        b1_mean_test,
        b1_std_test,
        b2_mean_test,
        b2_std_test,
        diff_mean_test,
        prod_mean_test,
        inc_angle_test,
    ],
    axis=1,
)

X_test = (X_test - f_means) / f_stds
X_test = X_test.T  # (features, n_samples)

_, prob_test = predict(model_dict["w"], model_dict["b"], X_test)

submission = pd.DataFrame({"id": test_set["id"], "is_iceberg": prob_test.squeeze()})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
