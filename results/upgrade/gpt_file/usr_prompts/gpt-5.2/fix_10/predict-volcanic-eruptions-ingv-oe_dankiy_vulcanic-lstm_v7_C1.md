# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5793085.557252489

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 24402200.0) has done: 'The timeout is dominated by slow per-file feature extraction (4k train + 444 test CSVs) and an extremely long 2000‑epoch training loop that also re-runs full validation every epoch. I keep the exact same features, model, optimizer, loss, and epoch count, but make them run faster by (1) extracting features in vectorized NumPy from `float32` arrays (no per-column Pandas stats in Python loops), (2) parallelizing feature extraction across CPU cores, (3) making DataLoaders use pinned memory + nonblocking GPU transfers, and (4) evaluating validation MAE in a streaming way (no concatenation) to cut overhead while preserving identical MAE. I also enable deterministic/cuDNN settings and `torch.inference_mode()` for prediction/eval for additional speed without changing semantics.'
- What this solution (achieved 24402200.0) has done: 'I fix the crash by removing the hard requirement for deterministic CUDA algorithms (it triggers a CuBLAS non-determinism error at the Linear layer) while keeping the same seeding and cuDNN settings for stability. I also make the environment consistent with Kaggle’s Python 3.9 by not relying on the error-trace paths, and ensure the training loop runs end-to-end to produce `submission.csv`. No model architecture, features, loss, optimizer, batch size, or epoch count are changed, so this should be score-neutral aside from negligible floating-point nondeterminism. Finally, I keep the submission format aligned to `sample_submission.csv` (`segment_id,time_to_eruption`).'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn import preprocessing




## === cell 2
def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    if normalize_opt is not None:
        if normalize_opt == "min_max":
            scaler = preprocessing.MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalize_opt={normalize_opt}")
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_valid, X_test




## === cell 3
PATH_DATA = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/"
TRAIN_META_PATH = os.path.join(PATH_DATA, "train.csv")
TRAIN_DIR = os.path.join(PATH_DATA, "train")
TEST_DIR = os.path.join(PATH_DATA, "test")

train_meta = pd.read_csv(TRAIN_META_PATH)
submission = pd.read_csv(os.path.join(PATH_DATA, "sample_submission.csv"))

sensor_cols = [f"sensor_{i}" for i in range(1, 11)]


def extract_segment_features_fast(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=sensor_cols)
    arr = df.to_numpy(copy=False).astype(np.float32, copy=False)

    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)

    out = np.empty(40, dtype=np.float32)
    for j in range(10):
        k = 4 * j
        out[k + 0] = means[j]
        out[k + 1] = stds[j]
        out[k + 2] = mins[j]
        out[k + 3] = maxs[j]
    return out


from concurrent.futures import ThreadPoolExecutor, as_completed


def build_features_matrix(seg_ids, base_dir, max_workers=None):
    paths = [os.path.join(base_dir, f"{sid}.csv") for sid in seg_ids]
    n = len(paths)
    feats = np.empty((n, 40), dtype=np.float32)
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(extract_segment_features_fast, p): i for i, p in enumerate(paths)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            feats[i] = fut.result()
    return feats


train_ids = train_meta["segment_id"].values
test_ids = submission["segment_id"].values

train_features = build_features_matrix(train_ids, TRAIN_DIR)
train_sample = pd.DataFrame(
    train_features, columns=[f"f_{i}" for i in range(train_features.shape[1])]
)
targets = train_meta[["time_to_eruption"]].copy()

test_features = build_features_matrix(test_ids, TEST_DIR)
test = pd.DataFrame(test_features, columns=train_sample.columns)

gc.collect()



## === cell 4
y_for_strat = targets["time_to_eruption"].values
bins = pd.qcut(y_for_strat, q=20, labels=False, duplicates="drop")

train_x, valid_x, train_y, valid_y = train_test_split(
    train_sample, targets, test_size=0.2, random_state=0, stratify=bins
)

train_x, valid_x, test_scaled = normalize(
    train_x.copy(), valid_x.copy(), test.copy(), "min_max", []
)

train_x = train_x.values.reshape(train_x.shape[0], train_x.shape[1], 1).astype(
    np.float32
)
valid_x = valid_x.values.reshape(valid_x.shape[0], valid_x.shape[1], 1).astype(
    np.float32
)
train_y = train_y.to_numpy().astype(np.float32)
valid_y = valid_y.to_numpy().astype(np.float32)
test_scaled = test_scaled.values.reshape(
    test_scaled.shape[0], test_scaled.shape[1], 1
).astype(np.float32)



## === cell 5
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 6
from torch.optim.optimizer import Optimizer


class Nadam(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                m_nesterov = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1**t)

                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_nesterov, denom, value=-lr)

        return loss




## === cell 7
NUM_MODELS = 1
BATCH_SIZE = 8192
NUM_EPOCHS = 2000
PATH_MODEL = "/kaggle/working/models/"
os.makedirs(PATH_MODEL, exist_ok=True)




## === cell 8
class VolcanicLSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_features=num_features)

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=128, num_layers=1, batch_first=True
        )

        self.conv1 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=2, padding=1, stride=2
        )
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=84, kernel_size=2, padding=1, stride=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=84, out_channels=64, kernel_size=2, padding=1, stride=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(in_features=64, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        x = x.transpose(1, 2)  # (batch, 1, seq_len)
        x = self.bn(x.squeeze(1)).unsqueeze(
            1
        )  # BN1d expects (batch, C); normalize C=seq_len
        x = x.transpose(1, 2)  # back to (batch, seq_len, 1)

        x, _ = self.lstm(x)  # (batch, seq_len, 128)

        x = x.permute(0, 2, 1)  # (batch, 128, seq_len)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.pool(x)  # (batch, 64, 1)
        x = self.flat(x)  # (batch, 64)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)

        return x




## === cell 9
train_idx = np.arange(train_x.shape[0], dtype=np.int64)
valid_idx = np.arange(valid_x.shape[0], dtype=np.int64)
test_idx = np.arange(test_scaled.shape[0], dtype=np.int64)

train_ds = TensorDataset(
    torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_idx)
)
valid_ds = TensorDataset(
    torch.from_numpy(valid_x), torch.from_numpy(valid_y), torch.from_numpy(valid_idx)
)
test_ds = TensorDataset(torch.from_numpy(test_scaled), torch.from_numpy(test_idx))

pin = device.type == "cuda"
num_workers = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=True if num_workers > 0 else False,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=True if num_workers > 0 else False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=True if num_workers > 0 else False,
)


def evaluate_mae(model, loader):
    model.eval()
    abs_sum = 0.0
    n = 0
    with torch.inference_mode():
        for xb, yb, _idx in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = model(xb)
            abs_sum += (pb - yb).abs().sum().item()
            n += yb.numel()
    return abs_sum / max(n, 1)


def predict_with_index(model, loader):
    model.eval()
    preds_all = []
    idx_all = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                xb, _yb, idx = batch
            else:
                xb, idx = batch
            xb = xb.to(device, non_blocking=True)
            pb = model(xb).detach().reshape(-1).cpu().numpy()
            preds_all.append(pb)
            idx_all.append(idx.numpy())
    return np.concatenate(preds_all, axis=0), np.concatenate(idx_all, axis=0)




## === cell 10
sub_final = np.zeros(len(submission), dtype=np.float64)
oof_final = np.zeros(len(valid_x), dtype=np.float64)

for i in range(NUM_MODELS):
    print("Running model", i + 1)

    seed_all(0 + i)

    model = VolcanicLSTM(num_features=train_x.shape[1]).to(device)
    optimizer = Nadam(model.parameters(), lr=0.005)

    best_val = float("inf")
    best_path = os.path.join(PATH_MODEL, f"best_epoch-{i+1}.pt")

    torch.save(model.state_dict(), best_path)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for xb, yb, _idx in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = F.l1_loss(preds, yb)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_mae(model, valid_loader)

        if np.isfinite(val_loss) and (val_loss < best_val):
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        if epoch in (1, 10, 50, 100, 200, 500, 1000, 1500, 2000):
            print(f"epoch {epoch:4d} | val_mae {val_loss:.4f} | best {best_val:.4f}")

    best_model = VolcanicLSTM(num_features=train_x.shape[1]).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    oof_pred, oof_idx = predict_with_index(best_model, valid_loader)
    oof_fold = np.empty_like(oof_final)
    oof_fold[oof_idx] = oof_pred

    test_pred, test_idx_out = predict_with_index(best_model, test_loader)
    test_fold = np.empty_like(sub_final)
    test_fold[test_idx_out] = test_pred

    oof_final += oof_fold
    sub_final += test_fold



## === cell 11
oof_final /= NUM_MODELS
sub_final /= NUM_MODELS

mae = np.mean(np.abs(valid_y.reshape(-1) - oof_final))
print(f"\nValidation MAE: {mae:.0f}")

submission["time_to_eruption"] = sub_final
submission.to_csv("submission.csv", index=False)

print("\nWrote submission.csv with shape:", submission.shape)
print(submission.head())
