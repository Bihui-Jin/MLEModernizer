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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

1.3499225663798473

# 6. Current score

1.54352

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.61897) has done: 'I remove the batch-size divisibility assertion (it prevents the notebook from running but is not required for correctness), and make the notebook runnable in a non-interactive Kaggle script environment by removing the `%matplotlib inline` magic. The main score issue is that `pressure` (the target) is mistakenly taken from the *normalized* dataframe, which destroys the label scale and yields very poor MAE; I keep your feature normalization logic but take targets from the original (non-normalized) train dataframe. Finally, because the referenced pretrained model file is missing, I enable training (same model/loop) and then use the trained weights for inference and produce `submission.csv` in the required format.'
- What this solution (achieved 1.54359) has done: 'I keep your LSTMCell model and training loop intact, and make only metric-aligned fixes that should improve MAE toward your target. First, I change the split to be by `breath_id` (not random rows) to avoid leakage and to make validation reflect the Kaggle scoring setup more closely. Second, I ensure normalization uses train-derived min/max for `R` and `C` and applies the same scaling to test (your current per-file normalization can shift distributions and hurt generalization). Third, I compute validation MAE only on the inspiratory phase (`u_out == 0`) like the competition metric; training remains unchanged (MSE), but model selection be based on the correct validation MAE so the saved checkpoint is closer to the leaderboard objective.'
- What this solution (achieved 1.54352) has done: 'Your current MAE (1.54359) is worse than the target (1.3499), so we want a small, low-risk improvement without changing the model or training loop. The biggest score-aligned win with minimal logic change is to match the competition post-processing: the true pressures are on a discrete grid, so snapping predictions to the nearest allowed pressure value usually reduces MAE. I add a tiny “pressure grid” computation from the training set and apply a nearest-neighbor mapping to both validation metrics (for a fairer check) and the final test predictions, while keeping everything else (normalization, split-by-breath_id, model, optimizer, epochs) the same. This should move the score downward toward your target band without altering architecture/training semantics.'
- What this solution (achieved 1.54352) has done: 'We’re currently worse than the target (lower-is-better), so we want a small, low-risk improvement without changing your model/training loop. The most direct score-aligned fix is to make the training loss match the competition evaluation by ignoring expiratory timesteps (`u_out==1`) in the MSE loss; this keeps architecture/loop the same but stops the model from spending capacity on unscored regions. To do that safely with minimal disruption, I add a mask-aware MSE criterion (still MSE) and pass it into your existing `train_model` unchanged. Everything else (breath-wise split, min/max normalization stats, snapping to pressure grid, submission formatting) remains the same.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
datapath = "/kaggle/input/ventilator-pressure-prediction"

train = pd.read_csv(datapath + "/train.csv")
test = pd.read_csv(datapath + "/test.csv")

print("Train Shape -> ", train.shape)
print(train.head())

print("\nTest Shape -> ", test.shape)
print(test.head())



## === cell 2
train_wo_ids = train[[col for col in train.columns if col not in {"id", "breath_id"}]]
train_wo_ids.head()




## === cell 3
def fit_minmax(data_frame, columns):
    stats = {}
    for col in columns:
        stats[col] = (float(data_frame[col].min()), float(data_frame[col].max()))
    return stats


def normalize_features_with_stats(data_frame, columns, stats):
    data = data_frame.copy()
    for col in columns:
        cmin, cmax = stats[col]
        denom = cmax - cmin
        if denom == 0:
            data[col] = 0.0
        else:
            data[col] = (data[col] - cmin) / denom
    return data




## === cell 4
train_minmax_stats = fit_minmax(train_wo_ids, ["R", "C"])
normalized_train = normalize_features_with_stats(
    train_wo_ids, ["R", "C"], train_minmax_stats
)
normalized_train.head()



## === cell 5
features_columns = ["R", "C", "time_step", "u_in", "u_out"]
target_columns = ["pressure"]

raw_features = normalized_train[features_columns]
raw_targets = train_wo_ids[target_columns]



## === cell 6
N_TIME_STEPS_PER_EXAMPLE = 80
N_EXAMPLES = normalized_train.shape[0] // N_TIME_STEPS_PER_EXAMPLE
N_FEATURES = len(features_columns)

features = raw_features.to_numpy()
targets = raw_targets.to_numpy()

features = features.reshape(N_EXAMPLES, N_TIME_STEPS_PER_EXAMPLE, N_FEATURES)
targets = targets.reshape(N_EXAMPLES, N_TIME_STEPS_PER_EXAMPLE)

features.shape, targets.shape




## === cell 7
def add_time_diff(data: np.array):
    adding = np.zeros(data[:, :, 2].shape)
    adding[:, 1:] = data[:, :-1, 2]
    data[:, :, 2] -= adding
    return data


features = add_time_diff(features)
print("The difference between the actual time step and the previous:")
print(features[0, :, 2])



## === cell 8
breath_ids = (
    train["breath_id"].to_numpy().reshape(N_EXAMPLES, N_TIME_STEPS_PER_EXAMPLE)[:, 0]
)
unique_breath_ids = breath_ids.copy()

train_bids, val_bids = train_test_split(
    unique_breath_ids, test_size=0.11, random_state=SEED, shuffle=True
)
train_bids = set(train_bids.tolist())
val_bids = set(val_bids.tolist())

idx_train = np.array([bid in train_bids for bid in breath_ids])
idx_val = np.array([bid in val_bids for bid in breath_ids])

features_t = torch.from_numpy(features)
targets_t = torch.from_numpy(targets)

features_train = features_t[idx_train]
targets_train = targets_t[idx_train]
features_test = features_t[idx_val]
targets_test = targets_t[idx_val]

val_insp_mask = features_test[:, :, 4] == 0

features_train.shape, features_test.shape




## === cell 9
def mult(x) -> set:
    return set([i for i in range(2, x) if x % i == 0])


pbs = mult(features_train.size(0)).intersection(mult(features_test.size(0)))
print("Possible Batch Sizes (intersection divisors):", sorted(list(pbs))[:30], "...")



## === cell 10
BATCH_SIZE = 128

trainset = TensorDataset(features_train, targets_train)
testset = TensorDataset(features_test, targets_test)

trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
testLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)




## === cell 11
class LongShortTerm(nn.Module):

    def __init__(
        self, input_size, n_hidden: int = 16, linear_hidden: int = 32, device="cpu"
    ):
        super(LongShortTerm, self).__init__()
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.device = device

        self.lstm1 = nn.LSTMCell(self.input_size, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.fc1 = nn.Linear(self.n_hidden, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, 1)

    def init_hidden(self, n_samples: int):
        h_t = torch.zeros(n_samples, self.n_hidden, device=self.device)
        c_t = torch.zeros(n_samples, self.n_hidden, device=self.device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, device=self.device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, device=self.device)
        return (h_t, c_t), (h_t2, c_t2)

    def forward(self, input_t, hidden_state=None, return_hidden_state: bool = False):
        outputs = []

        if hidden_state:
            (h_t, c_t), (h_t2, c_t2) = hidden_state
        else:
            (h_t, c_t), (h_t2, c_t2) = self.init_hidden(n_samples=input_t.size(0))

        for x in input_t.split(1, dim=1):
            x = x.view(-1, self.input_size)
            h_t, c_t = self.lstm1(x, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            output = self.fc1(h_t2)
            output = self.fc2(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)

        if not return_hidden_state:
            return outputs
        else:
            return outputs, (h_t, c_t), (h_t2, c_t2)




## === cell 12
pressure_grid = np.sort(train["pressure"].unique().astype(np.float32))
pressure_grid_t = torch.from_numpy(
    pressure_grid
)  # kept on CPU; moved to device when needed


def snap_to_pressure_grid_np(pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    pred_flat = pred.reshape(-1).astype(np.float32)
    idx = np.searchsorted(grid, pred_flat, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    idx0 = np.clip(idx - 1, 0, len(grid) - 1)
    left = grid[idx0]
    right = grid[idx]
    choose_right = np.abs(right - pred_flat) <= np.abs(pred_flat - left)
    snapped = np.where(choose_right, right, left)
    return snapped.reshape(pred.shape)


def snap_to_pressure_grid_torch(
    pred_t: torch.Tensor, grid_t: torch.Tensor
) -> torch.Tensor:
    grid_t = grid_t.to(pred_t.device)
    pred_flat = pred_t.reshape(-1)
    idx = torch.searchsorted(grid_t, pred_flat)
    idx = torch.clamp(idx, 0, grid_t.numel() - 1)
    idx0 = torch.clamp(idx - 1, 0, grid_t.numel() - 1)
    left = grid_t[idx0]
    right = grid_t[idx]
    choose_right = torch.abs(right - pred_flat) <= torch.abs(pred_flat - left)
    snapped = torch.where(choose_right, right, left)
    return snapped.reshape(pred_t.shape)




## === cell 13
class InspiratoryMaskedMSELoss(nn.Module):
    def __init__(self, u_out_feature_index: int = 4, eps: float = 1e-12):
        super().__init__()
        self.u_out_feature_index = u_out_feature_index
        self.eps = eps

    def forward(
        self, preds_flat: torch.Tensor, targets_flat: torch.Tensor
    ) -> torch.Tensor:
        features_local = globals().get("features_b", None)
        if features_local is None:
            return torch.mean((preds_flat - targets_flat) ** 2)

        mask = (features_local[:, :, self.u_out_feature_index] == 0).reshape(-1)
        mask = mask.to(preds_flat.device)

        diff2 = (preds_flat - targets_flat) ** 2
        masked_sum = (diff2 * mask).sum()
        denom = mask.sum().clamp_min(
            1.0
        )  # safe if a batch had no inspiratory steps (shouldn't happen)
        return masked_sum / denom




## === cell 14
def evaluate(net: nn.Module, criterion, dataloader, device) -> float:
    net.eval()
    losses = []

    with torch.no_grad():
        for features_b, targets_b in dataloader:
            features_b = features_b.to(device).float()
            targets_b = targets_b.to(device).float()

            outputs = net(features_b)
            loss = criterion(outputs.flatten(), targets_b.flatten())
            losses.append(loss.item())

    net.train()
    return float(np.mean(losses)) if len(losses) else float("nan")


def evaluate_mae_insp(net: nn.Module, dataloader, device) -> float:
    net.eval()
    abs_errors = []
    counts = 0

    with torch.no_grad():
        for features_b, targets_b in dataloader:
            features_b = features_b.to(device).float()
            targets_b = targets_b.to(device).float()
            outputs_b = net(features_b)  # (B, 80)

            outputs_b = snap_to_pressure_grid_torch(outputs_b, pressure_grid_t)

            mask = features_b[:, :, 4] == 0
            diff = torch.abs(outputs_b - targets_b)

            masked_diff = diff[mask]
            if masked_diff.numel() > 0:
                abs_errors.append(masked_diff.sum().item())
                counts += masked_diff.numel()

    net.train()
    if counts == 0:
        return float("nan")
    return float(np.sum(abs_errors) / counts)


def train_model(
    net: nn.Module,
    optim,
    criterion,
    n_epochs,
    trainloader,
    testloader,
    print_every,
    device,
    save_min_loss_model: bool = False,
    **kwargs,
):
    lr = 0.03
    if save_min_loss_model is True:
        min_metric = float("inf")

    if "lrdecay" in kwargs:
        gamma = 0.5
        assert isinstance(
            kwargs["lrdecay"], int
        ), "Number of steps to take before reducing the learning rate must be int!"
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=kwargs["lrdecay"], gamma=gamma
        )

    counter = 0
    net.train()
    net.to(device)

    for e in range(n_epochs):
        for features_b, targets_b in trainloader:
            optim.zero_grad()
            counter += 1

            features_b = features_b.to(device).float()
            targets_b = targets_b.to(device).float()

            losses = []

            def closure():
                outputs = net(features_b)
                loss = criterion(outputs.flatten(), targets_b.flatten())
                loss.backward()
                losses.append(loss.item())
                return loss

            optim.step(closure)

            if "lrdecay" in kwargs:
                scheduler.step()
                if counter % kwargs["lrdecay"] == 0:
                    lr = lr * gamma
                    print(f"Info: Reducting learning rate to {lr}")

            if counter % print_every == 0:
                val_mse = evaluate(net, criterion, testloader, device)
                val_mae_insp = evaluate_mae_insp(net, testloader, device)
                print(
                    f"Epoch: {e+1}/{n_epochs}... Step: {counter}... Loss(MSE): {np.mean(losses):.4f}... "
                    f"Val MSE: {val_mse:.4f}... Val MAE(insp): {val_mae_insp:.4f}"
                )
                net.train()

        val_mse = evaluate(net, criterion, testloader, device)
        val_mae_insp = evaluate_mae_insp(net, testloader, device)
        print(
            f"\nEND Epoch: {e+1}... Loss(MSE): {np.mean(losses):.4f}... Valid MSE: {val_mse:.4f}... Valid MAE(insp): {val_mae_insp:.4f}"
        )

        if save_min_loss_model is True and val_mae_insp < min_metric:
            name = "model.pt" if "model_name" not in kwargs else kwargs["model_name"]
            torch.save(net.cpu().state_dict(), name)
            net.to(device)
            print(f"Info: {name!r} was saved (Min Valid MAE(insp): {val_mae_insp:.4f})")
            min_metric = val_mae_insp
        print()




## === cell 15
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device is:", device, "(CUDA is recommended for faster training!)\n")

model = LongShortTerm(5, 5, 100, device)
print(model)



## === cell 16
train_model(
    net=model,
    optim=torch.optim.Adam(model.parameters(), lr=0.03),
    criterion=InspiratoryMaskedMSELoss(u_out_feature_index=4),
    n_epochs=15,
    trainloader=trainLoader,
    testloader=testLoader,
    print_every=500,
    device=device,
    save_min_loss_model=True,
    lrdecay=10000,
    model_name="model.pt",
)



## === cell 17
saved_model = LongShortTerm(5, 5, 100, device).to(device)
if os.path.exists("model.pt"):
    state_dict = torch.load("model.pt", map_location=device)
    saved_model.load_state_dict(state_dict)
saved_model



## === cell 18
with torch.no_grad():
    saved_model.eval()
    predictions_val = saved_model(features_test.to(device).float()).cpu().numpy()

predictions_val_snapped = snap_to_pressure_grid_np(predictions_val, pressure_grid)

mse = mean_squared_error(targets_test.flatten().numpy(), predictions_val.flatten())
mae = mean_absolute_error(targets_test.flatten().numpy(), predictions_val.flatten())
mse_s = mean_squared_error(
    targets_test.flatten().numpy(), predictions_val_snapped.flatten()
)
mae_s = mean_absolute_error(
    targets_test.flatten().numpy(), predictions_val_snapped.flatten()
)

with torch.no_grad():
    pred_val_t = torch.from_numpy(predictions_val_snapped)
true_val_t = targets_test.float().cpu()
mask_val = val_insp_mask.cpu()
mae_insp = torch.abs(pred_val_t - true_val_t)[mask_val].mean().item()

print(
    f"MSE (all timesteps): {mse:.4f}... MAE (all timesteps): {mae:.4f} | "
    f"SNAPPED -> MSE: {mse_s:.4f}... MAE: {mae_s:.4f}... MAE (insp only, snapped): {mae_insp:.4f}"
)



## === cell 19
k = np.random.randint(0, predictions_val.shape[0])
plt.figure(figsize=(10, 4))
plt.plot(targets_test[k].numpy())
plt.plot(predictions_val[k])
plt.plot(predictions_val_snapped[k])
plt.legend(["Targets", "Predictions", "Predictions (snapped)"])
plt.ylabel("Pressure")
plt.show()



## === cell 20
test.head()



## === cell 21
test_feats_df = test[["R", "C", "time_step", "u_in", "u_out"]]



## === cell 22
test_feats_df.head()



## === cell 23
norm_test = normalize_features_with_stats(test_feats_df, ["R", "C"], train_minmax_stats)



## === cell 24
norm_test.head()



## === cell 25
n_ex = test.shape[0] // N_TIME_STEPS_PER_EXAMPLE
n_features = 5

test_features = norm_test.to_numpy()
test_features = test_features.reshape(n_ex, N_TIME_STEPS_PER_EXAMPLE, n_features)
test_features.shape



## === cell 26
test_features = add_time_diff(test_features)
test_features[0, :, 2]



## === cell 27
with torch.no_grad():
    test_features_t = torch.from_numpy(test_features)
    saved_model.eval()
    predictions = saved_model(test_features_t.to(device).float()).cpu().numpy()

predictions = snap_to_pressure_grid_np(predictions, pressure_grid)



## === cell 28
plt.figure(figsize=(10, 3))
plt.plot(predictions[132])
plt.show()



## === cell 29
sample_subs = pd.read_csv(datapath + "/sample_submission.csv")



## === cell 30
sample_subs.head()



## === cell 31
sample_subs.shape



## === cell 32
submission = pd.DataFrame(
    {"id": sample_subs["id"].values, "pressure": predictions.flatten()}
)



## === cell 33
submission.head()



## === cell 34
submission.shape



## === cell 35
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
