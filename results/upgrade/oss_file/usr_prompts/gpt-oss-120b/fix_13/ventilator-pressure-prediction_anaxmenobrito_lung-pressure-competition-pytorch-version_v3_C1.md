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

6.18133

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.92904) has done: 'The changes keep the model and training logic identical but fix the huge slowdown caused by a batch size of 1 (the original code forced a common divisor of train‑ and test‑set sizes). We now set a reasonable batch size (256) directly and configure the DataLoaders for faster loading. Additionally, cuDNN benchmarking is enabled to let PyTorch choose the fastest convolution algorithms on GPU, which speeds up LSTM operations without altering results.'
- What this solution (achieved 1.70265) has done: 'I slightly increase model capacity, train longer, lower the learning rate, and normalize all numeric input features (R, C, time_step, u_in, u_out). These small adjustments keep the original architecture and training loop but should reduce the MAE toward the target value.'
- What this solution (achieved 1.79527) has done: 'I lower the optimizer learning rate and double the number of training epochs, keeping the same model and data processing. This modest change should improve convergence and reduce the validation MAE, moving the score closer to the target without altering the core architecture or any other logic.'
- What this solution (achieved 11.01609) has done: 'I enable model checkpointing so the best‑validation‑loss weights are used for the final predictions, and I give the model more training epochs (60 instead of 30) to let it converge further. These minimal tweaks keep the architecture and all other logic unchanged while expectedly lowering the MAE toward the target.'
- What this solution (achieved 12.18278) has done: 'The change only selects a GPU when one is available, keeping every model‑definition, training loop, and data preprocessing identical. Using the GPU dramatically reduces the per‑epoch compute time while preserving exact arithmetic and results, so the script now finishes well inside the 600 s limit.'
- What this solution (achieved 12.35073) has done: 'The fix ensures that the LSTM hidden states are always created on the same device as the input tensor, eliminating the mixed‑device error that halted training. No other logic is changed, preserving the original model architecture and training procedure while allowing the script to run to completion and produce a valid `submission.csv`.'
- What this solution (achieved 11.65364) has done: 'The bottleneck is the model training on CPU, which is very slow for 100 epochs over millions of rows. Switching to GPU (when available) provides a massive speed‑up while keeping the exact architecture, training loops, and data processing unchanged. The only modification is detecting CUDA and setting the device accordingly; all other logic remains identical.'
- What this solution (achieved 5.93068) has done: 'I fixed the device‑mismatch by explicitly moving the model to the chosen device, and I added proper pressure‑target normalization (with inverse‑scaling for evaluation and submission) so the network trains on a 0‑1 range, which markedly lowers the MAE toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 6.18133) has done: 'The fix ensures that the hidden‑state tensors inside the LSTM are always created on the same device as the model parameters (GPU if available), eliminating the runtime device‑mismatch error. No other logic is changed, so the model, training loop, and submission generation stay intact while the script now runs end‑to‑end and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

torch.backends.cudnn.benchmark = True




## === cell 2
datapath = "../input/ventilator-pressure-prediction"

train = pd.read_csv(datapath + "/train.csv")
test = pd.read_csv(datapath + "/test.csv")

print("Train Shape -> ", end="")
print(train.shape)
print(train.head())

print("\nTest Shape -> ", end="")
print(test.shape)
print(test.head())




## === cell 3
train = train[[col for col in train.columns if col not in {"id", "breath_id"}]]
train.head()




## === cell 4
def normalize_features(data_frame, columns):
    data = data_frame.copy()
    for col in columns:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data




## === cell 5
features_columns = ["R", "C", "time_step", "u_in", "u_out"]
normalized_train = normalize_features(train, features_columns)
normalized_train.head()




## === cell 6
target_column = "pressure"
pressure_min = normalized_train[target_column].min()
pressure_max = normalized_train[target_column].max()
raw_features = normalized_train[features_columns]
raw_targets = (normalized_train[[target_column]] - pressure_min) / (
    pressure_max - pressure_min
)




## === cell 7
N_TIME_STEPS_PER_EXAMPLE = 80
N_EXAMPLES = normalized_train.shape[0] // N_TIME_STEPS_PER_EXAMPLE
N_FEATURES = len(features_columns)

features = raw_features.to_numpy()
targets = raw_targets.to_numpy()

features = features.reshape(N_EXAMPLES, N_TIME_STEPS_PER_EXAMPLE, N_FEATURES)
targets = targets.reshape(N_EXAMPLES, N_TIME_STEPS_PER_EXAMPLE)

features.shape, targets.shape




## === cell 8
def add_time_diff(data: np.array):
    adding = np.zeros(data[:, :, 2].shape)
    adding[:, 1:] = data[:, :-1, 2]
    data[:, :, 2] -= adding
    return data


features = add_time_diff(features)
print("The difference between the actual time step and the previous:")
print(features[0, :, 2])




## === cell 9
features = torch.from_numpy(features)
targets = torch.from_numpy(targets)

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.11, random_state=42
)
features_train.shape, features_test.shape




## === cell 10
BATCH_SIZE = 256
print("Using batch size:", BATCH_SIZE)




## === cell 11
trainset = TensorDataset(features_train, targets_train)
testset = TensorDataset(features_test, targets_test)

trainLoader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
testLoader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 12
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

    def init_hidden(self, n_samples: int, device):
        h_t = torch.zeros(n_samples, self.n_hidden, device=device)
        c_t = torch.zeros(n_samples, self.n_hidden, device=device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, device=device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, device=device)

        return (h_t, c_t), (h_t2, c_t2)

    def forward(self, input_t, hidden_state=None, return_hidden_state: bool = False):
        """
        Forward pass that guarantees hidden states are on the same device as the model
        parameters, preventing CUDA/CPU mismatches.
        """
        outputs = []

        model_device = next(self.parameters()).device

        if hidden_state is not None:
            (h_t, c_t), (h_t2, c_t2) = hidden_state
            h_t, c_t = h_t.to(model_device), c_t.to(model_device)
            h_t2, c_t2 = h_t2.to(model_device), c_t2.to(model_device)
        else:
            (h_t, c_t), (h_t2, c_t2) = self.init_hidden(
                n_samples=input_t.size(0),
                device=model_device,
            )

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




## === cell 13
def evaluate(net: nn.Module, criterion, dataloader, device) -> float:
    net.eval()
    losses = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device).float()
            targets = targets.to(device).float()

            outputs = net(features)
            loss = criterion(outputs.flatten(), targets.flatten())
            losses.append(loss.item())

    net.train()
    return np.mean(losses)


def train(
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
    if save_min_loss_model:
        min_loss = float("inf")

    if "lrdecay" in kwargs:
        gamma = 0.5
        assert isinstance(kwargs["lrdecay"], int), "lrdecay must be int"
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=kwargs["lrdecay"], gamma=gamma
        )

    counter = 0
    net.to(device)
    net.train()

    for e in range(n_epochs):
        epoch_losses = []
        for features, targets in trainloader:
            optim.zero_grad()
            counter += 1

            features = features.to(device).float()
            targets = targets.to(device).float()

            outputs = net(features)
            loss = criterion(outputs.flatten(), targets.flatten())
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())

            if "lrdecay" in kwargs:
                scheduler.step()
                if counter % kwargs["lrdecay"] == 0:
                    lr = lr * gamma
                    print(f"Info: Reducing learning rate to {lr}")

            if counter % print_every == 0:
                val_loss = evaluate(net, criterion, testloader, device)
                print(
                    f"Epoch: {e+1}/{n_epochs}... Step: {counter}... "
                    f"Train Loss: {np.mean(epoch_losses):.4f}... Val Loss: {val_loss:.4f}"
                )

        val_loss = evaluate(net, criterion, testloader, device)
        print(
            f"\nEND Epoch {e+1}... Train Loss: {np.mean(epoch_losses):.4f}... Val Loss: {val_loss:.4f}"
        )

        if save_min_loss_model and val_loss < min_loss:
            name = (
                "best_model.pt" if "model_name" not in kwargs else kwargs["model_name"]
            )
            torch.save(net.cpu().state_dict(), name)
            print(f"Info: {name!r} saved (Min Valid Loss: {val_loss:.4f})")
            min_loss = val_loss
        print()




## === cell 14
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device is:", device, "(GPU will be used if available)\n")

model = LongShortTerm(input_size=5, n_hidden=64, linear_hidden=128, device=device)
model = model.to(device)  # ensure all parameters reside on the correct device
print(model)




## === cell 15
train(
    net=model,
    optim=torch.optim.Adam(model.parameters(), lr=0.0001),
    criterion=nn.L1Loss(),
    n_epochs=100,  # more epochs for better convergence
    trainloader=trainLoader,
    testloader=testLoader,
    print_every=500,
    device=device,
    save_min_loss_model=True,
    lrdecay=2000,
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3219064753.py in <cell line: 0>()
----> 1 train(
      2     net=model,
      3     optim=torch.optim.Adam(model.parameters(), lr=0.0001),
      4     criterion=nn.L1Loss(),
      5     n_epochs=100,  # more epochs for better convergence

/tmp/ipykernel_55/1794349517.py in train(net, optim, criterion, n_epochs, trainloader, testloader, print_every, device, save_min_loss_model, **kwargs)
     52             targets = targets.to(device).float()
     53 
---> 54             outputs = net(features)
     55             loss = criterion(outputs.flatten(), targets.flatten())
     56             loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2864485301.py in forward(self, input_t, hidden_state, return_hidden_state)
     46         for x in input_t.split(1, dim=1):
     47             x = x.view(-1, self.input_size)
---> 48             h_t, c_t = self.lstm1(x, (h_t, c_t))
     49             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
     50 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/rnn.py in forward(self, input, hx)
   1704             hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx
   1705 
-> 1706         ret = _VF.lstm_cell(
   1707             input,
   1708             hx,

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_mm)

## === cell 16
best_path = "best_model.pt"
if os.path.exists(best_path):
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state)
    print("Loaded best model from checkpoint.")
else:
    print("Best checkpoint not found; using final model state.")

saved_model = model.to(device)
saved_model.eval()




## === cell 17
with torch.no_grad():
    predictions = saved_model(features_test.to(device).float()).cpu().numpy()

predictions_rescaled = predictions * (pressure_max - pressure_min) + pressure_min
targets_rescaled = targets_test.numpy() * (pressure_max - pressure_min) + pressure_min

mse = mean_squared_error(targets_rescaled.flatten(), predictions_rescaled.flatten())
mae = mean_absolute_error(targets_rescaled.flatten(), predictions_rescaled.flatten())

print(f"MSE: {mse:.4f}... MAE: {mae:.4f}")




## === cell 18
k = np.random.randint(0, predictions_rescaled.shape[0])

plt.plot(targets_rescaled[k])
plt.plot(predictions_rescaled[k])
plt.legend(["Targets", "Predictions"])
plt.ylabel("Pressure")
plt.show()




## === cell 19
test.head()




## === cell 20
test = test[["R", "C", "time_step", "u_in", "u_out"]]




## === cell 21
test.head()




## === cell 22
norm_test = normalize_features(test, features_columns)




## === cell 23
norm_test.head()




## === cell 24
n_ex = test.shape[0] // N_TIME_STEPS_PER_EXAMPLE
n_features = 5

test_features = norm_test.to_numpy()
test_features = test_features.reshape(n_ex, N_TIME_STEPS_PER_EXAMPLE, n_features)
test_features.shape




## === cell 25
test_features = add_time_diff(test_features)
test_features[0, :, 2]




## === cell 26
with torch.no_grad():
    test_features = torch.from_numpy(test_features)
    saved_model.eval()
    predictions_test = saved_model(test_features.to(device).float()).cpu().numpy()
predictions_test = predictions_test * (pressure_max - pressure_min) + pressure_min




## === cell 27
plt.plot(predictions_test[132])
plt.show()




## === cell 28
sample_subs = pd.read_csv(datapath + "/sample_submission.csv")




## === cell 29
sample_subs.head()




## === cell 30
sample_subs.shape




## === cell 31
submission = pd.DataFrame(
    {"id": sample_subs.id, "pressure": predictions_test.flatten()}
)




## === cell 32
submission.head()




## === cell 33
submission.shape




## === cell 34
submission.to_csv("submission.csv", index=False)
