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
Predict the formation energy and bandgap energy of a material.

## Metric
Column-wise root mean squared logarithmic error.

## Submission Format
For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev. The file should contain a header and have the following format:
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
3,0.1779,1.8892
...
```

## Dataset
The following information has been included:

- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ($\N_{total}$)
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ($10^{-10}$ meters) and $\alpha, \beta, \gamma$ (which are angles in degrees between 0° and 360°)

Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

# 2. Python version

3.6

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        input/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        working/
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
```

-> data/nomad2018-predict-transparent-conductors/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/nomad2018-predict-transparent-conductors/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/nomad2018-predict-transparent-conductors/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> data/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> (stopped after 10 files for performance)

# 5. Target score

0.07851

# 6. Current score

0.12425

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21153) has done: 'I fix the TensorFlow import/runtime failure by avoiding the broken TF1 graph + `tf.compat.v1.layers.*` path (it’s incompatible with Keras 3 and also triggers a protobuf-related crash). I keep the same core model idea (batch norm on inputs, then an 8-layer tanh MLP with 16 units, two separate heads, Adam optimizer, RMSE on log1p targets) but implement it with `tf.keras` so it runs in the current Kaggle environment. I also ensure batch-norm updates are applied correctly (via the `training` flag) and keep determinism with fixed random seeds. Finally, I write a valid `subm.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.11297) has done: 'I fix the TensorFlow import crash by forcing a compatible protobuf runtime before importing TensorFlow (the current error comes from an incompatible protobuf C++/python API mismatch). Then I keep your exact model/training logic intact, but also add a robust fallback that uses `tensorflow-cpu` if available and otherwise forces the pure-Python protobuf implementation early enough to take effect. Finally, I ensure the submission is always written with the required `.csv` suffix and exact column order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.12389) has done: 'I fix the TensorFlow import/runtime failure by switching to a stable, TF-free fallback implementation that preserves your core modeling idea: an input batch-normalization followed by an 8-layer tanh MLP (16 units) with two separate heads trained on log1p targets. This removes the protobuf/TensorFlow incompatibility causing `MessageFactory.GetPrototype` errors while keeping the same features, targets, loss (RMSE on log1p), and training loop semantics. To move the score toward your target, I also train on the full training set (no holdout) after confirming the pipeline works, because your current split sacrifices data and typically hurts the leaderboard score for this small dataset. Finally, I ensure the submission is written as a valid `subm.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.12425) has done: 'Your current score (0.12389, lower-is-better) is still above the target (0.07851), so we should cautiously improve it with minimal, metric-aligned changes while keeping your NumPy MLP and training loop intact. The largest issue is that your internal training loss is RMSE on log1p targets, but the competition metric is RMSLE on original targets; aligning the loss to RMSLE (while still predicting in log1p space) is a minimal semantic correction that typically improves leaderboard score without changing architecture. To make RMSLE well-defined, we also apply the same non-negativity clamp to training targets that you already apply at inference, preventing log1p of values < -1 and matching the metric’s domain. Finally, we keep the full-data training approach but make the batch-norm “inference” behavior consistent by running one final forward pass over the full training set in training mode to stabilize running mean/variance before predicting the test set.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



## === cell 2
train.describe()



## === cell 3
train.head()



## === cell 4
np.all(
    np.abs(
        train.loc[:, "percent_atom_al"]
        + train.loc[:, "percent_atom_ga"]
        + train.loc[:, "percent_atom_in"]
        - 1
    )
    <= 0.001
)



## === cell 5
train = train.drop(["spacegroup", "percent_atom_in"], axis=1)
test = test.drop(["spacegroup", "percent_atom_in"], axis=1)



## === cell 6
from sklearn.model_selection import train_test_split

t1 = "formation_energy_ev_natom"
t2 = "bandgap_energy_ev"

X_train, X_validation = train_test_split(train, test_size=0.3, random_state=1)

y1_train_raw = X_train[t1].to_numpy()[:, np.newaxis].astype(np.float32)
y2_train_raw = X_train[t2].to_numpy()[:, np.newaxis].astype(np.float32)
y1_train_raw[y1_train_raw < 0] = 0.0
y2_train_raw[y2_train_raw < 0] = 0.0
y1_train = np.log1p(y1_train_raw).astype(np.float32)
y2_train = np.log1p(y2_train_raw).astype(np.float32)
X_train = X_train.drop(["id", t1, t2], axis=1)

y1_validation_raw = X_validation[t1].to_numpy()[:, np.newaxis].astype(np.float32)
y2_validation_raw = X_validation[t2].to_numpy()[:, np.newaxis].astype(np.float32)
y1_validation_raw[y1_validation_raw < 0] = 0.0
y2_validation_raw[y2_validation_raw < 0] = 0.0
y1_validation = np.log1p(y1_validation_raw).astype(np.float32)
y2_validation = np.log1p(y2_validation_raw).astype(np.float32)
X_validation = X_validation.drop(["id", t1, t2], axis=1)

X_train = X_train.to_numpy(dtype=np.float32)
X_validation = X_validation.to_numpy(dtype=np.float32)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)



## === cell 7
import matplotlib.pyplot as plt



## === cell 8
plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)

plt.show()



## === cell 9
import math

np.random.seed(1)

print("Using NumPy-only MLP fallback (no TensorFlow).")



## === cell 10
n_units = 16
n_layers = 8
learning_rate = 0.004
n_steps = 500

input_dim = X_train.shape[1]




## === cell 11
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.99):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
        self.running_mean = np.zeros((1, dim), dtype=np.float32)
        self.running_var = np.ones((1, dim), dtype=np.float32)

        self.xc = None
        self.std_inv = None
        self.xhat = None

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x, training=False):
        x = x.astype(np.float32, copy=False)
        if training:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True)
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            xc = x - mu
            std_inv = 1.0 / np.sqrt(var + self.eps)
            xhat = xc * std_inv

            self.xc = xc
            self.std_inv = std_inv
            self.xhat = xhat
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * xhat + self.beta
        return out

    def backward(self, dout):
        N = dout.shape[0]
        self.dbeta = dout.sum(axis=0, keepdims=True)
        self.dgamma = (dout * self.xhat).sum(axis=0, keepdims=True)

        dxhat = dout * self.gamma
        dvar = (dxhat * self.xc * -0.5 * (self.std_inv**3)).sum(axis=0, keepdims=True)
        dmu = (dxhat * -self.std_inv).sum(axis=0, keepdims=True) + dvar * (
            -2.0 * self.xc
        ).mean(axis=0, keepdims=True)
        dx = dxhat * self.std_inv + dvar * (2.0 * self.xc) / N + dmu / N
        return dx.astype(np.float32)


class Dense:
    def __init__(self, in_dim, out_dim, activation=None):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, size=(in_dim, out_dim)).astype(
            np.float32
        )
        self.b = np.zeros((1, out_dim), dtype=np.float32)
        self.activation = activation

        self.x = None
        self.z = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        self.z = z
        if self.activation is None:
            return z
        if self.activation == "tanh":
            return np.tanh(z).astype(np.float32)
        raise ValueError("Unknown activation")

    def backward(self, dout):
        if self.activation == "tanh":
            dz = dout * (1.0 - np.tanh(self.z) ** 2)
        else:
            dz = dout

        self.dW = (self.x.T @ dz).astype(np.float32)
        self.db = dz.sum(axis=0, keepdims=True).astype(np.float32)
        dx = (dz @ self.W.T).astype(np.float32)
        return dx


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def _key(self, name):
        return name

    def step(self):
        self.t += 1
        for name, p, g in self.params:
            if g is None:
                continue
            k = self._key(name)
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            mhat = self.m[k] / (1 - self.beta1**self.t)
            vhat = self.v[k] / (1 - self.beta2**self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class TwoHeadMLP_NP:
    def __init__(self, input_dim, n_units, n_layers):
        self.bn_in = BatchNorm1D(input_dim, eps=1e-5, momentum=0.99)

        self.y1_dense = []
        self.y1_bn = []
        dim = input_dim
        for _ in range(n_layers):
            self.y1_dense.append(Dense(dim, n_units, activation="tanh"))
            self.y1_bn.append(BatchNorm1D(n_units, eps=1e-5, momentum=0.99))
            dim = n_units
        self.out_y1 = Dense(dim, 1, activation=None)

        self.y2_dense = []
        self.y2_bn = []
        dim = input_dim
        for _ in range(n_layers):
            self.y2_dense.append(Dense(dim, n_units, activation="tanh"))
            self.y2_bn.append(BatchNorm1D(n_units, eps=1e-5, momentum=0.99))
            dim = n_units
        self.out_y2 = Dense(dim, 1, activation=None)

    def forward(self, x, training=False):
        x = self.bn_in.forward(x, training=training)

        h1 = x
        for d, b in zip(self.y1_dense, self.y1_bn):
            h1 = d.forward(h1)
            h1 = b.forward(h1, training=training)
        y1 = self.out_y1.forward(h1)

        h2 = x
        for d, b in zip(self.y2_dense, self.y2_bn):
            h2 = d.forward(h2)
            h2 = b.forward(h2, training=training)
        y2 = self.out_y2.forward(h2)
        return y1, y2

    def params_with_grads(self):
        items = []
        items.append(("bn_in_gamma", self.bn_in.gamma, self.bn_in.dgamma))
        items.append(("bn_in_beta", self.bn_in.beta, self.bn_in.dbeta))
        for i, d in enumerate(self.y1_dense):
            items.append((f"y1_W_{i}", d.W, d.dW))
            items.append((f"y1_b_{i}", d.b, d.db))
        for i, b in enumerate(self.y1_bn):
            items.append((f"y1_bn_gamma_{i}", b.gamma, b.dgamma))
            items.append((f"y1_bn_beta_{i}", b.beta, b.dbeta))
        items.append(("y1_out_W", self.out_y1.W, self.out_y1.dW))
        items.append(("y1_out_b", self.out_y1.b, self.out_y1.db))
        for i, d in enumerate(self.y2_dense):
            items.append((f"y2_W_{i}", d.W, d.dW))
            items.append((f"y2_b_{i}", d.b, d.db))
        for i, b in enumerate(self.y2_bn):
            items.append((f"y2_bn_gamma_{i}", b.gamma, b.dgamma))
            items.append((f"y2_bn_beta_{i}", b.beta, b.dbeta))
        items.append(("y2_out_W", self.out_y2.W, self.out_y2.dW))
        items.append(("y2_out_b", self.out_y2.b, self.out_y2.db))
        return items

    def backward(self, x, dy1, dy2):
        dh1 = self.out_y1.backward(dy1)
        for d, b in zip(reversed(self.y1_dense), reversed(self.y1_bn)):
            dh1 = b.backward(dh1)
            dh1 = d.backward(dh1)

        dh2 = self.out_y2.backward(dy2)
        for d, b in zip(reversed(self.y2_dense), reversed(self.y2_bn)):
            dh2 = b.backward(dh2)
            dh2 = d.backward(dh2)

        dxbn = dh1 + dh2
        _ = self.bn_in.backward(dxbn)


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))




## === cell 12
train_full = train.copy()

X_full = train_full.drop(["id", t1, t2], axis=1).to_numpy(dtype=np.float32)

y1_full_raw = train_full[t1].to_numpy()[:, np.newaxis].astype(np.float32)
y2_full_raw = train_full[t2].to_numpy()[:, np.newaxis].astype(np.float32)
y1_full_raw[y1_full_raw < 0] = 0.0
y2_full_raw[y2_full_raw < 0] = 0.0
y1_full = np.log1p(y1_full_raw).astype(np.float32)
y2_full = np.log1p(y2_full_raw).astype(np.float32)

model = TwoHeadMLP_NP(input_dim=input_dim, n_units=n_units, n_layers=n_layers)
optimizer = Adam(model.params_with_grads(), lr=learning_rate)

loss_data = []
for step in range(n_steps):
    pred1_tr, pred2_tr = model.forward(X_full, training=True)

    eps = 1e-8
    diff1 = (pred1_tr - y1_full).astype(np.float32)
    diff2 = (pred2_tr - y2_full).astype(np.float32)
    rmsle1 = np.sqrt(np.mean(diff1**2) + eps).astype(np.float32)
    rmsle2 = np.sqrt(np.mean(diff2**2) + eps).astype(np.float32)
    loss = float((rmsle1 + rmsle2) / 2.0)

    N = X_full.shape[0]
    dy1 = (diff1 / (N * rmsle1)).astype(np.float32) * 0.5
    dy2 = (diff2 / (N * rmsle2)).astype(np.float32) * 0.5

    model.backward(X_full, dy1, dy2)
    optimizer.params = model.params_with_grads()
    optimizer.step()

    if step % 50 == 0 or step == n_steps - 1:
        pred1_va, pred2_va = model.forward(X_validation, training=False)
        loss_va = (
            rmse_np(y1_validation, pred1_va) + rmse_np(y2_validation, pred2_va)
        ) / 2.0
        loss_data.append([step, loss, float(loss_va)])

loss_data = np.array(loss_data, dtype=np.float64)
print("logged checkpoints (step, train_loss, val_loss):")
print(loss_data[-5:])

plt.plot(loss_data[:, 0], loss_data[:, 1], "r-")
plt.plot(loss_data[:, 0], loss_data[:, 2], "b-")
plt.show()



## === cell 13
pred_y1_va, pred_y2_va = model.forward(X_validation, training=False)
val_loss = (
    rmse_np(y1_validation, pred_y1_va) + rmse_np(y2_validation, pred_y2_va)
) / 2.0
print("loss:", val_loss)

m_y1 = max(float(y1_validation.max()), float(pred_y1_va.max()))
ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(len(pred_y1_va)), pred_y1_va, c="red")

m_y2 = max(float(y2_validation.max()), float(pred_y2_va.max()))
ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(len(pred_y2_va)), pred_y2_va, c="red")

plt.show()



## === cell 14
sample = pd.read_csv("../input/sample_submission.csv")
sample.head()



## === cell 15
X_test = test.drop(["id"], axis=1).to_numpy(dtype=np.float32)

_ = model.forward(X_full, training=True)

pred_y1, pred_y2 = model.forward(X_test, training=False)
pred_y1 = np.expm1(pred_y1.astype(np.float64))
pred_y2 = np.expm1(pred_y2.astype(np.float64))

pred_y1[pred_y1 < 0] = 0
pred_y2[pred_y2 < 0] = 0

subm = pd.DataFrame(
    {
        "id": sample["id"].to_numpy(),
        "formation_energy_ev_natom": pred_y1.reshape(-1),
        "bandgap_energy_ev": pred_y2.reshape(-1),
    }
)

subm = subm[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]
subm.to_csv("subm.csv", index=False)
print("Wrote submission:", os.path.abspath("subm.csv"), "shape:", subm.shape)
print(subm.head())
