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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9536

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, json, random, math, glob
from pathlib import Path



## === cell 1
if not Path("/kaggle/working/monk_v1").exists():
    subprocess.run(
        ["git", "clone", "https://github.com/Tessellate-Imaging/monk_v1.git"],
        check=True,
    )



## === cell 2
req_path = "/kaggle/working/monk_v1/installation/requirements_kaggle.txt"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", req_path], check=True
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/2752840001.py in <cell line: 0>()
      1 # Install requirements (quiet-ish); keep as close to original as possible
      2 req_path = "/kaggle/working/monk_v1/installation/requirements_kaggle.txt"
----> 3 subprocess.run(
      4     [sys.executable, "-m", "pip", "install", "-q", "-r", req_path], check=True
      5 )

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command '['/usr/bin/python3', '-m', 'pip', 'install', '-q', '-r', '/kaggle/working/monk_v1/installation/requirements_kaggle.txt']' returned non-zero exit status 1.

## === cell 3
MONK_ROOT = "/kaggle/working/monk_v1/monk"
if MONK_ROOT not in sys.path:
    sys.path.append(MONK_ROOT)

os.chdir("/kaggle/working")



## === cell 4
train_dir = Path("/kaggle/working/train")
test_dir = Path("/kaggle/working/test")

if not train_dir.exists():
    subprocess.run(
        [
            "unzip",
            "-q",
            "/kaggle/input/aerial-cactus-identification/train.zip",
            "-d",
            "/kaggle/working",
        ],
        check=True,
    )

if not test_dir.exists():
    subprocess.run(
        [
            "unzip",
            "-q",
            "/kaggle/input/aerial-cactus-identification/test.zip",
            "-d",
            "/kaggle/working",
        ],
        check=True,
    )

assert (
    train_dir.exists() and test_dir.exists()
), "Train/test directories were not created as expected."



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1524271804.py in <cell line: 0>()
     29 # Sanity check a couple files
     30 assert (
---> 31     train_dir.exists() and test_dir.exists()
     32 ), "Train/test directories were not created as expected."
     33 

AssertionError: Train/test directories were not created as expected.

## === cell 5
from pytorch_prototype import prototype



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1211554623.py in <cell line: 0>()
----> 1 from pytorch_prototype import prototype
      2 

/kaggle/working/monk_v1/monk/pytorch_prototype.py in <module>
----> 1 from monk.pytorch.finetune.imports import *
      2 from monk.system.imports import *
      3 from monk.pytorch.finetune.level_14_master_main import prototype_master
      4 
      5 

ModuleNotFoundError: No module named 'monk'

## === cell 6
gtf = prototype(verbose=1)
gtf.Prototype("sample-project-1", "sample-experiment-1")

gtf.Dataset_Params(
    dataset_path=str(train_dir) + "/",
    path_to_csv="/kaggle/input/aerial-cactus-identification/train.csv",
    input_size=(32, 32),
    batch_size=16,
    shuffle_data=True,
    num_processors=3,
)

gtf.apply_random_horizontal_flip(train=True, val=True)
gtf.apply_normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    train=True,
    val=True,
    test=True,
)

gtf.Dataset()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/262502532.py in <cell line: 0>()
      1 # Build experiment + dataset
----> 2 gtf = prototype(verbose=1)
      3 gtf.Prototype("sample-project-1", "sample-experiment-1")
      4 
      5 # FIX: Use absolute dataset_path to avoid ambiguity; keep core parameters unchanged

NameError: name 'prototype' is not defined

## === cell 7
network = []
network.append(gtf.convolution(output_channels=16))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.convolution(output_channels=16))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.max_pooling())
gtf.debug_custom_model_design(network)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3656537276.py in <cell line: 0>()
      1 # Model definition (kept identical)
      2 network = []
----> 3 network.append(gtf.convolution(output_channels=16))
      4 network.append(gtf.batch_normalization())
      5 network.append(gtf.relu())

NameError: name 'gtf' is not defined

## === cell 8
subnetwork = []
branch1 = []
branch1.append(gtf.convolution(output_channels=16))
branch1.append(gtf.batch_normalization())
branch1.append(gtf.convolution(output_channels=16))
branch1.append(gtf.batch_normalization())

branch2 = []
branch2.append(gtf.convolution(output_channels=16))
branch2.append(gtf.batch_normalization())

branch3 = []
branch3.append(gtf.identity())

subnetwork.append(branch1)
subnetwork.append(branch2)
subnetwork.append(branch3)
subnetwork.append(gtf.concatenate())

network.append(subnetwork)
gtf.debug_custom_model_design(network)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/125520945.py in <cell line: 0>()
      1 subnetwork = []
      2 branch1 = []
----> 3 branch1.append(gtf.convolution(output_channels=16))
      4 branch1.append(gtf.batch_normalization())
      5 branch1.append(gtf.convolution(output_channels=16))

NameError: name 'gtf' is not defined

## === cell 9
network.append(gtf.convolution(output_channels=32))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.max_pooling())
gtf.debug_custom_model_design(network)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/694105132.py in <cell line: 0>()
----> 1 network.append(gtf.convolution(output_channels=32))
      2 network.append(gtf.batch_normalization())
      3 network.append(gtf.relu())
      4 network.append(gtf.max_pooling())
      5 gtf.debug_custom_model_design(network)

NameError: name 'gtf' is not defined

## === cell 10
subnetwork = []
branch1 = []
branch1.append(gtf.convolution(output_channels=32))
branch1.append(gtf.batch_normalization())
branch1.append(gtf.convolution(output_channels=32))
branch1.append(gtf.batch_normalization())

branch2 = []
branch2.append(gtf.convolution(output_channels=32))
branch2.append(gtf.batch_normalization())

branch3 = []
branch3.append(gtf.identity())

subnetwork.append(branch1)
subnetwork.append(branch2)
subnetwork.append(branch3)
subnetwork.append(gtf.add())

network.append(subnetwork)
gtf.debug_custom_model_design(network)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3330679362.py in <cell line: 0>()
      1 subnetwork = []
      2 branch1 = []
----> 3 branch1.append(gtf.convolution(output_channels=32))
      4 branch1.append(gtf.batch_normalization())
      5 branch1.append(gtf.convolution(output_channels=32))

NameError: name 'gtf' is not defined

## === cell 11
network.append(gtf.convolution(output_channels=32))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.max_pooling())
gtf.debug_custom_model_design(network)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/694105132.py in <cell line: 0>()
----> 1 network.append(gtf.convolution(output_channels=32))
      2 network.append(gtf.batch_normalization())
      3 network.append(gtf.relu())
      4 network.append(gtf.max_pooling())
      5 gtf.debug_custom_model_design(network)

NameError: name 'gtf' is not defined

## === cell 12
network.append(gtf.flatten())
network.append(gtf.fully_connected(units=1024))
network.append(gtf.dropout(drop_probability=0.2))
network.append(gtf.fully_connected(units=2))
gtf.Compile_Network(network, data_shape=(3, 32, 32))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637329125.py in <cell line: 0>()
----> 1 network.append(gtf.flatten())
      2 network.append(gtf.fully_connected(units=1024))
      3 network.append(gtf.dropout(drop_probability=0.2))
      4 network.append(gtf.fully_connected(units=2))
      5 gtf.Compile_Network(network, data_shape=(3, 32, 32))

NameError: name 'gtf' is not defined

## === cell 13
gtf.Training_Params(
    num_epochs=2,
    display_progress=True,
    display_progress_realtime=True,
    save_intermediate_models=False,
    save_training_logs=True,
)

gtf.optimizer_sgd(0.001)
gtf.lr_fixed()
gtf.loss_softmax_crossentropy()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/75459468.py in <cell line: 0>()
      1 # Training params (kept identical)
----> 2 gtf.Training_Params(
      3     num_epochs=2,
      4     display_progress=True,
      5     display_progress_realtime=True,

NameError: name 'gtf' is not defined

## === cell 14
gtf.Train()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3754948409.py in <cell line: 0>()
----> 1 gtf.Train()
      2 

NameError: name 'gtf' is not defined

## === cell 15
from pytorch_prototype import prototype

ptf = prototype(verbose=1)
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)

output = ptf.Infer(img_dir=str(test_dir) + "/")

assert isinstance(output, list) and len(output) > 0, "Inference returned empty output."



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3227143029.py in <cell line: 0>()
      1 # Inference
----> 2 from pytorch_prototype import prototype
      3 
      4 ptf = prototype(verbose=1)
      5 ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)

/kaggle/working/monk_v1/monk/pytorch_prototype.py in <module>
----> 1 from monk.pytorch.finetune.imports import *
      2 from monk.system.imports import *
      3 from monk.pytorch.finetune.level_14_master_main import prototype_master
      4 
      5 

ModuleNotFoundError: No module named 'monk'

## === cell 16
import numpy as np
import pandas as pd

sub_path = "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
sub = pd.read_csv(sub_path)

pred_map = {}
for rec in output:
    img_name = rec.get("img_name")
    if img_name is None:
        continue

    pred_class = rec.get("predicted_class", None)
    pred_score = rec.get("predicted_score", None)

    prob1 = None
    if pred_score is not None and pred_class is not None:
        try:
            pc = int(pred_class)
            ps = float(pred_score)
            ps = min(max(ps, 0.0), 1.0)
            prob1 = ps if pc == 1 else (1.0 - ps)
        except Exception:
            prob1 = None

    if prob1 is None and pred_class is not None:
        try:
            prob1 = float(int(pred_class))
        except Exception:
            prob1 = 0.5

    if prob1 is None:
        prob1 = 0.5

    pred_map[img_name] = prob1

sub["has_cactus"] = sub["id"].map(pred_map).fillna(0.5).astype(float)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4176649901.py in <cell line: 0>()
      9 # FIX: Submission requires probability; prefer predicted_score if present.
     10 pred_map = {}
---> 11 for rec in output:
     12     img_name = rec.get("img_name")
     13     if img_name is None:

NameError: name 'output' is not defined

## === cell 17
assert Path(
    "/kaggle/working/submission.csv"
).exists(), "submission.csv was not created."
chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == [
    "id",
    "has_cactus",
], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(
    pd.read_csv(sub_path)
), "Row count mismatch vs sample submission."
assert (
    chk["has_cactus"].between(0, 1).all()
), "Predictions must be probabilities in [0,1]."
chk.describe(include="all")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/250529455.py in <cell line: 0>()
      1 # Quick checks
----> 2 assert Path(
      3     "/kaggle/working/submission.csv"
      4 ).exists(), "submission.csv was not created."
      5 chk = pd.read_csv("/kaggle/working/submission.csv")

AssertionError: submission.csv was not created.

## === cell 18
def _safe_rm(path):
    p = Path(path)
    if p.exists():
        if p.is_dir():
            subprocess.run(["rm", "-rf", str(p)], check=True)
        else:
            p.unlink()


_safe_rm("/kaggle/working/monk_v1")
_safe_rm("/kaggle/working/test")
_safe_rm("/kaggle/working/train")
