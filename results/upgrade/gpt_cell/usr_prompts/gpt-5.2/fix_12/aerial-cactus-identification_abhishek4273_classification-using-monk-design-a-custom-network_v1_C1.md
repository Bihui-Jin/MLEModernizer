# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0
! git clone https://github.com/Tessellate-Imaging/monk_v1.git


## === cell 1
!pip install -r monk_v1/installation/requirements_kaggle.txt


## === cell 2
import sys
sys.path.append("/kaggle/working/monk_v1/monk/")


## === cell 3
!unzip -q /kaggle/input/aerial-cactus-identification/train.zip


## === cell 4
!unzip -q /kaggle/input/aerial-cactus-identification/test.zip


## === cell 5
import sys
import os

stub_dir = "/kaggle/working/_stub_modules"
os.makedirs(stub_dir, exist_ok=True)
if stub_dir not in sys.path:
    sys.path.insert(0, stub_dir)

if "GPUtil" not in sys.modules:
    try:
        import GPUtil  # noqa: F401
    except ModuleNotFoundError:
        stub_path = os.path.join(stub_dir, "GPUtil.py")
        if not os.path.exists(stub_path):
            with open(stub_path, "w") as f:
                f.write(
                    "class GPU:\n"
                    "    def __init__(self, id=0, load=0.0, memoryTotal=0, memoryUsed=0, memoryFree=0, temperature=0):\n"
                    "        self.id = id\n"
                    "        self.load = load\n"
                    "        self.memoryTotal = memoryTotal\n"
                    "        self.memoryUsed = memoryUsed\n"
                    "        self.memoryFree = memoryFree\n"
                    "        self.temperature = temperature\n"
                    "\n"
                    "def getGPUs():\n"
                    "    return []\n"
                    "\n"
                    "def showUtilization(all=False):\n"
                    "    return ''\n"
                )

if "pylg" not in sys.modules:
    try:
        import pylg  # noqa: F401
    except ModuleNotFoundError:
        stub_path = os.path.join(stub_dir, "pylg.py")
        if not os.path.exists(stub_path):
            with open(stub_path, "w") as f:
                f.write(
                    "class TraceFunction:\n"
                    "    def __init__(self, *args, **kwargs):\n"
                    "        pass\n"
                    "    def __call__(self, func=None, *args, **kwargs):\n"
                    "        return func\n"
                    "\n"
                    "def trace(*args, **kwargs):\n"
                    "    return None\n"
                )

repo_root = "/kaggle/working/monk_v1"
if repo_root not in sys.path:
    sys.path.append(repo_root)

from pytorch_prototype import prototype


## === cell 6
gtf = prototype(verbose=1)
gtf.Prototype("sample-project-1", "sample-experiment-1")

candidate_train_dirs = [
    "/kaggle/working/train/",
    "/kaggle/working/aerial-cactus-identification/train/",
    "/kaggle/input/aerial-cactus-identification/train/",
    "/kaggle/data/aerial-cactus-identification/train/",
]
dataset_path = next((p for p in candidate_train_dirs if os.path.isdir(p)), None)
if dataset_path is None:
    raise FileNotFoundError(
        "Could not find extracted train folder. Tried: "
        + ", ".join(candidate_train_dirs)
    )

gtf.Dataset_Params(
    dataset_path=dataset_path,
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


## === cell 8
import networkx as nx

if not hasattr(nx.draw_networkx, "_monk_with_label_compat"):
    _orig_draw_networkx = nx.draw_networkx

    def _draw_networkx_with_label_compat(
        G, pos=None, arrows=None, with_labels=True, **kwds
    ):
        if "with_label" in kwds and "with_labels" not in kwds:
            kwds["with_labels"] = kwds.pop("with_label")

        if "with_labels" in kwds:
            with_labels = kwds.pop("with_labels")

        return _orig_draw_networkx(
            G, pos=pos, arrows=arrows, with_labels=with_labels, **kwds
        )

    _draw_networkx_with_label_compat._monk_with_label_compat = True
    nx.draw_networkx = _draw_networkx_with_label_compat


network = []
network.append(gtf.convolution(output_channels=16))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.convolution(output_channels=16))
network.append(gtf.batch_normalization())
network.append(gtf.relu())
network.append(gtf.max_pooling())
gtf.debug_custom_model_design(network)


## === cell 9
subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=16));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.concatenate());


network.append(subnetwork);
gtf.debug_custom_model_design(network);


## === cell 10
network.append(gtf.convolution(output_channels=32));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);


## === cell 11
subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=32));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=32));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=32));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.add());


network.append(subnetwork);
gtf.debug_custom_model_design(network);


## === cell 12
network.append(gtf.convolution(output_channels=32));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());
gtf.debug_custom_model_design(network);


## === cell 13
network.append(gtf.flatten());
network.append(gtf.fully_connected(units=1024));
network.append(gtf.dropout(drop_probability=0.2));
network.append(gtf.fully_connected(units=2));
gtf.Compile_Network(network, data_shape=(3, 32, 32));


## === cell 17
gtf.Training_Params(num_epochs=2, display_progress=True, display_progress_realtime=True, 
        save_intermediate_models=False, save_training_logs=True);


gtf.optimizer_sgd(0.001);
gtf.lr_fixed();
gtf.loss_softmax_crossentropy();


## === cell 18
gtf.Train();


## === cell 21
from pytorch_prototype import prototype

try:
    import torch
    from torch.nn.modules.container import Sequential
    from torch.nn import (
        Conv2d,
        BatchNorm2d,
        Linear,
        Dropout,
        ReLU,
        MaxPool2d,
        Identity,
        Flatten,
    )

    if hasattr(torch, "serialization") and hasattr(
        torch.serialization, "add_safe_globals"
    ):
        safe = [
            Sequential,
            Conv2d,
            BatchNorm2d,
            Linear,
            Dropout,
            ReLU,
            MaxPool2d,
            Identity,
            Flatten,
        ]

        try:
            from monk.pytorch.models import layers as monk_layers

            for name in [
                "Net_Concat",
                "Net_Add",
                "Net_Identity",
                "Net_Flatten",
                "Net_Conv",
                "Net_BatchNorm",
                "Net_Relu",
                "Net_MaxPool",
                "Net_FC",
                "Net_Dropout",
            ]:
                cls = getattr(monk_layers, name, None)
                if cls is not None:
                    safe.append(cls)
        except Exception:
            pass

        torch.serialization.add_safe_globals(safe)
except Exception:
    pass

ptf = prototype(verbose=1)
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)

import os

candidate_test_dirs = [
    "test/",
    "/kaggle/working/test/",
    "/kaggle/working/aerial-cactus-identification/test/",
    "/kaggle/input/aerial-cactus-identification/test/",
    "/kaggle/data/aerial-cactus-identification/test/",
]
test_dir = next((p for p in candidate_test_dirs if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError(
        "Could not find test folder. Tried: " + ", ".join(candidate_test_dirs)
    )

output = ptf.Infer(img_dir=test_dir)


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIsADirectoryError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4262712737.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     74[0m     )
[1;32m     75[0m [0;34m[0m[0m
[0;32m---> 76[0;31m [0moutput[0m [0;34m=[0m [0mptf[0m[0;34m.[0m[0mInfer[0m[0;34m([0m[0mimg_dir[0m[0;34m=[0m[0mtest_dir[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    490[0m [0;34m[0m[0m
[1;32m    491[0m [0;34m[0m[0m
[0;32m--> 492[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    493[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    494[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/finetune/level_14_master_main.py[0m in [0;36mInfer[0;34m(self, img_name, img_dir, return_raw, img_thresh)[0m
[1;32m    251[0m                 [0mpredictions[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mset_prediction_final[0m[0;34m([0m[0mimg_name[0m[0;34m=[0m[0mimg_name[0m[0;34m,[0m [0mreturn_raw[0m[0;34m=[0m[0mreturn_raw[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 253[0;31m                 [0mpredictions[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mset_prediction_final[0m[0;34m([0m[0mimg_dir[0m[0;34m=[0m[0mimg_dir[0m[0;34m,[0m [0mreturn_raw[0m[0;34m=[0m[0mreturn_raw[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    254[0m             [0;32mreturn[0m [0mpredictions[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m    255[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/finetune/level_4_evaluation_base.py[0m in [0;36mset_prediction_final[0;34m(self, img_name, img_dir, return_raw)[0m
[1;32m    251[0m                 [0mimg_name[0m [0;34m=[0m [0mimg_dir[0m [0;34m+[0m [0;34m"/"[0m [0;34m+[0m [0mimg_list[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m
[0;32m--> 253[0;31m                 [0mlabel[0m[0;34m,[0m [0mscore[0m[0;34m,[0m [0mraw_output[0m [0;34m=[0m [0mprocess_single[0m[0;34m([0m[0mimg_name[0m[0;34m,[0m [0mreturn_raw[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    254[0m [0;34m[0m[0m
[1;32m    255[0m                 [0mtmp[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/testing/process.py[0m in [0;36mprocess_single[0;34m(img_name, return_raw, system_dict)[0m
[1;32m     18[0m         [0mfloat[0m[0;34m:[0m [0mprediction[0m [0mscore[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m     '''
[0;32m---> 20[0;31m     [0mimg[0m [0;34m=[0m [0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mimg_name[0m[0;34m)[0m[0;34m.[0m[0mconvert[0m[0;34m([0m[0;34m'RGB'[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m     [0mimg[0m [0;34m=[0m [0msystem_dict[0m[0;34m[[0m[0;34m"local"[0m[0;34m][0m[0;34m[[0m[0;34m"data_transforms"[0m[0;34m][0m[0;34m[[0m[0;34m"test"[0m[0;34m][0m[0;34m([0m[0mimg[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m     [0mimg[0m [0;34m=[0m [0mimg[0m[0;34m.[0m[0munsqueeze[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/PIL/Image.py[0m in [0;36mopen[0;34m(fp, mode, formats)[0m
[1;32m   3511[0m     [0;32mif[0m [0mis_path[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3512[0m         [0mfilename[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mfspath[0m[0;34m([0m[0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3513[0;31m         [0mfp[0m [0;34m=[0m [0mbuiltins[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mfilename[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3514[0m         [0mexclusive_fp[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3515[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIsADirectoryError[0m: [Errno 21] Is a directory: '/kaggle/working/aerial-cactus-identification/test//test'

## === cell 22
num_0 = 0;
num_1 = 1;
