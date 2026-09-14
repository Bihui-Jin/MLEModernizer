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

    if hasattr(torch, "serialization") and hasattr(
        torch.serialization, "add_safe_globals"
    ):
        torch.serialization.add_safe_globals([Sequential])
except Exception:
    pass

ptf = prototype(verbose=1)
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True)


output = ptf.Infer(img_dir="test/")


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mUnpicklingError[0m                           Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1604178537.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     17[0m [0;34m[0m[0m
[1;32m     18[0m [0mptf[0m [0;34m=[0m [0mprototype[0m[0;34m([0m[0mverbose[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0mptf[0m[0;34m.[0m[0mPrototype[0m[0;34m([0m[0;34m"sample-project-1"[0m[0;34m,[0m [0;34m"sample-experiment-1"[0m[0;34m,[0m [0meval_infer[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

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

[0;32m/kaggle/working/monk_v1/monk/pytorch_prototype.py[0m in [0;36mPrototype[0;34m(self, project_name, experiment_name, eval_infer, resume_train, copy_from, pseudo_copy_from, summary)[0m
[1;32m     44[0m         ''' 
[1;32m     45[0m         [0mself[0m[0;34m.[0m[0mset_system_project[0m[0;34m([0m[0mproject_name[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 46[0;31m         self.set_system_experiment(experiment_name, eval_infer=eval_infer, resume_train=resume_train, copy_from=copy_from, 
[0m[1;32m     47[0m             pseudo_copy_from=pseudo_copy_from, summary=summary);
[1;32m     48[0m         [0mself[0m[0;34m.[0m[0mcustom_print[0m[0;34m([0m[0;34m"Experiment Details"[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/base_class.py[0m in [0;36mset_system_experiment[0;34m(self, experiment_name, eval_infer, copy_from, pseudo_copy_from, resume_train, summary)[0m
[1;32m    129[0m [0;34m[0m[0m
[1;32m    130[0m             [0;32mif[0m[0;34m([0m[0meval_infer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 131[0;31m                 [0mself[0m[0;34m.[0m[0mset_system_state_eval_infer[0m[0;34m([0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    132[0m             [0;32melif[0m[0;34m([0m[0mresume_train[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m                 [0mself[0m[0;34m.[0m[0mset_system_state_resume_train[0m[0;34m([0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/finetune/level_5_state_base.py[0m in [0;36mset_system_state_eval_infer[0;34m(self)[0m
[1;32m     42[0m                 [0mself[0m[0;34m.[0m[0mcustom_print[0m[0;34m([0m[0;34m"Add test transforms"[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m                 [0mself[0m[0;34m.[0m[0mcustom_print[0m[0;34m([0m[0;34m""[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 44[0;31m             [0mself[0m[0;34m.[0m[0mset_model_final[0m[0;34m([0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     45[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     46[0m             [0mmsg[0m [0;34m=[0m [0;34m"Model in {} not trained. Cannot perform testing or inferencing"[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m[[0m[0;34m"experiment_name"[0m[0;34m][0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/finetune/level_2_model_base.py[0m in [0;36mset_model_final[0;34m(self, path)[0m
[1;32m     63[0m             [0;32mif[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0misfile[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m[[0m[0;34m"model_dir_relative"[0m[0;34m][0m [0;34m+[0m [0;34m'final'[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     64[0m                 [0mself[0m[0;34m.[0m[0mcustom_print[0m[0;34m([0m[0;34m"    Loading model - {}"[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m[[0m[0;34m"model_dir_relative"[0m[0;34m][0m [0;34m+[0m [0;34m'final'[0m[0;34m)[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 65[0;31m                 [0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m[[0m[0;34m"local"[0m[0;34m][0m[0;34m[[0m[0;34m"model"[0m[0;34m][0m [0;34m=[0m [0mload_model[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     66[0m                 [0mself[0m[0;34m.[0m[0msystem_dict[0m [0;34m=[0m [0mmodel_to_device[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msystem_dict[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m     67[0m                 [0mself[0m[0;34m.[0m[0mcustom_print[0m[0;34m([0m[0;34m"    Model loaded!"[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/system/imports.py[0m in [0;36mdecorator_wrapper[0;34m(*function_args, **function_args_dicts)[0m
[1;32m    140[0m                                                       False)
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m             [0;32mreturn[0m [0mvalidate_function[0m[0;34m([0m[0;34m*[0m[0mfunction_args[0m[0;34m,[0m [0;34m**[0m[0mfunction_args_dicts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    143[0m         [0;32mreturn[0m [0mdecorator_wrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     [0;32mreturn[0m [0maccept_decorator[0m[0;34m[0m[0;34m[0m[0m

[0;32m/kaggle/working/monk_v1/monk/pytorch/models/return_model.py[0m in [0;36mload_model[0;34m(system_dict, path, final, resume, external_path)[0m
[1;32m     31[0m                 [0mfinetune_net[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mpath[0m [0;34m+[0m [0;34m"final"[0m[0;34m,[0m [0mmap_location[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mdevice[0m[0;34m([0m[0;34m'cpu'[0m[0;34m)[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m                 [0mfinetune_net[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mload[0m[0;34m([0m[0msystem_dict[0m[0;34m[[0m[0;34m"model_dir_relative"[0m[0;34m][0m [0;34m+[0m [0;34m"final"[0m[0;34m,[0m [0mmap_location[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mdevice[0m[0;34m([0m[0;34m'cpu'[0m[0;34m)[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m         [0;32mif[0m[0;34m([0m[0mresume[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m             [0mfinetune_net[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mload[0m[0;34m([0m[0msystem_dict[0m[0;34m[[0m[0;34m"model_dir_relative"[0m[0;34m][0m [0;34m+[0m [0;34m"resume_state"[0m[0;34m,[0m [0mmap_location[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mdevice[0m[0;34m([0m[0;34m'cpu'[0m[0;34m)[0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/serialization.py[0m in [0;36mload[0;34m(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)[0m
[1;32m   1468[0m                         )
[1;32m   1469[0m                     [0;32mexcept[0m [0mpickle[0m[0;34m.[0m[0mUnpicklingError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1470[0;31m                         [0;32mraise[0m [0mpickle[0m[0;34m.[0m[0mUnpicklingError[0m[0;34m([0m[0m_get_wo_message[0m[0;34m([0m[0mstr[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1471[0m                 return _load(
[1;32m   1472[0m                     [0mopened_zipfile[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mUnpicklingError[0m: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL torch.nn.modules.conv.Conv2d was not an allowed global by default. Please use `torch.serialization.add_safe_globals([Conv2d])` or the `torch.serialization.safe_globals([Conv2d])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

## === cell 22
num_0 = 0;
num_1 = 1;
