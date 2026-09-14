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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.91331

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the stray non‑code text that caused a syntax error in the first cell and replace it with a proper comment. I also fix the log‑display cell so that it uses a plain shell command (no backticks). These minimal changes eliminate the syntax errors, let the script run, and ensure a valid `submission.csv` is produced.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/742843208.py", line 1
    I will remove the stray non‑code text that caused a syntax error in the first cell and replace it with a proper comment. I also fix the log‑display cell so that it uses a plain shell command (no backticks). These minimal changes eliminate the syntax errors, let the script run, and ensure a valid `submission.csv` is produced.
                               ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 2
!pip install git+https://github.com/rwightman/pytorch-image-models -q
!pip install torchdistill -q



## === cell 3
%%writefile distill.py
import argparse
import datetime
import cv2
import os
import time

import timm
import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.functional import auroc

from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import (
    is_main_process,
    init_distributed_mode,
    load_ckpt,
    save_ckpt,
    set_seed,
)
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import TrainingBox
from torchdistill.datasets import util
from torchdistill.datasets.registry import register_dataset
from torchdistill.datasets.wrapper import BaseDatasetWrapper
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model, register_model_func
from torchdistill.optim.registry import register_optimizer
from torchdistill.common import misc_util
from torchdistill.losses.single import register_org_loss

import numpy as np
import pandas as pd
from PIL import Image

logger = def_logger.getChild("aaa")


def get_train_file_path(image_id):
    return f"../input/plant-pathology-2020-fgvc7/images/{image_id}.jpg"


def get_test_file_path(image_id):
    return f"../input/plant-pathology-2020-fgvc7/images/{image_id}.jpg"


"""SAMここから"""
OPTIM_DICT = misc_util.get_classes_as_dict('torch.optim')


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.defaults = defaults
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.defaults["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.defaults["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if self.defaults["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@register_optimizer
class samwrapper(SAM):
    def __init__(self, params, base_optim_name, base_optim_params, sam_params):
        base_optimizer = OPTIM_DICT[base_optim_name.lower()]
        super().__init__(params, base_optimizer, **sam_params)


try:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from apex import amp
except ImportError:
    amp = None


class SAM_TrainingBox(TrainingBox):
    def first_update_params(self, loss, **kwargs):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        elif self.apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = (
                    amp.master_params(self.optimizer)
                    if self.apex
                    else [p for group in self.optimizer.param_groups for p in group["params"]]
                )
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)

            self.optimizer.first_step(zero_grad=True)

    def second_update_params(self, loss, **kwargs):
        self.stage_grad_count += 1
        if self.grad_accum_step > 1:
            loss /= self.grad_accum_step

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        elif self.apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self.stage_grad_count % self.grad_accum_step == 0:
            if self.max_grad_norm is not None:
                target_params = (
                    amp.master_params(self.optimizer)
                    if self.apex
                    else [p for group in self.optimizer.param_groups for p in group["params"]]
                )
                torch.nn.utils.clip_grad_norm_(target_params, self.max_grad_norm)

            self.optimizer.second_step(zero_grad=True)


def get_training_box(model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor, accelerator=None):
    return SAM_TrainingBox(model, data_loader_dict, train_config, device, device_ids, distributed, lr_factor, accelerator)


"""SAMここまで"""


@register_dataset
class PLANT_2020(torch.utils.data.Dataset):
    def __init__(self, inf, csv_path, transform=None, transform_params=None):
        df = pd.read_csv(csv_path)
        self.transform = util.build_transform(transform_params)
        df["path"] = df["image_id"].apply(get_train_file_path)
        self.df = df
        if inf:
            df["path"] = df["image_id"].apply(get_test_file_path)
            self.labels = np.zeros(df.shape[0])
        else:
            df["path"] = df["image_id"].apply(get_train_file_path)
            self.labels = df[["healthy", "multiple_diseases", "rust", "scab"]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.loc[idx, "path"]
        image = Image.open(file_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).float()
        return image, label


@register_model_func
def timm_model(timm_model_name=None, num_classes=1, pretrained=False):
    model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_argparser():
    parser = argparse.ArgumentParser(description="Knowledge distillation for image classification models")
    parser.add_argument("--config", required=True, help="yaml file path")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--log", help="log file path")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--seed", type=int, help="seed in random number generator")
    parser.add_argument("-sync_bn", action="store_true", help="use sync batch norm")
    parser.add_argument("-test_only", action="store_true", help="only test the models")
    parser.add_argument("-student_only", action="store_true", help="test the student model only")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("-adjust_lr", action="store_true", help="multiply learning rate by number of distributed processes (world_size)")
    return parser


def load_model(model_config, device, distributed, sync_bn):
    model = get_image_classification_model(model_config, distributed, sync_bn)
    if model is None:
        repo_or_dir = model_config.get("repo_or_dir", None)
        model = get_model(model_config["name"], repo_or_dir, **model_config["params"])
    ckpt_file_path = model_config["ckpt"]
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)


def train_one_epoch(training_box, device, epoch, log_freq, using_sam=False):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for sample_batch, targets, supp_dict in metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)

        if using_sam:
            loss = training_box(sample_batch, targets, supp_dict)
            training_box.first_update_params(loss)
            loss = training_box(sample_batch, targets, supp_dict)
            training_box.second_update_params(loss)
        else:
            loss = training_box(sample_batch, targets, supp_dict)
            training_box.update_params(loss)

        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError(f"The training loop was broken due to loss = {loss}")


@torch.no_grad()
def evaluate(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header="Test:"):
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith("cuda"):
        model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    batch_output = []
    batch_target = []
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        batch_output.append(output)
        batch_target.append(target)

    metric_logger.synchronize_between_processes()
    outputs = torch.cat(batch_output, dim=0)
    targets = torch.cat(batch_target, dim=0)
    auc = auroc(outputs, targets.int(), num_classes=4, pos_label=1)

    logger.info(f" * AUC {auc:.4f}\n")
    return auc, outputs


@torch.no_grad()
def inference(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header="Test:"):
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith("cuda"):
        model = DataParallel(model, device_ids=device_ids)
    if title is not None:
        logger.info(title)
    model.eval()
    batch_output = []
    metric_logger = MetricLogger(delimiter="  ")

    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        output = model(image)
        batch_output.append(output)

    outputs = torch.cat(batch_output, dim=0).cpu().numpy()
    probs = 1 / (1 + np.exp(-outputs))
    return probs


def train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info("Start training")
    train_config = config["train"]
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = (
        get_training_box(student_model, dataset_dict, train_config, device, device_ids, distributed, lr_factor)
        if teacher_model is None
        else get_distillation_box(
            teacher_model,
            student_model,
            dataset_dict,
            train_config,
            device,
            device_ids,
            distributed,
            lr_factor,
        )
    )
    best_val_top1_accuracy = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_top1_accuracy, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config["log_freq"]
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        using_sam = train_config["optimizer"]["type"] == "SAMWrapper"
        train_one_epoch(training_box, device, epoch, log_freq, using_sam=using_sam)
        val_top1_accuracy, _ = evaluate(
            student_model,
            training_box.val_data_loader,
            device,
            device_ids,
            distributed,
            log_freq=log_freq,
            header="Validation:",
        )
        if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
            logger.info(f"Best top-1 AUC: {best_val_top1_accuracy:.4f} -> {val_top1_accuracy:.4f}")
            logger.info(f"Updating ckpt at {ckpt_file_path}")
            best_val_top1_accuracy = val_top1_accuracy
            save_ckpt(
                student_model_without_ddp,
                optimizer,
                lr_scheduler,
                best_val_top1_accuracy,
                config,
                args,
                ckpt_file_path,
            )
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    training_box.clean_modules()


def make_submission_file(predictions, inf_config):
    """
    Build the submission CSV directly from the test CSV (image_id order) and the predicted probabilities.
    """
    logger.info("Start saving submission")
    test_df = pd.read_csv(inf_config["test_path"])
    test_df[["healthy", "multiple_diseases", "rust", "scab"]] = predictions
    test_df.to_csv(inf_config["save_path"], index=False)
    logger.info(f"Submission saved to {inf_config['save_path']}")


def main(args):
    if is_main_process() and args.log is not None:
        setup_log_file(os.path.expanduser(args.log))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    cudnn.benchmark = True
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    device = torch.device(args.device)
    dataset_dict = util.get_all_datasets(config["datasets"])
    models_config = config["models"]
    teacher_model_config = models_config.get("teacher_model", None)
    teacher_model = (
        load_model(teacher_model_config, device, distributed, False)
        if teacher_model_config is not None
        else None
    )
    student_model_config = (
        models_config.get("student_model", models_config.get("model"))
    )
    ckpt_file_path = student_model_config["ckpt"]
    student_model = load_model(student_model_config, device, distributed, args.sync_bn)

    if not args.test_only:
        train(
            teacher_model,
            student_model,
            dataset_dict,
            ckpt_file_path,
            device,
            device_ids,
            distributed,
            config,
            args,
        )
        student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config["ckpt"], model=student_model_without_ddp, strict=True)

    inf_config = config["inf"]
    inf_data_loader_config = inf_config["inf_data_loader"]
    inf_data_loader = util.build_data_loader(
        dataset_dict[inf_data_loader_config["dataset_id"]],
        inf_data_loader_config,
        distributed,
    )

    inf_predictions = inference(
        student_model,
        inf_data_loader,
        device,
        device_ids,
        distributed,
        title=f"[Student: {student_model_config['name']}]",
    )

    make_submission_file(inf_predictions, inf_config)


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())



## === cell 4
%%writefile config.yaml
datasets:
  PLANT_2020:
    name: &dataset_name "PLANT_2020"
    type: *dataset_name
    root: &root_dir "../input/plant-pathology-2020-fgvc7"
    splits:
      dummy:
        dataset_id: "dummy"
        params:
          inf: False
          csv_path: !join [ *root_dir, "/train.csv" ]
        random_split:
          lengths: [ 0.8, 0.2 ]
          generator_seed: 42
          sub_splits:
            - dataset_id: &dataset_train !join [ *dataset_name, "/train" ]
              transform_params:
                - type: "Resize"
                  params:
                    size: [ 224, 224 ]
                - &totensor
                  type: "ToTensor"
                  params:
                - &normalize
                  type: "Normalize"
                  params:
                    mean: [ 0.49139968, 0.48215841, 0.44653091 ]
                    std: [ 0.24703223, 0.24348513, 0.26158784 ]
            - dataset_id: &dataset_val !join [ *dataset_name, "/val" ]
              transform_params: &val_transform
                - type: "Resize"
                  params:
                    size: [ 224, 224 ]
                - *totensor
                - *normalize
      inf:
        dataset_id: &dataset_inf !join [ *dataset_name, "/inf" ]
        params:
          inf: True
          csv_path: !join [ *root_dir, "/test.csv" ]
          transform_params: *val_transform

models:
  model:
    name: "timm_model"
    params:
      timm_model_name: "tf_efficientnet_b0_ns"
      num_classes: 4
      pretrained: True
    ckpt: "model.ckpt"

train:
  seed: 42
  log_freq: 100
  start_epoch: 0
  num_epochs: 3

  optimizer:
    type: "SAMWrapper"
    params:
      base_optim_name: "Adam"
      base_optim_params:
        lr: 0.001
        momentum: 0.9
        weight_decay: 0.00001
      sam_params:
        rho: 0.5
        adaptive: True

  criterion:
    type: "GeneralizedCustomLoss"
    org_term:
      criterion:
        type: "BCEWithLogitsLoss"
        params:
      factor: 1.0

  scheduler:
    type: "CosineAnnealingLR"
    params:
      T_max: 6
      eta_min: 0
      last_epoch: -1

  train_data_loader:
    dataset_id: *dataset_train
    random_sample: True
    batch_size: 64
    num_workers: 2

  val_data_loader:
    dataset_id: *dataset_val
    random_sample: False
    batch_size: 128
    num_workers: 2

test:
  test_data_loader:
    dataset_id: *dataset_val
    random_sample: False
    batch_size: 32
    num_workers: 2

inf:
  inf_data_loader:
    dataset_id: *dataset_inf
    random_sample: False
    batch_size: 128
    num_workers: 2

  submission_file_path: !join [ *root_dir, "/sample_submission.csv" ]
  test_path: !join [ *root_dir, "/test.csv" ]
  save_path: "submission.csv"



## === cell 5
!python distill.py --config config.yaml --device cuda --seed 10 -student_only --log log.log



## === cell 6
!cat log.log
```

## --- ERROR in cell 6, traceback:
  File "/tmp/ipykernel_55/1404415410.py", line 2
    ```
    ^
SyntaxError: invalid syntax
