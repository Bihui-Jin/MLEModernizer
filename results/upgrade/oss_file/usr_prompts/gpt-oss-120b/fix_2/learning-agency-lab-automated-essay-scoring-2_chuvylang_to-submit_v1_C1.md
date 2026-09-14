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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7993434084034929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
LOAD_FROM = None  # Disabled loading from a non‑existent checkpoint



## === cell 1
import warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score

warnings.simplefilter("ignore")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    model_path = "/kaggle/input/vuxvuxregression/transformers/fuckyou/1/output_v1/checkpoint-27692"


USE_REGRESSION = True



## === cell 3
default_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM or default_model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    LOAD_FROM or default_model_name,
    num_labels=6,  # ensure 6 output classes for scores 1‑6
)




## === cell 4
class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
        ds = Dataset.from_dict(
            {
                "essay_id": df["essay_id"].tolist(),
                "full_text": df["full_text"].tolist(),
                "label": df["label"].tolist(),
            }
        )
        return ds

    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example["full_text"], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs

    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)

        tokenized_train = train_ds.map(self.tokenize_function, batched=True)
        tokenized_valid = valid_ds.map(self.tokenize_function, batched=True)

        return tokenized_train, tokenized_valid, self.tokenizer




## === cell 5
class CFG:
    n_splits = 5
    seed = 42
    max_length = 512  # reduced for speed & memory safety
    lr = 1e-5
    train_batch_size = 2
    eval_batch_size = 2
    train_epochs = 3  # modest epochs to keep runtime reasonable
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6




## === cell 6
training_args = TrainingArguments(
    output_dir="output_v",
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    logging_first_step=True,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3584781370.py in <cell line: 0>()
----> 1 training_args = TrainingArguments(
      2     output_dir="output_v",
      3     fp16=True,
      4     learning_rate=CFG.lr,
      5     per_device_train_batch_size=CFG.train_batch_size,

/usr/local/lib/python3.11/dist-packages/transformers/training_args.py in __init__(self, output_dir, overwrite_output_dir, do_train, do_eval, do_predict, eval_strategy, prediction_loss_only, per_device_train_batch_size, per_device_eval_batch_size, per_gpu_train_batch_size, per_gpu_eval_batch_size, gradient_accumulation_steps, eval_accumulation_steps, eval_delay, torch_empty_cache_steps, learning_rate, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, num_train_epochs, max_steps, lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps, log_level, log_level_replica, log_on_each_node, logging_dir, logging_strategy, logging_first_step, logging_steps, logging_nan_inf_filter, save_strategy, save_steps, save_total_limit, save_safetensors, save_on_each_node, save_only_model, restore_callback_states_from_checkpoint, no_cuda, use_cpu, use_mps_device, seed, data_seed, jit_mode_eval, use_ipex, bf16, fp16, fp16_opt_level, half_precision_backend, bf16_full_eval, fp16_full_eval, tf32, local_rank, ddp_backend, tpu_num_cores, tpu_metrics_debug, debug, dataloader_drop_last, eval_steps, dataloader_num_workers, dataloader_prefetch_factor, past_index, run_name, disable_tqdm, remove_unused_columns, label_names, load_best_model_at_end, metric_for_best_model, greater_is_better, ignore_data_skip, fsdp, fsdp_min_num_params, fsdp_config, fsdp_transformer_layer_cls_to_wrap, accelerator_config, deepspeed, label_smoothing_factor, optim, optim_args, adafactor, group_by_length, length_column_name, report_to, ddp_find_unused_parameters, ddp_bucket_cap_mb, ddp_broadcast_buffers, dataloader_pin_memory, dataloader_persistent_workers, skip_memory_metrics, use_legacy_prediction_loop, push_to_hub, resume_from_checkpoint, hub_model_id, hub_strategy, hub_token, hub_private_repo, hub_always_push, hub_revision, gradient_checkpointing, gradient_checkpointing_kwargs, include_inputs_for_metrics, include_for_metrics, eval_do_concat_batches, fp16_backend, push_to_hub_model_id, push_to_hub_organization, push_to_hub_token, mp_parameters, auto_find_batch_size, full_determinism, torchdynamo, ray_scope, ddp_timeout, torch_compile, torch_compile_backend, torch_compile_mode, include_tokens_per_second, include_num_input_tokens_seen, neftune_noise_alpha, optim_target_modules, batch_eval_metrics, eval_on_start, use_liger_kernel, liger_kernel_config, eval_use_gather_object, average_tokens_across_devices)

/usr/local/lib/python3.11/dist-packages/transformers/training_args.py in __post_init__(self)
   1661         if self.load_best_model_at_end and self.save_strategy != SaveStrategy.BEST:
   1662             if self.eval_strategy != self.save_strategy:
-> 1663                 raise ValueError(
   1664                     "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
   1665                     f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"

ValueError: --load_best_model_at_end requires the save and eval strategy to match, but found
- Evaluation strategy: IntervalStrategy.NO
- Save strategy: SaveStrategy.EPOCH

## === cell 7
test = pd.read_csv(PATHS.test_path)
print("Test shape:", test.shape)
test.head()



## === cell 9
all_pred = []
test["label"] = 0.0

for fold in range(CFG.n_splits):
    if LOAD_FROM:
        tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM)
        model = AutoModelForSequenceClassification.from_pretrained(
            LOAD_FROM, num_labels=CFG.num_labels
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(default_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            default_model_name, num_labels=CFG.num_labels
        )

    tokenize = Tokenize(test, test, tokenizer)
    tokenized_test, _, _ = tokenize()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_test,  # using test as dummy train for inference
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append(predictions)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3277646854.py in <cell line: 0>()
     22     trainer = Trainer(
     23         model=model,
---> 24         args=training_args,
     25         train_dataset=tokenized_test,  # using test as dummy train for inference
     26         data_collator=data_collator,

NameError: name 'training_args' is not defined

## === cell 10
preds = np.mean(np.stack(all_pred, axis=0), axis=0)
print("Predictions shape:", preds.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2104414390.py in <cell line: 0>()
----> 1 preds = np.mean(np.stack(all_pred, axis=0), axis=0)
      2 print("Predictions shape:", preds.shape)
      3 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 11
sub = pd.read_csv(PATHS.sub_path)
if USE_REGRESSION:
    sub["score"] = preds.clip(0, 5).round(0) + 1
else:
    sub["score"] = preds.argmax(axis=1) + 1

sub["score"] = sub["score"].fillna(1).astype("int32")
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3339447499.py in <cell line: 0>()
      2 if USE_REGRESSION:
      3     # Clip to valid score range, round, shift to 1‑6
----> 4     sub["score"] = preds.clip(0, 5).round(0) + 1
      5 else:
      6     sub["score"] = preds.argmax(axis=1) + 1

NameError: name 'preds' is not defined
