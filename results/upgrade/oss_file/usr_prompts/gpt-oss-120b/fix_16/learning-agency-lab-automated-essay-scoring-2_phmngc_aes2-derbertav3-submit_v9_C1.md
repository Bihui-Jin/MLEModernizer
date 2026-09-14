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

3.14

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
sentence-transformers==4.1.0
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
tqdm==4.67.1
transformers==4.53.3

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

0.7855532689009783

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01767) has done: 'I fixed the protobuf import issue, corrected the model path handling (trying the local folder first and falling back to the public DeBERTa‑v3‑large model), added a safe device selection, and cleaned up the imports so the script runs from start to finish and writes a proper `submission.csv` file.'
- What this solution (achieved 0.70697) has done: 'Implemented targeted speed‑ups while keeping the exact model architecture, training loop, and prediction logic unchanged.

**Key changes**
- Added automatic mixed‑precision (`torch.autocast`) to the training loop to reduce GPU compute time without affecting final accuracy.
- Pre‑computed and stored tensors (`input_ids` and `attention_mask`) inside `ChunkedEssayDataset` during construction, eliminating per‑sample Python tensor conversion during loading.
- Moved the `nullcontext` import to the top level to avoid repeated imports.
- Enabled a slightly larger inference batch size (`256`) while retaining safe GPU memory use.
- Minor clean‑ups (unused imports removal) that do not alter functionality.

These adjustments cut the runtime substantially, ensuring the script completes well within the 600‑second limit while preserving identical prediction semantics.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import collections  # efficient aggregation
from contextlib import nullcontext  # moved to top level

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

np.random.seed(42)
torch.manual_seed(42)

TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
LOCAL_MODEL_PATH = "/kaggle/input/aes2-debertav3-large"
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3
TRAIN_BATCH_SIZE = 32  # increased batch size to reduce optimizer steps
PRED_BATCH_SIZE = 256  # larger batch for faster inference, still safe on GPU
NUM_LABELS = 6
NUM_WORKERS = min(4, os.cpu_count() or 0)  # a few more workers for loading


def load_tokenizer_and_model(local_path: str, fallback_name: str, num_labels: int):
    """
    Load tokenizer and model. If a local directory exists, load from it;
    otherwise, fall back to the HuggingFace hub model.
    """
    if os.path.isdir(local_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                local_path, num_labels=num_labels, local_files_only=True
            )
            print(f"Loaded tokenizer & model from local path: {local_path}")
            return tokenizer, model
        except Exception as e:
            print(f"Failed loading from local path ({e}); falling back to hub.")
    tokenizer = AutoTokenizer.from_pretrained(fallback_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_name, num_labels=num_labels
    )
    print(f"Loaded tokenizer & model from hub: {fallback_name}")
    return tokenizer, model


tokenizer, model = load_tokenizer_and_model(
    LOCAL_MODEL_PATH, FALLBACK_MODEL_NAME, NUM_LABELS
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()


def compute_metrics(preds, labels):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 2
_TAG_TOKENS = tokenizer("[B]", add_special_tokens=False)["input_ids"]


class ChunkedEssayDataset(Dataset):
    """
    Tokenises essays into overlapping chunks using fast tokenizer overflow handling.
    Works for both training (with labels) and inference (without labels).
    Pre‑computes tensors to avoid per‑sample Python overhead.
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 64,
        min_chunk_ratio: float = 0.3,
        with_labels: bool = False,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.with_labels = with_labels

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)
        len_tag = len(_TAG_TOKENS)
        self.max_content_len = max_len - self.num_special - len_tag

        essay_ids_arr = df["essay_id"].astype(str).values
        scores_arr = df["score"].values if with_labels else None
        texts = df["full_text"].astype(str).tolist()

        tokenised = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_content_len,
            stride=self.overlap,
            return_overflowing_tokens=True,
        )
        input_ids_chunks = tokenised["input_ids"]
        overflow_to_sample = tokenised["overflow_to_sample_mapping"]

        pad_id = tokenizer.pad_token_id
        build_fn = tokenizer.build_inputs_with_special_tokens
        min_len_allowed = self.max_content_len * self.min_chunk_ratio

        first_chunk_seen = {}

        for chunk_idx, chunk_ids in enumerate(input_ids_chunks):
            essay_idx = overflow_to_sample[chunk_idx]
            essay_id = essay_ids_arr[essay_idx]

            if len(chunk_ids) < min_len_allowed:
                if first_chunk_seen.get(essay_id, False):
                    continue

            first_chunk_seen[essay_id] = True

            chunk_with_tag = _TAG_TOKENS + chunk_ids
            processed = build_fn(chunk_with_tag)

            if len(processed) > self.max_len:
                processed = processed[: self.max_len]
            else:
                processed += [pad_id] * (self.max_len - len(processed))

            input_ids_tensor = torch.tensor(processed, dtype=torch.long)
            attention_mask_tensor = (input_ids_tensor != pad_id).long()

            if self.with_labels:
                label = int(scores_arr[essay_idx]) - 1
                self.samples.append(
                    (input_ids_tensor, attention_mask_tensor, essay_id, label)
                )
            else:
                self.samples.append((input_ids_tensor, attention_mask_tensor, essay_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.with_labels:
            input_ids, attention_mask, essay_id, label = self.samples[idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "essay_id": essay_id,
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            input_ids, attention_mask, essay_id = self.samples[idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "essay_id": essay_id,
            }




## === cell 3
df_train_full = pd.read_csv(TRAIN_PATH)
df_train = df_train_full.sample(n=5000, random_state=42).reset_index(drop=True)

train_dataset = ChunkedEssayDataset(
    df_train, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, with_labels=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

print("Starting fine‑tuning on a 5k‑sample subset...")
for epoch in range(2):  # two epochs for a modest boost
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device.type):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

torch.cuda.empty_cache()  # free unused memory before inference




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2008212535.py in <cell line: 0>()
     36 
     37         with torch.autocast(device.type):
---> 38             outputs = model(
     39                 input_ids=input_ids,
     40                 attention_mask=attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1077         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1078 
-> 1079         outputs = self.deberta(
   1080             input_ids,
   1081             token_type_ids=token_type_ids,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    784         )
    785 
--> 786         encoder_outputs = self.encoder(
    787             embedding_output,
    788             attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, hidden_states, attention_mask, output_hidden_states, output_attentions, query_states, relative_pos, return_dict)
    657         rel_embeddings = self.get_rel_embedding()
    658         for i, layer_module in enumerate(self.layer):
--> 659             output_states, attn_weights = layer_module(
    660                 next_kv,
    661                 attention_mask,

/usr/local/lib/python3.11/dist-packages/transformers/modeling_layers.py in __call__(self, *args, **kwargs)
     81 
     82             return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
---> 83         return super().__call__(*args, **kwargs)

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, hidden_states, attention_mask, query_states, relative_pos, rel_embeddings, output_attentions)
    436         output_attentions: bool = False,
    437     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
--> 438         attention_output, att_matrix = self.attention(
    439             hidden_states,
    440             attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, hidden_states, attention_mask, output_attentions, query_states, relative_pos, rel_embeddings)
    369         rel_embeddings=None,
    370     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
--> 371         self_output, att_matrix = self.self(
    372             hidden_states,
    373             attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, hidden_states, attention_mask, output_attentions, query_states, relative_pos, rel_embeddings)
    249         if self.relative_attention:
    250             rel_embeddings = self.pos_dropout(rel_embeddings)
--> 251             rel_att = self.disentangled_attention_bias(
    252                 query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    253             )

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
    323             c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
    324             c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
--> 325             c2p_att = torch.gather(
    326                 c2p_att,
    327                 dim=-1,

OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 16.88 MiB is free. Process 3652628 has 47.51 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, and 57.79 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 4
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    """
    Perform inference on chunked essays and aggregate chunk predictions
    by simple averaging per essay (identical to original logic).
    Uses lightweight Python dicts for aggregation to avoid heavy pandas ops.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    model.eval()
    sum_preds = collections.defaultdict(float)
    count_preds = collections.defaultdict(int)

    autocast_ctx = (
        torch.autocast(device.type) if device.type == "cuda" else nullcontext()
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            essay_ids = batch["essay_id"]

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1  # back to 1‑6

            for eid, pred in zip(essay_ids, pred_labels):
                sum_preds[eid] += pred
                count_preds[eid] += 1

    final_essay_ids = []
    final_scores = []
    for eid in sum_preds:
        mean_score = sum_preds[eid] / count_preds[eid]
        rounded = int(np.rint(mean_score))
        clipped = int(np.clip(rounded, 1, num_labels))
        final_essay_ids.append(eid)
        final_scores.append(clipped)

    final_results_df = pd.DataFrame(
        {"essay_id": final_essay_ids, "score": final_scores}
    )
    raw_preds = np.array([int(v) for v in sum_preds.values()])  # placeholder, not used
    return final_results_df, raw_preds




## === cell 5
df_test = pd.read_csv(TEST_PATH)
test_dataset = ChunkedEssayDataset(
    df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, with_labels=False
)




## === cell 6
test_results, _ = predict_essay_score(
    model=model,
    dataset=test_dataset,
    num_labels=NUM_LABELS,
    batch_size=PRED_BATCH_SIZE,  # larger batch for faster inference
    device=device,
)

print("Submission preview:")
print(test_results.head())




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2975134462.py in <cell line: 0>()
----> 1 test_results, _ = predict_essay_score(
      2     model=model,
      3     dataset=test_dataset,
      4     num_labels=NUM_LABELS,
      5     batch_size=PRED_BATCH_SIZE,  # larger batch for faster inference

/tmp/ipykernel_55/3536892917.py in predict_essay_score(model, dataset, num_labels, batch_size, device)
     28 
     29             with autocast_ctx:
---> 30                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)
     31 
     32             logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1077         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1078 
-> 1079         outputs = self.deberta(
   1080             input_ids,
   1081             token_type_ids=token_type_ids,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    776             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    777 
--> 778         embedding_output = self.embeddings(
    779             input_ids=input_ids,
    780             token_type_ids=token_type_ids,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, token_type_ids, position_ids, mask, inputs_embeds)
    539 
    540         if inputs_embeds is None:
--> 541             inputs_embeds = self.word_embeddings(input_ids)
    542 
    543         if self.position_embeddings is not None:

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/sparse.py in forward(self, input)
    188 
    189     def forward(self, input: Tensor) -> Tensor:
--> 190         return F.embedding(
    191             input,
    192             self.weight,

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
   2549         # remove once script supports set_grad_enabled
   2550         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
-> 2551     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
   2552 
   2553 

OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 12.88 MiB is free. Process 3652628 has 47.51 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, and 59.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2055677509.py in <cell line: 0>()
      1 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(submission_path, index=False)
      3 print(f"Submission written to {submission_path}")

NameError: name 'test_results' is not defined
