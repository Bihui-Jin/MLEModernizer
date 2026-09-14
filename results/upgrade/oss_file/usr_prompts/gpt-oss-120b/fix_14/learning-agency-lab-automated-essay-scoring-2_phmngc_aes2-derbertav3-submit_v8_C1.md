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

0.8019899540067545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I set the protobuf implementation before importing transformers, replace the invalid local model path with a public DeBERTa‑v3‑large model, add a safe device selection, and ensure the script creates the test dataset only after the tokenizer is defined. These fixes remove the import errors, allow the model to load correctly, and guarantee that a “submission.csv” with the required columns is written, enabling a valid end‑to‑end run.'
- What this solution (achieved 0.0) has done: 'Implemented a robust fix for the protobuf import error by resetting the environment variable right before importing `transformers`. Added a lightweight fine‑tuning loop (1 epoch) on the training data using the same chunking logic as the test dataset, which raises the model from a raw pretrained state to a task‑specific model and therefore improves the quadratic weighted kappa score toward the target. All other logic (prediction, aggregation, CSV output) remains unchanged, and the script now reliably writes a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # enable fast kernel selection
    torch.backends.cudnn.deterministic = False  # keep nondeterministic for speed

MODEL_NAME_OR_PATH = "microsoft/deberta-v3-large"  # public pretrained model
NUM_LABELS = 6  # scores 1‑6
MAX_LEN = 512
OVERLAP = 128
MIN_CHUNK_RATIO = 0.3
BATCH_SIZE = 32  # larger batch size reduces iteration count
NUM_WORKERS = min(4, os.cpu_count() or 1)  # parallel data loading

possible_base_dirs = [
    os.path.join("data", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("data", "input", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("/kaggle", "input", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("/kaggle", "working", "learning-agency-lab-automated-essay-scoring-2"),
]
BASE_DIR = None
for d in possible_base_dirs:
    if os.path.isdir(d):
        BASE_DIR = d
        break
if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = os.path.join("output")
os.makedirs(OUTPUT_PATH, exist_ok=True)




## === cell 1
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    accuracy_score,
)

import matplotlib.pyplot as plt

use_transformer = True
tokenizer = None
model = None

from transformers import AutoTokenizer, AutoModelForSequenceClassification

if use_transformer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, num_labels=NUM_LABELS
    )

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 3
def compute_metrics(preds, labels):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 4
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            essay_ids = batch["essay_id"]

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1  # shift to 1‑6

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})
    aggregated = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred = np.rint(aggregated.values)
    final_scores = np.clip(y_pred, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated.index, "score": final_scores}
    )
    return final_results_df, np.array(preds)




## === cell 5
class EssayDataset(Dataset):
    """
    Shared dataset for both training and testing.
    For training, `labels` are required (0‑based).
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len=512,
        overlap=128,
        min_chunk_ratio=0.3,  # kept for API compatibility
        is_train=False,
    ):
        self.is_train = is_train
        self.tokenizer = tokenizer

        texts = ("[A] " + df["full_text"].astype(str)).tolist()

        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            stride=overlap,
            padding="max_length",
            return_overflowing_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",  # returns torch tensors
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

        mappings = encodings["overflow_to_sample_mapping"]
        self.essay_ids = [df.iloc[int(idx)]["essay_id"] for idx in mappings]

        if self.is_train:
            raw_labels = df["score"].astype(int).values - 1
            self.labels = torch.tensor(
                [raw_labels[int(idx)] for idx in mappings], dtype=torch.long
            )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        out = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "essay_id": self.essay_ids[idx],
        }
        if self.is_train:
            out["labels"] = self.labels[idx]
        return out




## === cell 6
df_train = pd.read_csv(TRAIN_PATH)

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=42,
    stratify=df_train["score"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if use_transformer:
    train_dataset = EssayDataset(
        train_df.reset_index(drop=True),
        tokenizer,
        MAX_LEN,
        OVERLAP,
        MIN_CHUNK_RATIO,
        is_train=True,
    )
    val_dataset = EssayDataset(
        val_df.reset_index(drop=True),
        tokenizer,
        MAX_LEN,
        OVERLAP,
        MIN_CHUNK_RATIO,
        is_train=True,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1):  # 1 epoch – lightweight fine‑tuning
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1} average loss: {np.mean(epoch_losses):.4f}")

        model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = (
                        outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    )
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        val_metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        print(
            f"Validation QWK: {val_metrics['quadratic_weighted_kappa']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )
        model.train()
else:
    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), stop_words="english"
    )
    X_train = vectorizer.fit_transform(df_train["full_text"].astype(str))
    y_train = df_train["score"].astype(int)  # keep 1‑6 labels
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, n_jobs=-1
    )
    clf.fit(X_train, y_train)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/4123329518.py in <cell line: 0>()
     51 
     52             with torch.cuda.amp.autocast():
---> 53                 outputs = model(
     54                     input_ids=input_ids, attention_mask=attention_mask, labels=labels
     55                 )

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

OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 16.88 MiB is free. Process 1501136 has 47.51 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, and 57.79 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
df_test = pd.read_csv(TEST_PATH)

if use_transformer:
    test_dataset = EssayDataset(
        df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, is_train=False
    )
else:
    test_texts = df_test["full_text"].astype(str)




## === cell 8
if use_transformer:
    test_results, _ = predict_essay_score(
        model=model,
        dataset=test_dataset,
        num_labels=NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )
else:
    X_test = vectorizer.transform(test_texts)
    test_preds = clf.predict(X_test)
    test_results = pd.DataFrame(
        {"essay_id": df_test["essay_id"], "score": test_preds.astype(int)}
    )

print("Submission preview:")
print(test_results.head())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3981868004.py in <cell line: 0>()
      1 if use_transformer:
----> 2     test_results, _ = predict_essay_score(
      3         model=model,
      4         dataset=test_dataset,
      5         num_labels=NUM_LABELS,

/tmp/ipykernel_55/1613349014.py in predict_essay_score(model, dataset, num_labels, batch_size, device)
     22 
     23             with torch.cuda.amp.autocast():
---> 24                 outputs = model(input_ids=input_ids, attention_mask=attention_mask)
     25 
     26             logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

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

OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 14.88 MiB is free. Process 1501136 has 47.51 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, and 59.78 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 9
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3321823371.py in <cell line: 0>()
      1 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(submission_path, index=False)
      3 print(f"Submission saved to {submission_path}")

NameError: name 'test_results' is not defined
