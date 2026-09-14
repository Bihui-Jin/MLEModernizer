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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.7344911694526672

# 6. Current score

0.06726

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00298) has done: 'The ensemble inference loop was restructured to load the model only once and reuse it for each checkpoint, avoiding repeated model construction and large intermediate arrays. A single weighted‑sum accumulation replaces storing every model’s logits separately, cutting memory churn and runtime dramatically while keeping exactly the same weighted averaging logic. Minor tweaks (enabling CuDNN benchmarking) further speed up GPU kernels.'
- What this solution (achieved 0.01269) has done: 'The fix pre‑loads all test batches once (moving them to the device) and re‑uses these tensors for every checkpoint, eliminating repeated DataLoader work and costly host‑to‑GPU copies. It also disables unnecessary worker processes for the test loader, which speeds up the single preparation pass. All other logic—including model architecture, weighting of logits, and post‑processing—remains unchanged.'
- What this solution (achieved 0.0) has done: 'Implemented robust data path handling and corrected variable scope so the notebook runs end‑to‑end and generates a valid `submission.csv`. The script now safely locates the required CSV files (checking common Kaggle directories), loads them, builds the question‑to‑answer lookup, and produces the cleaned predictions file without raising errors. This fixes the FileNotFoundError and NameError, ensuring a proper submission output.'
- What this solution (achieved 0.0119) has done: 'Implemented a lightweight answer‑retrieval fallback: after the exact‑question lookup, the code now scans all unique training answers and returns the first one that appears verbatim in the test context. This adds no new model training, preserves the original architecture, and modestly boosts overlap‑based Jaccard scores, moving the result toward the target. Minor clean‑up of imports and comments were added for clarity.'
- What this solution (achieved 0.06726) has done: 'I add a lightweight semantic‑similarity fallback using a multilingual Sentence‑Transformer and a nearest‑neighbor search. This keeps the exact‑question lookup unchanged, replaces the very slow “scan all answers in context” loop with a fast embedding‑based lookup, and therefore should raise the Jaccard score toward the target while leaving the model architecture untouched. The script also adds the required imports and ensures the final CSV is written correctly.'

# 9. Code solution

## === cell 0
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
import pandas as pd
import numpy as np
import collections
from string import punctuation
from tqdm import tqdm  # progress bar for the new fallback loop
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class Config:
    model_type = "xlm-roberta"
    model_name_or_path = "xlm-roberta-base"
    config_name = "xlm-roberta-base"
    fp16 = False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "xlm-roberta-base"
    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    logging_steps = 10

    output_dir = "output"
    seed = 2021




## === cell 2
class DatasetRetriever(Dataset):
    def __init__(self, features, mode="train"):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == "train":
            return {
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    feature["attention_mask"], dtype=torch.long
                ),
                "offset_mapping": torch.tensor(
                    feature["offset_mapping"], dtype=torch.long
                ),
                "start_position": torch.tensor(
                    feature["start_position"], dtype=torch.long
                ),
                "end_position": torch.tensor(feature["end_position"], dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    feature["attention_mask"], dtype=torch.long
                ),
                "offset_mapping": feature["offset_mapping"],
                "sequence_ids": feature["sequence_ids"],
                "id": feature["example_id"],
                "context": feature["context"],
                "question": feature["question"],
            }




## === cell 3
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.linear_layer = nn.Linear(config.hidden_size, 64)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(64, 2)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        linear_output = self.linear_layer(sequence_output)
        linear_output = self.dropout(linear_output)
        qa_logits = self.qa_outputs(linear_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def postprocess_qa_predictions(
    examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1
            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )
        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]
    return predictions




## === cell 5
def locate_file(rel_path):
    candidates = [
        rel_path,
        os.path.join("input", rel_path),
        os.path.join("data", rel_path),
        os.path.join("/", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to locate file: {rel_path}")


test_path = locate_file("chaii-hindi-and-tamil-question-answering/test.csv")
test = pd.read_csv(test_path)


def clean_text(txt):
    return " ".join(str(txt).split()).lstrip()


test["context"] = test["context"].apply(clean_text)
test["question"] = test["question"].apply(clean_text)

cfg = Config()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

encodings = tokenizer(
    test["question"].tolist(),
    test["context"].tolist(),
    truncation="only_second",
    max_length=cfg.max_seq_length,
    stride=cfg.doc_stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)

test_features = []
overflow_to_sample = encodings["overflow_to_sample_mapping"]
for i in range(len(encodings["input_ids"])):
    sample_idx = overflow_to_sample[i]
    feature = {
        "example_id": test.iloc[sample_idx]["id"],
        "context": test.iloc[sample_idx]["context"],
        "question": test.iloc[sample_idx]["question"],
        "input_ids": encodings["input_ids"][i],
        "attention_mask": encodings["attention_mask"][i],
        "offset_mapping": encodings["offset_mapping"][i],
        "sequence_ids": [
            0 if seq_id is None else seq_id for seq_id in encodings.sequence_ids(i)
        ],
    }
    test_features.append(feature)

train_path = locate_file("chaii-hindi-and-tamil-question-answering/train.csv")
train = pd.read_csv(train_path)
train["question"] = train["question"].apply(clean_text)
question_answer_map = dict(zip(train["question"], train["answer_text"]))



## === cell 6
print("Building sentence‑transformer embeddings for training questions...")
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
train_questions = train["question"].tolist()
train_embeddings = embed_model.encode(
    train_questions,
    batch_size=256,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(train_embeddings)

print("Encoding test questions...")
test_questions = test["question"].tolist()
test_embeddings = embed_model.encode(
    test_questions,
    batch_size=256,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

distances, indices = nn.kneighbors(test_embeddings, n_neighbors=1)

fin_preds = {}
for _, row in tqdm(test.iterrows(), total=len(test), desc="Exact‑question lookup"):
    q = row["question"]
    pred = question_answer_map.get(q, "")
    fin_preds[row["id"]] = pred

for i, row in tqdm(test.iterrows(), total=len(test), desc="Semantic fallback"):
    pid = row["id"]
    if fin_preds[pid]:  # already have a prediction from exact match
        continue
    nn_idx = indices[i][0]
    fallback_answer = train.iloc[nn_idx]["answer_text"]
    fin_preds[pid] = fallback_answer if isinstance(fallback_answer, str) else ""



## === cell 7
bad_starts = [".", ",", "(", ")", "-", "–", ";"]
bad_endings = ["...", "-", "(", ")", "–", ",", ";"]
tamil_ad = "கி.பி"
tamil_bc = "கி.மு"
tamil_km = "கி.மீ"
hindi_ad = "ई"
hindi_bc = "ஈ.पू"

cleaned = []
for pid, pred in fin_preds.items():
    pred = " ".join(pred.split())
    while pred and any(pred.startswith(s) for s in bad_starts):
        pred = pred[1:].lstrip()
    while pred and any(pred.endswith(e) for e in bad_endings):
        if pred.endswith("..."):
            pred = pred[:-3]
        else:
            pred = pred[:-1].rstrip()
    context = test.loc[test["id"] == pid, "context"].values[0]
    if (
        any(
            pred.endswith(x) for x in [tamil_ad, tamil_bc, tamil_km, hindi_ad, hindi_bc]
        )
        and pred + "." in context
    ):
        pred = pred + "."
    cleaned.append((pid, pred.strip(punctuation)))

submission_df = pd.DataFrame(cleaned, columns=["id", "PredictionString"])
submission_df.to_csv("submission.csv", index=False)
