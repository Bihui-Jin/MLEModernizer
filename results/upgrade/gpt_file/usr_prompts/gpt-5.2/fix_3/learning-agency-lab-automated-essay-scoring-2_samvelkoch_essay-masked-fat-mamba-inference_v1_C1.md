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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8174961363497553

# 6. Current score

0.5776

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47411) has done: 'I fix the immediate runtime blockers by removing the dependency on missing Hugging Face model paths (the `hf-proxy-d-trick-*` datasets are not present in this environment), which is why `tokenizer/model` never get defined and `score` is never created. To keep the pipeline end-to-end and produce a valid `submission.csv`, I replace the broken inference section with a lightweight, fully local baseline using `sentence-transformers` embeddings plus a simple classifier, which should yield a reasonable QWK without changing any external I/O paths. I also ensure the predicted labels are constrained to the required 1–6 range and that the submission columns match exactly. The rest of the notebook structure remains minimal and focused on producing a correct submission file.'
- What this solution (achieved 0.5776) has done: 'I fix the `SentenceTransformer` import/runtime crash (the protobuf `MessageFactory.GetPrototype` issue) by removing the dependency on `sentence-transformers` and switching to a fully local Hugging Face `transformers` embedding pipeline. This preserves the same core approach (text embeddings → multinomial LogisticRegression) while making it compatible with the Kaggle Python 3.12 environment. To move the score toward the target, I use a stronger DeBERTa-v3 embedding model (mean pooled last hidden states) which typically yields a large QWK jump versus MiniLM, without changing the downstream classifier/training semantics. I also ensure deterministic behavior, correct label range (1–6), and that `submission.csv` is written with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch



## === cell 1
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device_count:", torch.cuda.device_count())



## === cell 2
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)

print("train shape:", train.shape, "test shape:", test.shape)
print(train.columns.tolist())
print(test.columns.tolist())



## === cell 3
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}



## === cell 4
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(embed_model_name, use_fast=True)
model = AutoModel.from_pretrained(embed_model_name)
model.to(device)
model.eval()


def mean_pool_last_hidden(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / denom


@torch.inference_mode()
def encode_texts(texts, batch_size=32, max_length=512):
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        emb = mean_pool_last_hidden(
            out.last_hidden_state, enc["attention_mask"]
        )  # [B, H]
        emb = torch.nn.functional.normalize(
            emb, p=2, dim=1
        )  # match prior normalize_embeddings=True behavior
        all_embeds.append(emb.detach().cpu())
    return torch.cat(all_embeds, dim=0).numpy()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
train_texts = train["full_text"].astype(str).tolist()
y_train = train["score"].astype(int).values  # 1..6
test_texts = test["full_text"].astype(str).tolist()

X_train = encode_texts(train_texts, batch_size=32, max_length=512)
X_test = encode_texts(test_texts, batch_size=32, max_length=512)

print("Embeddings:", X_train.shape, X_test.shape)



## === cell 6
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    n_jobs=None,
    random_state=42,
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test).astype(int)
pred = np.clip(pred, 1, 6)

test["score"] = pred
print(test[["essay_id", "score"]].head())



## === cell 7
test["score"] = test["score"].astype(int)
sub = test[["essay_id", "score"]]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
