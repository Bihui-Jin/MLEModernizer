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

0.3970480505992886

# 6. Current score

0.70376

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with the required `essay_id,score` columns. The main blocker is missing local HF model paths (`hf-proxy-part-*-spacy-mamba`), so I add a robust fallback that uses the provided `sample_submission.csv` structure and generates a safe baseline prediction when the model artifacts aren’t present. I also fix the classification head usage (your `Linear(1,1)` cannot produce 6-class logits) by only running that path when the head weights are compatible; otherwise we fall back without crashing. These changes are minimal, score-neutral-to-slightly-better than “no submission”, and ensure Kaggle accepts the output.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a “flat” baseline (constant mean-rounded class), which usually yields near-zero QWK on this task. To move toward the target (~0.397) with minimal core-logic disruption, I keep your existing fallback structure but improve the fallback predictor to a simple, legitimate text-based model: TF‑IDF features + Ridge regression trained on all train data, then round/clip to 1–6 for submission. This preserves your overall pipeline (load CSVs → predict scores → write `submission.csv`) and only changes the fallback branch used when the local HF artifacts are missing. It should produce a non-trivial spread of predictions and typically improves QWK substantially versus a constant predictor, moving you toward the target band.'
- What this solution (achieved 0.70376) has done: 'I fix the Ridge regression crash by forcing a stable solver that doesn’t rely on SciPy’s `cg(tol=...)`, which is incompatible in this Kaggle environment. This keeps your fallback core logic (TF‑IDF → linear model → round/clip to 1–6) intact while making it run end-to-end and produce a valid `submission.csv`. I also add a small safety import for scikit-learn version visibility and keep the model branch unchanged. The resulting submission should materially improve over the current 0.0 (which came from failure/no valid predictions) and move toward the target QWK.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

INPUT_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device_count:", torch.cuda.device_count())



## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(train.shape, test.shape, sample_sub.shape)
print("train cols:", train.columns.tolist())
print("test cols:", test.columns.tolist())

reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

if "score" not in test.columns:
    test["score"] = np.nan



## === cell 2
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_path_part_1 = "/kaggle/input/hf-proxy-part-1-spacy-mamba"
local_model_path_part_2 = "/kaggle/input/hf-proxy-part-2-spacy-mamba"
local_tokenizer_path = "/kaggle/input/hf-proxy-part-1-spacy-mamba"

combined_model_path = "/kaggle/working/combined_model"
os.makedirs(combined_model_path, exist_ok=True)


def _dir_exists(p: str) -> bool:
    return isinstance(p, str) and os.path.isdir(p)


have_parts = _dir_exists(local_model_path_part_1) and _dir_exists(
    local_model_path_part_2
)

tokenizer = None
model = None
head = None
use_model = False

if have_parts:
    for file_name in os.listdir(local_model_path_part_1):
        if not file_name.endswith(
            (".safetensors", ".json", ".bin", ".py", ".model", ".txt")
        ):
            continue
        full_file_name = os.path.join(local_model_path_part_1, file_name)
        symlink_name = os.path.join(combined_model_path, file_name)
        if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
            os.symlink(full_file_name, symlink_name)

    for file_name in os.listdir(local_model_path_part_2):
        if not file_name.endswith(".safetensors"):
            continue
        full_file_name = os.path.join(local_model_path_part_2, file_name)
        symlink_name = os.path.join(combined_model_path, file_name)
        if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
            os.symlink(full_file_name, symlink_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            combined_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

        head_weights_path = os.path.join(
            local_model_path_part_1, "classification_head.pth"
        )
        if os.path.isfile(head_weights_path):
            head_weights = torch.load(head_weights_path, map_location="cpu")

            if isinstance(head_weights, dict) and "weight" in head_weights:
                w = head_weights["weight"]
            else:
                w = head_weights

            if torch.is_tensor(w) and w.ndim == 2:
                out_features, in_features = int(w.shape[0]), int(w.shape[1])
                head = torch.nn.Linear(in_features, out_features, bias=False)
                head.weight.data = w.to(dtype=head.weight.dtype)
                head = head.to(next(model.parameters()).device).eval()
                use_model = True
                print(f"Loaded model+tokenizer+head. Head shape: {tuple(w.shape)}")
            else:
                print(
                    "classification_head.pth found but has unexpected format; falling back."
                )
        else:
            print("classification_head.pth not found; falling back.")
    except Exception as e:
        print("Model load failed; falling back. Error:", repr(e))
        tokenizer = None
        model = None
        head = None
        use_model = False
else:
    print(
        "Local HF model parts not found in /kaggle/input; using baseline fallback submission."
    )



## === cell 3
if use_model:
    device = next(model.parameters()).device

    for idx, row in test.iterrows():
        text = row["full_text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs).logits  # [B, T, V]
            last_hidden_like = out[:, -1, :]  # [B, V]
            logits = head(last_hidden_like)  # [B, C] expected C=6
            pred_class = int(torch.argmax(logits, dim=1).item())

        test.at[idx, "score"] = reverse_score_mapping.get(pred_class, 3)

        del inputs, out, last_hidden_like, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    print("sklearn:", sklearn.__version__)

    train_text = train["full_text"].fillna("").astype(str).values
    train_y = train["score"].astype(float).values
    test_text = test["full_text"].fillna("").astype(str).values

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=200_000,
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    reg = Ridge(alpha=3.0, random_state=42, solver="lsqr")
    reg.fit(X_train, train_y)
    preds = reg.predict(X_test)

    test["score"] = np.clip(np.rint(preds), 1, 6).astype(int)



## === cell 4
test["score"] = (
    pd.to_numeric(test["score"], errors="coerce").fillna(3).round().astype(int)
)
test["score"] = test["score"].clip(1, 6)

sub = test[["essay_id", "score"]].copy()

if "essay_id" in sample_sub.columns and len(sample_sub) == len(sub):
    sub = sample_sub[["essay_id"]].merge(sub, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print(sub.head())
print("Wrote:", out_path, "rows:", len(sub), "cols:", sub.columns.tolist())
print("score value counts:\n", sub["score"].value_counts().sort_index())
