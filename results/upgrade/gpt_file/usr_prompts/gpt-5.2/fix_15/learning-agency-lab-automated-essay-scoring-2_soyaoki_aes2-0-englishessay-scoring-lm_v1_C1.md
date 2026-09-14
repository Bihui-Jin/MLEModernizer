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

0.3194937486952924

# 6. Current score

0.60545

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03435) has done: 'The runtime failure is because the notebook assumes a local DeBERTa model directory exists inside the dataset input, but it doesn’t, so `tokenizer`/`model` never get created and later cells crash. I keep the same core approach (Transformer inference → use its output as a single feature → LinearRegression calibration → clip+round to 1–6) but switch to a commonly available Kaggle-built-in base model name and load it from local cache when possible. I also make the model output handling robust to either regression (1 logit) or classification (6 logits) heads without changing the overall pipeline. Finally, I ensure `submission.csv` is always written with the required columns and row count.'
- What this solution (achieved 0.60545) has done: 'I fix the failure to load a transformer model in Kaggle’s offline environment by forcing selection of a locally-available model directory from `/kaggle/input` (and avoiding any Hugging Face Hub download attempts). If no compatible model is found locally, I fall back to a lightweight, score-neutral baseline feature (text length) so the notebook still runs end-to-end and always writes a valid `submission.csv`. I also fix the spurious `MessageFactory.GetPrototype` crash by disabling protobuf’s C++ implementation before importing `transformers`. These changes preserve the core pipeline (single feature → LinearRegression calibration → clip+round to 1–6) while making it robust and improving score when a local transformer is present.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is substantially higher than the target (0.31949), so we should *reduce* performance toward the target with the smallest safe change that preserves the same pipeline (single feature → LinearRegression → clip+round 1–6). The most direct knob is to intentionally weaken the transformer feature by using much heavier truncation (smaller `max_length`) and a smaller batch size for stability; this keeps the same model, same calibration, same post-processing, but generally degrades QWK because the feature carries less essay information. I make `max_length` a global constant used for both train/test inference, and keep all I/O and submission formatting identical. If the run is in baseline (no transformer found), nothing changes because the baseline feature doesn’t use `max_length`.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so we should *intentionally weaken* the signal slightly while keeping the exact same pipeline (single feature from model logits → LinearRegression → clip+round 1–6). The smallest, safest knob is to truncate more aggressively during inference so the transformer feature contains less essay information; this typically reduces QWK without changing architecture, training loop, or metric semantics. I also fix determinism (set seeds + deterministic flags) so the score shift is stable run-to-run rather than noisy. Everything else (paths, CSV format, calibration model, post-processing) stays identical.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so we should intentionally weaken performance with the smallest safe change while keeping the same pipeline (single feature from logits → LinearRegression → clip+round to 1–6). The simplest knob is to make transformer inference far less informative by truncating much more aggressively and using only the first few tokens; this preserves architecture and training semantics but typically lowers QWK. I also keep determinism as-is and leave the baseline (no-transformer) path unchanged so it still runs end-to-end offline. The submission format and paths remain identical, and `submission.csv` still be written.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so the goal is to *reduce* performance toward the target with the smallest, safest change while keeping the exact same pipeline (single feature from transformer logits → LinearRegression → clip+round 1–6). The most direct knob is to weaken the transformer-derived feature by truncating even more aggressively so it captures less essay content; this should lower QWK without changing model, training, or post-processing logic. I only adjust `MAX_LENGTH` (used consistently for train/test inference) and keep all paths, feature construction, regression calibration, and CSV writing unchanged. This should move the score downward in a stable, predictable way.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is far above the target (0.31949), so we should intentionally reduce performance with the smallest safe change while keeping the exact same pipeline (single feature from transformer logits → LinearRegression → clip+round 1–6). The most direct knob is to weaken the transformer-derived feature further by truncating even more aggressively, which removes most essay information but preserves identical inference/training semantics. I only adjust `MAX_LENGTH` (and keep it applied consistently for both train/test inference) and leave paths, model loading, regression calibration, and submission writing unchanged. This should move QWK downward toward the target without risking invalid submissions.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so the goal is to *reduce* performance toward the target with the smallest safe change while preserving the same pipeline (single feature from logits → LinearRegression → clip+round 1–6). The simplest knob is to further weaken the transformer-derived feature by truncating even more aggressively, which removes almost all essay information but doesn’t change the model, training loop, or post-processing semantics. I only adjust `MAX_LENGTH` downward and keep everything else (paths, batching, regression calibration, submission format) identical to maintain stability and ensure `submission.csv` is still valid.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so the objective is to *decrease* performance toward the target with the smallest safe change. The most direct knob that preserves the exact same pipeline is to weaken the transformer-derived feature by truncating slightly less extremely than `MAX_LENGTH=1` but still very aggressively, which should produce a controlled, stable degradation/increase depending on where `1` currently lands; here it’s likely overly pathological and may be oddly strong/unstable due to calibration artifacts. I only adjust `MAX_LENGTH` to a small value that still removes most essay information while being less degenerate, and keep everything else (model loading, single-feature extraction, LinearRegression calibration, clip+round, submission writing) identical. This keeps runtime and offline robustness unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so we should *intentionally reduce* performance toward the target with the smallest safe change while keeping the exact same pipeline (single transformer-derived feature → LinearRegression calibration → clip+round 1–6). The smallest effective knob here is to further weaken the transformer feature by truncating much more aggressively during tokenization, which removes most essay information but preserves identical inference/training semantics. I only change `MAX_LENGTH` (and keep it applied consistently for both train and test inference), leaving model loading, batching, regression, and submission writing unchanged to maintain stability and validity.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is far above the target (0.31949), so the goal is to *decrease* performance toward the target with the smallest safe change while preserving the exact same pipeline (single feature from logits → LinearRegression → clip+round 1–6). The minimal and most controllable knob here is to weaken the transformer-derived feature further by truncating more aggressively during tokenization, which reduces how much essay information the feature can capture without changing the model, training loop, regression, or post-processing. I only change `MAX_LENGTH` (and keep it applied consistently to both train and test inference) and leave everything else untouched to keep runtime and submission validity stable. This should move QWK downward toward the target band while still producing a correct `submission.csv`.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is far above the target (0.31949), so we should intentionally reduce performance with the smallest safe change while keeping the same pipeline (single transformer-derived feature → LinearRegression → clip+round). Right now `MAX_LENGTH=1` can behave oddly/unstably because it effectively feeds mostly special tokens; making truncation slightly less degenerate (but still very aggressive) should reliably weaken the feature’s usefulness and move QWK downward. I only change `MAX_LENGTH` to a small value and keep all paths, batching, feature construction, regression calibration, and submission writing identical. This preserves end-to-end execution and still produces a valid `submission.csv`.'
- What this solution (achieved 0.60545) has done: 'Your current score (0.60545) is well above the target (0.31949), so we should *intentionally reduce* performance toward the target with the smallest safe change while preserving the exact same pipeline (single transformer-derived feature → LinearRegression → clip+round 1–6). The most controllable minimal knob is to further weaken the transformer signal by truncating more aggressively during tokenization, which keeps the model/feature/regression logic identical but removes most essay information. I only change `MAX_LENGTH` downward (applied consistently for both train and test inference) and keep batching, regression, and submission writing unchanged to maintain stability and ensure a valid `submission.csv`. This should move QWK downward toward the target band without risking runtime or format issues.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
BASE = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"


def find_local_hf_model_dir(search_roots):
    """
    Search Kaggle input dirs for a local HuggingFace model folder that contains:
      - config.json
      - and at least one weight file (pytorch_model.bin / model.safetensors / *.bin / *.safetensors)
    Prefer DeBERTa-like names if present.
    """
    candidates = []
    for root in search_roots:
        for cfg in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
            d = os.path.dirname(cfg)
            has_weights = any(
                os.path.exists(os.path.join(d, w))
                for w in [
                    "pytorch_model.bin",
                    "model.safetensors",
                ]
            ) or (
                len(glob.glob(os.path.join(d, "*.bin"))) > 0
                or len(glob.glob(os.path.join(d, "*.safetensors"))) > 0
            )
            if not has_weights:
                continue
            candidates.append(d)

    if not candidates:
        return None

    def pref_key(p):
        pl = p.lower()
        is_deberta = ("deberta" in pl) or ("debert" in pl)
        is_bertish = ("bert" in pl) or ("roberta" in pl) or ("electra" in pl)
        return (0 if is_deberta else (1 if is_bertish else 2), len(p))

    candidates = sorted(set(candidates), key=pref_key)
    return candidates[0]


local_model_dir = find_local_hf_model_dir(
    search_roots=[
        "/kaggle/input",
        BASE,
    ]
)

tokenizer = None
model = None
USE_TRANSFORMER = False

if local_model_dir is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_dir, local_files_only=True
        )
        model.to(device)
        model.eval()
        USE_TRANSFORMER = True
        print("Using local transformer model dir:", local_model_dir)
    except Exception as e:
        print(
            "Found local model dir but failed to load; will use baseline feature instead."
        )
        print("Load error:", repr(e))
        tokenizer, model = None, None
        USE_TRANSFORMER = False
else:
    print(
        "No local HF model found under /kaggle/input; will use baseline feature instead."
    )




## === cell 2
def inference_batch(texts, max_length=512):
    """
    Core logic preserved: produce a single numeric feature per essay.
    - If transformer available: use model logits → expected class index (if multi-class) or raw logit (if regression).
    - If not: use a deterministic baseline (log1p character length) to keep pipeline runnable offline.
    """
    if not USE_TRANSFORMER:
        feats = np.log1p(
            np.array(
                [len(t) if isinstance(t, str) else 0 for t in texts], dtype=np.float32
            )
        ).reshape(-1, 1)
        return feats

    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits
    logits = logits.detach().float().cpu().numpy()

    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    if logits.shape[1] > 1:
        x = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(x)
        p = p / p.sum(axis=1, keepdims=True)
        idx = np.arange(logits.shape[1], dtype=np.float32)[None, :]
        feats = (p * idx).sum(axis=1, keepdims=True)
    else:
        feats = logits

    return feats




## === cell 3
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)

df_train["full_text"] = df_train["full_text"].fillna("")
df_test["full_text"] = df_test["full_text"].fillna("")




## === cell 4
MAX_LENGTH = 2
BATCH_SIZE = 16

y_true_train = df_train["score"].to_numpy(dtype=np.float32)
train_feats = []

for i in tqdm(range(0, len(df_train), BATCH_SIZE)):
    batch_texts = df_train["full_text"].iloc[i : i + BATCH_SIZE].tolist()
    feats = inference_batch(batch_texts, max_length=MAX_LENGTH)
    train_feats.append(feats)

X_train = np.vstack(train_feats)
print("X_train shape:", X_train.shape)




## === cell 5
lr = LinearRegression()
lr.fit(X_train, y_true_train)

print("coefficient = ", lr.coef_)
print("intercept = ", lr.intercept_)




## === cell 6
test_feats = []
for i in tqdm(range(0, len(df_test), BATCH_SIZE)):
    batch_texts = df_test["full_text"].iloc[i : i + BATCH_SIZE].tolist()
    feats = inference_batch(batch_texts, max_length=MAX_LENGTH)
    test_feats.append(feats)

X_test = np.vstack(test_feats)
y_pred = lr.predict(X_test)

y_pred_1to6 = np.clip(y_pred, 1, 6).round().astype("int64")

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": y_pred_1to6})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Score value counts:\n", sub["score"].value_counts().sort_index())
assert sub.shape[0] == df_test.shape[0]
assert list(sub.columns) == ["essay_id", "score"]
