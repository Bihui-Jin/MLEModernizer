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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8046283730655472

# 6. Current score

0.57589

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51199) has done: 'I fix the Ridge runtime error by explicitly selecting a compatible solver so sklearn doesn’t call SciPy’s `cg()` with an unsupported `tol` argument in this environment. This allow the CV loop and final model fit to complete, which in turn defines `model` so inference works and `pred` exists for submission creation. I keep the same TF‑IDF + Ridge core approach and only adjust the Ridge configuration to be robust. Finally, I ensure the script always writes `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.42027) has done: 'Your current TF‑IDF + Ridge pipeline is likely under-scoring because the regression target is discrete (0, 0.25, …, 1.0) while the model outputs continuous values; lightly aligning predictions to that discrete scale (via minimal post-processing) typically improves Pearson for this competition without changing the model/training core. I keep the exact same model architecture and training approach, and only adjust inference post-processing by snapping predictions to the nearest allowed label (and keeping clipping). I also keep the Ridge solver fix you already made for runtime stability. This is a small, targeted change that should move your score upward toward the target.'
- What this solution (achieved 0.54792) has done: 'Your current score is far below the target, and the biggest issue is the discrete “snapping” of predictions to {0,0.25,…,1.0}, which typically harms Pearson correlation because Pearson rewards preserving relative ordering/continuous variation. To move the score upward toward the target while keeping the same TF‑IDF + Ridge core logic, I remove the snapping step and keep only clipping to [0,1]. I also make a minimal, metric-aligned tweak by selecting the Ridge regularization strength via the existing 5-fold CV loop (same training approach, just choosing `alpha` from a small grid) and then refit on all training data with that alpha. These are small, legitimate changes that usually improve Pearson for this competition without changing the overall pipeline or adding new dependencies, and they still write a valid `submission.csv`.'
- What this solution (achieved 0.57589) has done: 'Your current TF‑IDF + Ridge setup is valid but underperforms mainly because (a) the evaluation is Pearson correlation (ranking/linear association), while you select `alpha` using RMSE, and (b) you’re leaving easy signal on the table by not explicitly modeling the *difference/overlap* between anchor and target. I keep the same core pipeline (TF‑IDF → Ridge, same training loop style) but (1) choose `alpha` by mean CV Pearson instead of RMSE, and (2) minimally enrich the input text with a few lightweight, deterministic overlap features (shared tokens and context prefix) baked into the same text string so the model can learn them without changing architecture. These changes are directly metric-aligned and typically move scores upward toward your target while preserving the overall approach and producing the same `submission.csv` format.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "/kaggle/input/us-patent-phrase-to-phrase-matching"
train_data_path = os.path.join(DATA_DIR, "train.csv")
test_data_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Exists train_data_path:", os.path.exists(train_data_path), train_data_path)
print("Exists test_data_path:", os.path.exists(test_data_path), test_data_path)
print("Exists sample_sub_path:", os.path.exists(sample_sub_path), sample_sub_path)




## === cell 1
class SimpleWordPieceTokenizer:
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    ):
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"vocab_file not found: {vocab_file}")
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        for tok in [unk_token, cls_token, sep_token, pad_token]:
            if tok not in self.token_to_id:
                raise ValueError(f"Special token {tok} missing from vocab")
        self.unk_id = self.token_to_id[unk_token]
        self.cls_id = self.token_to_id[cls_token]
        self.sep_id = self.token_to_id[sep_token]
        self.pad_id = self.token_to_id[pad_token]

    def _basic_tokenize(self, text: str):
        if text is None:
            text = ""
        text = str(text)
        if self.do_lower_case:
            text = text.lower()
        return re.findall(r"[a-z0-9]+|[^\s\w]", text)

    def _wordpiece_tokenize(self, token: str):
        if token in self.token_to_id:
            return [token]
        chars = token
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = chars[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.token_to_id:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenize(self, text: str):
        out = []
        for tok in self._basic_tokenize(text):
            out.extend(self._wordpiece_tokenize(tok))
        return out

    def encode(self, text: str, add_special_tokens: bool = False):
        toks = self.tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]
        return ids

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_id] + list(token_ids_0) + [self.sep_id]
        return (
            [self.cls_id]
            + list(token_ids_0)
            + [self.sep_id]
            + list(token_ids_1)
            + [self.sep_id]
        )

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (1 + len(token_ids_0) + 1)
        return [0] * (1 + len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def convert_tokens_to_ids(self, token: str):
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, ids):
        toks = [self.id_to_token.get(int(i), self.unk_token) for i in ids]
        return " ".join(toks)


print("Tokenizer class defined (not instantiated: external vocab not available).")



## === cell 2
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print(train_data.head())
print("Train rows:", len(train_data), "cols:", list(train_data.columns))
print(test_data.head())
print("Test rows:", len(test_data), "cols:", list(test_data.columns))

required_train_cols = {"anchor", "target", "context", "score"}
required_test_cols = {"id", "anchor", "target", "context"}
missing_train = required_train_cols - set(train_data.columns)
missing_test = required_test_cols - set(test_data.columns)
if missing_train:
    raise RuntimeError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise RuntimeError(f"test.csv missing columns: {missing_test}")



## === cell 3
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


_tok_re = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def _simple_tokens(s: str):
    if s is None:
        return []
    return _tok_re.findall(str(s).lower())


def _make_text_df(df: pd.DataFrame) -> pd.Series:
    anchor = df["anchor"].fillna("").astype(str)
    target = df["target"].fillna("").astype(str)
    context = df["context"].fillna("").astype(str)

    shared_parts = []
    for a, t in zip(anchor.values, target.values):
        a_toks = set(_simple_tokens(a))
        t_toks = set(_simple_tokens(t))
        shared = sorted(a_toks.intersection(t_toks))
        shared_parts.append(" ".join(shared[:20]))

    ctx0 = context.str.slice(0, 1).fillna("")

    shared_parts = pd.Series(shared_parts, index=df.index)

    return (
        "anchor: "
        + anchor
        + " [SEP] target: "
        + target
        + " [CTX] "
        + context
        + " [CTX0] "
        + ctx0
        + " [SHARED] "
        + shared_parts
    )


X_text = _make_text_df(train_data)
y = train_data["score"].astype(np.float32).values
X_test_text = _make_text_df(test_data)


def _pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

RIDGE_SOLVER = "lsqr"

alpha_grid = [0.2, 0.5, 1.0, 2.0, 5.0]

alpha_to_pearson = {}
for alpha in alpha_grid:
    cv_ps = []
    for tr_idx, va_idx in kf.split(X_text):
        model_cv = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
                ("ridge", Ridge(alpha=alpha, solver=RIDGE_SOLVER, random_state=42)),
            ]
        )
        model_cv.fit(X_text.iloc[tr_idx], y[tr_idx])
        va_pred = model_cv.predict(X_text.iloc[va_idx]).astype(np.float32)
        va_pred = np.clip(va_pred, 0.0, 1.0)
        cv_ps.append(_pearsonr_np(y[va_idx], va_pred))
    alpha_to_pearson[alpha] = float(np.mean(cv_ps))

best_alpha = max(alpha_to_pearson, key=alpha_to_pearson.get)
print("Alpha grid mean CV Pearson:", alpha_to_pearson)
print("Selected best_alpha (by Pearson):", best_alpha)

model = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("ridge", Ridge(alpha=best_alpha, solver=RIDGE_SOLVER, random_state=42)),
    ]
)
model.fit(X_text, y)




## === cell 4
def _find_savedmodel_dir(base_dir: str):
    candidates = [
        base_dir,
        os.path.join(base_dir, "saved_model"),
        os.path.join(base_dir, "SavedModel"),
        os.path.join(base_dir, "model"),
        os.path.join(base_dir, "export"),
        os.path.join(base_dir, "tf_saved_model"),
    ]
    for c in candidates:
        if (
            os.path.isdir(c)
            and os.path.isdir(os.path.join(c, "variables"))
            and os.path.isfile(os.path.join(c, "saved_model.pb"))
        ):
            return c

    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            c = os.path.join(base_dir, name)
            if (
                os.path.isdir(c)
                and os.path.isdir(os.path.join(c, "variables"))
                and os.path.isfile(os.path.join(c, "saved_model.pb"))
            ):
                return c

    return None


print("SavedModel search (expected None here):", _find_savedmodel_dir(DATA_DIR))



## === cell 5
pred = model.predict(X_test_text)
pred = np.asarray(pred, dtype=np.float32).reshape(-1)

pred = np.clip(pred, 0.0, 1.0)

print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## === cell 6
submission = pd.read_csv(sample_sub_path)

if len(submission) != len(test_data):
    raise RuntimeError(
        f"Row mismatch: submission has {len(submission)} rows but test has {len(test_data)} rows"
    )

submission["score"] = pred
submission = submission[["id", "score"]]
submission["id"] = submission["id"].astype(str)
submission["score"] = submission["score"].astype(np.float32)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
