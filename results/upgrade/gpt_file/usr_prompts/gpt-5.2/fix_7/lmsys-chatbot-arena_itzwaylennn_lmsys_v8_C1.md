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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

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
textblob==0.19.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.090446606820894

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the non-Python prose/triple-backtick lines that are currently being executed as code, which is the direct cause of the SyntaxErrors preventing the pipeline from reaching CSV creation/validation. I will keep your feature extraction and XGBoost training logic the same, but make the data-path selection robust for both `/kaggle/input/...` and the `/kaggle/data/...` layout shown in your file tree so the notebook runs in this environment. Finally, I will harden submission building to always produce exactly the required columns in the correct order by starting from `sample_submission.csv`, merging predictions by `id`, and re-creating/filling any missing probability columns before renormalizing and writing `submission.csv`.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/351890490.py", line 1
    I will remove the non-Python prose/triple-backtick lines that are currently being executed as code, which is the direct cause of the SyntaxErrors preventing the pipeline from reaching CSV creation/validation. I will keep your feature extraction and XGBoost training logic the same, but make the data-path selection robust for both `/kaggle/input/...` and the `/kaggle/data/...` layout shown in your file tree so the notebook runs in this environment. Finally, I will harden submission building to always produce exactly the required columns in the correct order by starting from `sample_submission.csv`, merging predictions by `id`, and re-creating/filling any missing probability columns before renormalizing and writing `submission.csv`.
      ^
SyntaxError: invalid syntax


## === cell 1
import logging
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow/CUDA if used
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Attempting to register.*factory")
warnings.filterwarnings("ignore", message="The 'repr' attribute.*")



## === cell 2
try:
    import textstat  # type: ignore
    _HAS_TEXTSTAT = True
    print("textstat available.")
except Exception as e:
    _HAS_TEXTSTAT = False
    print(f"textstat not available; using fallback readability features. ({e})")



## === cell 3
print("Skipping bitsandbytes installation (not required for this pipeline).")



## === cell 4
import pandas as pd
import numpy as np
import json
import ast
import re
import time
from tqdm import tqdm
import logging

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting feature extraction + XGBoost pipeline.")

def _pick_first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0]

TRAIN_FILE = _pick_first_existing(
    "/kaggle/input/lmsys-chatbot-arena/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/lmsys-chatbot-arena/train.csv",
    "/kaggle/data/train.csv",
)
TEST_FILE = _pick_first_existing(
    "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/lmsys-chatbot-arena/test.csv",
    "/kaggle/data/test.csv",
)
SUBMISSION_SAMPLE = _pick_first_existing(
    "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/lmsys-chatbot-arena/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)

FEATURE_OUTPUT_TEST = "/kaggle/working/lmsys_test_features_final.csv"
FEATURE_OUTPUT_TRAIN = "/kaggle/working/lmsys_train_features_final.csv"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

RANDOM_STATE = 42

logger.info(f"Using TRAIN_FILE={TRAIN_FILE}")
logger.info(f"Using TEST_FILE={TEST_FILE}")
logger.info(f"Using SUBMISSION_SAMPLE={SUBMISSION_SAMPLE}")



## === cell 5
print("No offline transformer model directories will be used; using lightweight local features only.")



## === cell 6
def xgb_gpu_is_usable() -> bool:
    return True

_GPU_USABLE_HINT = xgb_gpu_is_usable()
print(f"XGBoost GPU usability hint: {_GPU_USABLE_HINT} (will verify by attempting training)")



## === cell 7
print("Skipping offline embedding model load; will use TF-IDF cosine similarity features instead.")



## === cell 8
import pandas as pd
test_df = pd.read_csv(TEST_FILE)
print("Test shape:", test_df.shape)
print(test_df.head(2))



## === cell 9
import pandas as pd
import numpy as np
import ast
import re
import time
import logging
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def safe_literal_eval(text):
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text

def get_last_turn(prompt_list):
    if not isinstance(prompt_list, list) or not prompt_list:
        return ""
    user_turns = [t for i, t in enumerate(prompt_list) if i % 2 == 0]
    return str(user_turns[-1]) if user_turns else ""

def safe_join(x):
    if isinstance(x, list):
        return " ".join([str(item) for item in x if item is not None])
    return str(x) if x is not None else ""

def clean_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def parse_list_first(text):
    if pd.isna(text):
        return ""
    s = str(text).strip()
    try:
        import json
        lst = json.loads(s)
        return lst[0] if isinstance(lst, list) and len(lst) else ""
    except Exception:
        pass
    s2 = s.replace(r"\/", "/").replace("null", "''")
    try:
        lst = ast.literal_eval(s2)
        return lst[0] if isinstance(lst, list) and len(lst) else ""
    except Exception:
        return ""

def get_textblob_features(text):
    if not isinstance(text, str) or not text.strip():
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}
    try:
        blob = TextBlob(text)
        return {
            "textblob_polarity": float(blob.sentiment.polarity),
            "textblob_subjectivity": float(blob.sentiment.subjectivity),
        }
    except Exception:
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}

def get_readability_scores(text):
    if not isinstance(text, str) or not text.strip():
        return {
            "flesch_kincaid_grade": np.nan,
            "gunning_fog": np.nan,
        }
    if _HAS_TEXTSTAT:
        try:
            import textstat  # type: ignore
            return {
                "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
                "gunning_fog": float(textstat.gunning_fog(text)),
            }
        except Exception:
            pass

    words = re.findall(r"\b\w+\b", text)
    n_words = len(words)
    n_sents = max(1, len(re.findall(r"[.!?]+", text)))
    avg_words_per_sent = n_words / n_sents if n_sents else 0.0
    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    fk_proxy = 0.39 * avg_words_per_sent + 11.8 * (avg_word_len / 5.0) - 15.59
    gf_proxy = 0.4 * (avg_words_per_sent + 100.0 * (1.0 if avg_word_len > 6.0 else 0.0))
    return {
        "flesch_kincaid_grade": float(fk_proxy),
        "gunning_fog": float(gf_proxy),
    }

def build_prompt_purpose_strict(df, prompt_col="prompt_clean", proximity_window=60):
    s = df[prompt_col].astype("string").fillna("")
    lo = s.str.lower()
    P = pd.DataFrame(index=df.index)
    P["p_codefences"] = s.str.count(r"```")
    P["p_bullets"] = s.str.count(r"(?m)^\s*[-*•]\s+")
    P["p_numlist"] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    P["p_list_lines"] = (P["p_bullets"] + P["p_numlist"]).astype("int16")
    tail_stripped = s.str.replace(r'[\s"”’\')\]]+$', "", regex=True)
    P["p_is_question"] = tail_stripped.str.endswith("?").fillna(False).astype("int8")

    steps_kw = r"\b(?:step[- ]?by[- ]?step|steps?|bullet(?:ed)?|checklist|enumerate|numbered list|procedure|instructions?|outline)\b"
    P["p_asks_steps"] = (lo.str.contains(steps_kw, regex=True, na=False) | (P["p_list_lines"] > 0)).astype("int8")

    langs = r"(?:python|java|javascript|typescript|c\+\+|c#|go|rust|ruby|php|sql|bash|powershell|kotlin|swift|matlab|r|scala|perl|haskell|lua|dart|c)"
    code_nouns = r"(?:code|function|method|class|script|snippet|program|algorithm|regex|query|api|unit test|unit tests|test case|module|package|library|endpoint)"
    code_verbs = r"(?:write|implement|provide|show|give|generate|create|produce|build|define|return|refactor)"
    W = proximity_window
    prox_verb_noun = rf"\b{code_verbs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b"
    prox_lang_noun = rf"\b{langs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b"
    prox_lang_verb = rf"\b{langs}\b[\s\S]{{0,{W}}}\b{code_verbs}\b"
    in_lang_phrase = rf"\b(?:in|using)\s+{langs}\b"
    P["p_asks_code"] = (
        (P["p_codefences"] > 0)
        | lo.str.contains(prox_verb_noun, regex=True, na=False)
        | lo.str.contains(prox_lang_noun, regex=True, na=False)
        | lo.str.contains(prox_lang_verb, regex=True, na=False)
        | lo.str.contains(rf"\b{code_nouns}\b\s+(?:example|sample)\b", regex=True, na=False)
        | lo.str.contains(rf"{in_lang_phrase}[\s\S]{{0,{W}}}\b{code_nouns}\b", regex=True, na=False)
    ).astype("int8")

    math_kw = r"\b(?:equation|solve|solution|derivative|integral|limit|matrix|vector|probability|statistics?|theorem|proof|algebra|calculus|gradient|expectation|variance|distribution)\b"
    latex = r"\$[^\$]+\$|\\\(|\\\)|\\begin\{equation"
    P["p_asks_math"] = (lo.str.contains(math_kw, regex=True, na=False) | s.str.contains(latex, regex=True, na=False)).astype("int8")

    advice_kw = (
        r"(?:\bwhat should i\b|\bhow should i\b|\bshould (?:i|we)\b|"
        r"\badvice\b|\badvise\b|\brecommend(?:ation)?s?\b|"
        r"\bpros and cons\b|\bis it (?:okay|ok|ethical|right|wrong|bad|good)\b|"
        r"\bmorally\b|\bwhat do you think\b)"
    )
    P["p_asks_advice"] = lo.str.contains(advice_kw, regex=True, na=False).astype("int8")

    compare_kw = r"(?:\bcompare\b|\bcomparison\b|\bdifference between\b|\bversus\b| vs\.? |\bwhich is better\b|\bbetter than\b)"
    P["p_compare"] = lo.str.contains(compare_kw, regex=True, na=False).astype("int8")

    summarize_kw = r"(?:\bsummariz(?:e|ation)\b|\bsummary\b|\btl;dr\b|\bcondense\b|\bbrief overview\b|\bkey points\b|\boutline the main points\b)"
    P["p_summarize"] = lo.str.contains(summarize_kw, regex=True, na=False).astype("int8")

    rewrite_kw = r"(?:\brewrite\b|\brephrase\b|\bparaphrase\b|\bpolish\b|\bedit for clarity\b|\bimprove (?:the )?writing\b|\bmake (?:it )?(?:formal|polite|concise)\b|\bfix grammar\b)"
    P["p_rewrite"] = lo.str.contains(rewrite_kw, regex=True, na=False).astype("int8")

    langs_words = r"(?:spanish|french|german|chinese|japanese|korean|hindi|arabic|portuguese|italian|russian|turkish|vietnamese|thai|indonesian|dutch|swedish|polish|greek)"
    translate_kw = rf"(?:\btranslate\b|\btranslate .* into (?:{langs_words})\b|\bto (?:{langs_words})\b)"
    P["p_translate"] = lo.str.contains(translate_kw, regex=True, na=False).astype("int8")

    classify_kw = r"(?:\bclassif(?:y|ication)\b|\blabel\b|\bcategorize\b|\bdetermine whether\b|\btrue or false\b|\byes or no\b|\bspam\b)"
    P["p_classify"] = lo.str.contains(classify_kw, regex=True, na=False).astype("int8")

    return P.astype(
        {
            "p_codefences": "int8",
            "p_bullets": "int8",
            "p_numlist": "int8",
            "p_list_lines": "int8",
            "p_is_question": "int8",
            "p_asks_steps": "int8",
            "p_asks_code": "int8",
            "p_asks_math": "int8",
            "p_asks_advice": "int8",
            "p_compare": "int8",
            "p_summarize": "int8",
            "p_rewrite": "int8",
            "p_translate": "int8",
            "p_classify": "int8",
        }
    )

def _mk_len_struct_useful(series, prefix):
    s = series.astype("string").fillna("")
    F = pd.DataFrame(index=s.index)
    F[f"{prefix}chars"] = s.str.len()
    F[f"{prefix}words"] = s.str.split().str.len()
    F[f"{prefix}sents"] = s.str.count(r"[.!?]+").clip(lower=1)
    F[f"{prefix}paragraphs"] = (s.str.count(r"\n\s*\n") + 1).where(F[f"{prefix}chars"] > 0, 0)
    F[f"{prefix}codefences"] = s.str.count(r"```")
    F[f"{prefix}headings"] = s.str.count(r"(?m)^\s*#{1,6}\s+")
    F[f"{prefix}bullets"] = s.str.count(r"(?m)^\s*[-*•]\s+")
    F[f"{prefix}numlist"] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    F[f"{prefix}list_lines"] = F[f"{prefix}bullets"] + F[f"{prefix}numlist"]
    F[f"{prefix}qmarks"] = s.str.count(r"\?")
    F[f"{prefix}exclaims"] = s.str.count(r"!")
    ww = F[f"{prefix}words"].replace(0, np.nan)
    for k in ["qmarks", "exclaims", "list_lines", "codefences", "headings"]:
        F[f"{prefix}{k}_per100w"] = (F[f"{prefix}{k}"] / ww * 100).fillna(0)
    return F

def build_lenstruct_useful(df, prompt_col="prompt_clean", resp_a_col="response_a_clean", resp_b_col="response_b_clean"):
    A = _mk_len_struct_useful(df[resp_a_col], "a_")
    B = _mk_len_struct_useful(df[resp_b_col], "b_")
    X = pd.concat([A, B], axis=1)
    kept_bases = [c[2:] for c in A.columns if c[2:] not in ("bullets", "numlist")]
    for k in kept_bases:
        X[f"diff_{k}"] = X[f"a_{k}"] - X[f"b_{k}"]
        X[f"ratio_{k}"] = (X[f"a_{k}"] + 1e-6) / (X[f"b_{k}"] + 1e-6)
    p_words = df[prompt_col].astype("string").str.split().str.len().replace(0, np.nan)
    X["a_to_prompt_word_ratio"] = (X["a_words"] / p_words).fillna(0)
    X["b_to_prompt_word_ratio"] = (X["b_words"] / p_words).fillna(0)
    X["a_longer_word"] = (X["a_words"] > X["b_words"]).astype("int8")
    X["a_longer_char"] = (X["a_chars"] > X["b_chars"]).astype("int8")
    return X.astype("float32", errors="ignore")

def compute_tfidf_cosine(a_texts, b_texts, max_features=50000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )
    combined = pd.Series(list(a_texts) + list(b_texts)).fillna("").astype(str).tolist()
    X = vec.fit_transform(combined)
    Xa = X[: len(a_texts)]
    Xb = X[len(a_texts) :]
    sims = cosine_similarity(Xa, Xb).diagonal()
    return sims.astype(np.float32)

def add_features(df):
    df = df.copy()

    df["parsed_prompt"] = df["prompt"].apply(safe_literal_eval)
    df["parsed_response_a"] = df["response_a"].apply(safe_literal_eval)
    df["parsed_response_b"] = df["response_b"].apply(safe_literal_eval)

    df["prompt_last_turn"] = df["parsed_prompt"].apply(get_last_turn)
    df["flat_prompt"] = df["parsed_prompt"].apply(safe_join)
    df["flat_response_a"] = df["parsed_response_a"].apply(safe_join)
    df["flat_response_b"] = df["parsed_response_b"].apply(safe_join)

    df["prompt_clean"] = df["prompt"].map(parse_list_first).map(clean_text)
    df["response_a_clean"] = df["response_a"].map(parse_list_first).map(clean_text)
    df["response_b_clean"] = df["response_b"].map(parse_list_first).map(clean_text)

    def tb_sent3(text):
        feats = get_textblob_features(text)
        pol = feats["textblob_polarity"]
        if pd.isna(pol):
            return (0.0, 1.0, 0.0)
        pos = float(max(pol, 0.0))
        neg = float(max(-pol, 0.0))
        neu = float(max(0.0, 1.0 - (pos + neg)))
        s = pos + neg + neu
        return (neg / s, neu / s, pos / s)

    sa = np.array([tb_sent3(t) for t in df["flat_response_a"].fillna("").astype(str).tolist()], dtype=np.float32)
    sb = np.array([tb_sent3(t) for t in df["flat_response_b"].fillna("").astype(str).tolist()], dtype=np.float32)

    feat = pd.DataFrame({"id": df["id"].values})
    feat["sentiment_negative_A"] = sa[:, 0]
    feat["sentiment_neutral_A"] = sa[:, 1]
    feat["sentiment_positive_A"] = sa[:, 2]
    feat["sentiment_negative_B"] = sb[:, 0]
    feat["sentiment_neutral_B"] = sb[:, 1]
    feat["sentiment_positive_B"] = sb[:, 2]

    prompt_texts = df["prompt_last_turn"].fillna("").astype(str).tolist()
    ra = df["flat_response_a"].fillna("").astype(str).tolist()
    rb = df["flat_response_b"].fillna("").astype(str).tolist()

    feat["semantic_similarity_A"] = compute_tfidf_cosine(prompt_texts, ra)
    feat["semantic_similarity_B"] = compute_tfidf_cosine(prompt_texts, rb)

    tba = df["flat_response_a"].fillna("").astype(str).map(get_textblob_features)
    tbb = df["flat_response_b"].fillna("").astype(str).map(get_textblob_features)
    feat["textblob_polarity_A"] = [d["textblob_polarity"] for d in tba]
    feat["textblob_subjectivity_A"] = [d["textblob_subjectivity"] for d in tba]
    feat["textblob_polarity_B"] = [d["textblob_polarity"] for d in tbb]
    feat["textblob_subjectivity_B"] = [d["textblob_subjectivity"] for d in tbb]

    rda = df["flat_response_a"].fillna("").astype(str).map(get_readability_scores)
    rdb = df["flat_response_b"].fillna("").astype(str).map(get_readability_scores)
    feat["flesch_kincaid_grade_A"] = [d["flesch_kincaid_grade"] for d in rda]
    feat["gunning_fog_A"] = [d["gunning_fog"] for d in rda]
    feat["flesch_kincaid_grade_B"] = [d["flesch_kincaid_grade"] for d in rdb]
    feat["gunning_fog_B"] = [d["gunning_fog"] for d in rdb]

    for base in [
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "textblob_polarity",
        "textblob_subjectivity",
        "flesch_kincaid_grade",
        "gunning_fog",
        "semantic_similarity",
    ]:
        a = f"{base}_A"
        b = f"{base}_B"
        if a in feat.columns and b in feat.columns:
            feat[f"{base}_diff_A_minus_B"] = feat[a] - feat[b]

    feat["len_prompt"] = df["parsed_prompt"].apply(lambda x: len(" ".join(x)) if isinstance(x, list) else len(str(x)))
    feat["len_response_a"] = df["flat_response_a"].fillna("").astype(str).map(len)
    feat["len_response_b"] = df["flat_response_b"].fillna("").astype(str).map(len)
    feat["len_diff_A_minus_B"] = feat["len_response_a"] - feat["len_response_b"]

    pstruct = build_prompt_purpose_strict(df, prompt_col="prompt_clean")
    lenstruct = build_lenstruct_useful(df, prompt_col="prompt_clean", resp_a_col="response_a_clean", resp_b_col="response_b_clean")
    feat = feat.merge(pd.concat([pstruct, lenstruct], axis=1).assign(id=df["id"].values), on="id", how="left")

    desired_order = [
        "id",
        "sentiment_negative_A", "sentiment_neutral_A", "sentiment_positive_A",
        "sentiment_negative_B", "sentiment_neutral_B", "sentiment_positive_B",
        "semantic_similarity_A", "semantic_similarity_B",
        "textblob_polarity_A", "textblob_subjectivity_A",
        "textblob_polarity_B", "textblob_subjectivity_B",
        "flesch_kincaid_grade_A", "gunning_fog_A",
        "flesch_kincaid_grade_B", "gunning_fog_B",
        "sentiment_positive_diff_A_minus_B",
        "sentiment_negative_diff_A_minus_B",
        "sentiment_neutral_diff_A_minus_B",
        "textblob_polarity_diff_A_minus_B",
        "textblob_subjectivity_diff_A_minus_B",
        "flesch_kincaid_grade_diff_A_minus_B",
        "gunning_fog_diff_A_minus_B",
        "semantic_similarity_diff_A_minus_B",
        "len_prompt", "len_response_a", "len_response_b", "len_diff_A_minus_B",
        "p_codefences", "p_bullets", "p_numlist", "p_list_lines", "p_is_question",
        "p_asks_steps", "p_asks_code", "p_asks_math", "p_asks_advice", "p_compare",
        "p_summarize", "p_rewrite", "p_translate", "p_classify",
        "a_chars", "a_words", "a_sents", "a_paragraphs",
        "a_codefences", "a_headings", "a_bullets", "a_numlist", "a_list_lines",
        "a_qmarks", "a_exclaims",
        "a_qmarks_per100w", "a_exclaims_per100w", "a_list_lines_per100w",
        "a_codefences_per100w", "a_headings_per100w",
        "b_chars", "b_words", "b_sents", "b_paragraphs",
        "b_codefences", "b_headings", "b_bullets", "b_numlist", "b_list_lines",
        "b_qmarks", "b_exclaims",
        "b_qmarks_per100w", "b_exclaims_per100w", "b_list_lines_per100w",
        "b_codefences_per100w", "b_headings_per100w",
        "diff_chars", "ratio_chars",
        "diff_words", "ratio_words",
        "diff_sents", "ratio_sents",
        "diff_paragraphs", "ratio_paragraphs",
        "diff_codefences", "ratio_codefences",
        "diff_headings", "ratio_headings",
        "diff_list_lines", "ratio_list_lines",
        "diff_qmarks", "ratio_qmarks",
        "diff_exclaims", "ratio_exclaims",
        "diff_qmarks_per100w", "ratio_qmarks_per100w",
        "diff_exclaims_per100w", "ratio_exclaims_per100w",
        "diff_list_lines_per100w", "ratio_list_lines_per100w",
        "diff_codefences_per100w", "ratio_codefences_per100w",
        "diff_headings_per100w", "ratio_headings_per100w",
        "a_to_prompt_word_ratio", "b_to_prompt_word_ratio",
        "a_longer_word", "a_longer_char",
    ]
    missing = [c for c in desired_order if c not in feat.columns]
    for c in missing:
        feat[c] = np.nan
    feat = feat[desired_order]
    return feat

start_time = time.time()

logger.info("Loading train/test...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

logger.info("Building features for train...")
train_feat = add_features(train_df)
logger.info(f"Train features shape: {train_feat.shape}")
train_feat.to_csv(FEATURE_OUTPUT_TRAIN, index=False)

logger.info("Building features for test...")
test_feat = add_features(test_df)
logger.info(f"Test features shape: {test_feat.shape}")
test_feat.to_csv(FEATURE_OUTPUT_TEST, index=False)

logger.info(f"Saved features to:\n- {FEATURE_OUTPUT_TRAIN}\n- {FEATURE_OUTPUT_TEST}")
logger.info(f"Elapsed feature time: {time.time() - start_time:.2f} seconds")



## === cell 10
import pandas as pd
import numpy as np
import xgboost as xgb

TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
ID_COL = "id"

df_train_feat = pd.read_csv(FEATURE_OUTPUT_TRAIN)
df_test_feat = pd.read_csv(FEATURE_OUTPUT_TEST)

train_raw = pd.read_csv(TRAIN_FILE, usecols=[ID_COL] + TARGET_COLS)
train_labels = train_raw.set_index(ID_COL).loc[df_train_feat[ID_COL].values, TARGET_COLS].values
y = np.argmax(train_labels, axis=1).astype(np.int32)

X_train = df_train_feat.drop(columns=[ID_COL])
X_test = df_test_feat.drop(columns=[ID_COL])

X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")
med = X_train.median(numeric_only=True)
X_train = X_train.fillna(med).fillna(0.0).astype(np.float32)
X_test = X_test.fillna(med).fillna(0.0).astype(np.float32)

dtrain = xgb.DMatrix(X_train, label=y, enable_categorical=False)
dtest = xgb.DMatrix(X_test, enable_categorical=False)

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "seed": RANDOM_STATE,
    "random_state": RANDOM_STATE,
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

num_boost_round = 500

trained = False
last_err = None
for tree_method in (["gpu_hist", "hist"] if _GPU_USABLE_HINT else ["hist"]):
    try:
        params_try = dict(params)
        params_try["tree_method"] = tree_method
        model = xgb.train(params_try, dtrain, num_boost_round=num_boost_round)
        trained = True
        print(f"Trained XGBoost with tree_method={tree_method}")
        break
    except Exception as e:
        last_err = e
        logger.warning(f"Training failed with tree_method={tree_method}: {e}")

if not trained:
    raise RuntimeError(f"XGBoost training failed for all attempted tree_method values. Last error: {last_err}")

preds = model.predict(dtest)  # (n,3)

preds = np.clip(preds, 1e-15, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

sample = pd.read_csv(SUBMISSION_SAMPLE)
pred_df = pd.DataFrame({ID_COL: df_test_feat[ID_COL].values})
pred_df[TARGET_COLS] = preds

submission = sample[[ID_COL]].merge(pred_df, on=ID_COL, how="left")

for c in TARGET_COLS:
    if c not in submission.columns:
        submission[c] = np.nan

submission[TARGET_COLS] = submission[TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(1.0 / 3.0)

vals = np.clip(submission[TARGET_COLS].to_numpy(dtype=np.float64), 1e-15, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
submission[TARGET_COLS] = vals

submission = submission[[ID_COL] + TARGET_COLS]
submission.to_csv(SUBMISSION_PATH, index=False)

print(submission.head())
print(f"Submission saved to {SUBMISSION_PATH} with shape: {submission.shape}")



## === cell 11
import pandas as pd
import numpy as np

sub = pd.read_csv(SUBMISSION_PATH)
required = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
assert list(sub.columns) == required, sub.columns

test_rows = pd.read_csv(TEST_FILE, usecols=["id"]).shape[0]
assert sub.shape[0] == test_rows, (sub.shape, test_rows)

row_sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
print("Row sum stats:", float(row_sums.min()), float(row_sums.max()))
assert np.isfinite(row_sums).all()
print("Done.")
```

## --- ERROR in cell 11, traceback:
  File "/tmp/ipykernel_55/2609893636.py", line 15
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame is missing required columns: ['winner_model_a', 'winner_model_b', 'winner_tie']
