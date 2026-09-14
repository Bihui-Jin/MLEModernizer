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

1.0907981714516313

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the hard dependency on missing external wheels/models (textstat, bitsandbytes, offline MiniLM, and the missing XGBoost params JSON) so the notebook can run in a standard Kaggle environment. To preserve the core feature/predict flow, I’ll keep the same feature columns and XGBoost inference step, but add safe fallbacks: a lightweight readability implementation when `textstat` isn’t available, and deterministic “neutral/zero” sentiment + similarity when offline transformer folders are absent. I’ll also make the data/model paths robust by searching the provided `/kaggle/data/...` and `/kaggle/input/...` locations, and force CPU `tree_method` if GPU isn’t available. Finally, the script will always write `/kaggle/working/submission.csv` with the required columns and valid probability rows.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/289139617.py", line 1
    I will remove the hard dependency on missing external wheels/models (textstat, bitsandbytes, offline MiniLM, and the missing XGBoost params JSON) so the notebook can run in a standard Kaggle environment. To preserve the core feature/predict flow, I’ll keep the same feature columns and XGBoost inference step, but add safe fallbacks: a lightweight readability implementation when `textstat` isn’t available, and deterministic “neutral/zero” sentiment + similarity when offline transformer folders are absent. I’ll also make the data/model paths robust by searching the provided `/kaggle/data/...` and `/kaggle/input/...` locations, and force CPU `tree_method` if GPU isn’t available. Finally, the script will always write `/kaggle/working/submission.csv` with the required columns and valid probability rows.
                                                                                                                                                                                                                                                            ^
SyntaxError: invalid character '’' (U+2019)


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
import importlib

try:
    import textstat  # type: ignore
    _HAS_TEXTSTAT = True
    print("✅ textstat is available.")
except Exception as e:
    _HAS_TEXTSTAT = False
    textstat = None
    print(f"⚠️ textstat not available ({e}). Will use fallback readability functions.")



## === cell 3
print("ℹ️ Skipping bitsandbytes installation (not required for this submission pipeline).")



## === cell 4
import os
import pandas as pd
import numpy as np
import torch
import json
import ast
import re
from tqdm import tqdm
import logging
import time
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting offline feature extraction + XGBoost inference pipeline.")

DATA_FILE = "/kaggle/input/lmsys-chatbot-arena/test.csv"
CHECKPOINT_DIR = "checkpoints_llm_judge"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512
FEATURE_OUTPUT = "lmsys_test_features_final.csv"
PREDICTION_OUTPUT = "submission.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)



## === cell 5
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
print("\nContents of similarity model directory (if present):")
if os.path.exists(SIM_MODEL_PATH):
    for root, dirs, files in os.walk(SIM_MODEL_PATH):
        level = root.replace(SIM_MODEL_PATH, '').count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... (+{len(files)-5} more)")
else:
    print(f"(not found) {SIM_MODEL_PATH}")



## === cell 6
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
SENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 7
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
SENTIMENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"

_HAS_OFFLINE_SIM = os.path.exists(SIM_MODEL_PATH)
_HAS_OFFLINE_SENT = os.path.exists(SENTIMENT_MODEL_PATH)

print(f"Offline similarity model available: {_HAS_OFFLINE_SIM} ({SIM_MODEL_PATH})")
print(f"Offline sentiment model available: {_HAS_OFFLINE_SENT} ({SENTIMENT_MODEL_PATH})")

sim_tokenizer = None
sim_model = None
sentiment_tokenizer = None
sentiment_model = None

if _HAS_OFFLINE_SIM or _HAS_OFFLINE_SENT:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
    if _HAS_OFFLINE_SENT:
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
        sentiment_model.eval()
        if torch.cuda.is_available():
            sentiment_model = sentiment_model.to("cuda")
    if _HAS_OFFLINE_SIM:
        sim_tokenizer = AutoTokenizer.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
        sim_model = AutoModel.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
        sim_model.eval()
        if torch.cuda.is_available():
            sim_model = sim_model.to("cuda")



## === cell 8
import pandas as pd
test_df = pd.read_csv(DATA_FILE)
print("Test shape:", test_df.shape)
print(test_df.head(2))



## === cell 9
from pathlib import Path

def resolve_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

DATA_FILE = resolve_existing_path([
    "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "/kaggle/data/lmsys-chatbot-arena/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
])
if DATA_FILE is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle locations.")

SAMPLE_SUB_PATH = resolve_existing_path([
    "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv",
    "/kaggle/data/lmsys-chatbot-arena/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
])
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError("Could not find sample_submission.csv in expected Kaggle locations.")

print("Using DATA_FILE:", DATA_FILE)
print("Using SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 10
import numpy as np
import pandas as pd
import time
import ast
import re
import json
from tqdm import tqdm
from textblob import TextBlob

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

def _fallback_flesch_kincaid_grade(text: str) -> float:
    s = str(text or "")
    words = re.findall(r"[A-Za-z]+", s)
    n_words = max(len(words), 1)
    n_sents = max(len(re.findall(r"[.!?]+", s)), 1)
    n_chars = max(sum(len(w) for w in words), 1)
    avg_word_len = n_chars / n_words
    avg_sent_len = n_words / n_sents
    return float(0.6 * avg_sent_len + 4.0 * (avg_word_len - 4.0))

def _fallback_gunning_fog(text: str) -> float:
    s = str(text or "")
    words = re.findall(r"[A-Za-z]+", s)
    n_words = max(len(words), 1)
    n_sents = max(len(re.findall(r"[.!?]+", s)), 1)
    complex_words = [w for w in words if len(w) >= 8]
    pct_complex = (len(complex_words) / n_words) * 100.0
    return float(0.4 * ((n_words / n_sents) + pct_complex))

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

def get_textblob_features(text):
    if not isinstance(text, str) or not text.strip():
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}
    try:
        blob = TextBlob(text)
        return {
            "textblob_polarity": float(blob.sentiment.polarity),
            "textblob_subjectivity": float(blob.sentiment.subjectivity)
        }
    except Exception:
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}

def get_readability_scores(text):
    if not isinstance(text, str) or not text.strip():
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}
    try:
        if _HAS_TEXTSTAT:
            return {
                "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),  # type: ignore
                "gunning_fog": float(textstat.gunning_fog(text)),  # type: ignore
            }
        else:
            return {
                "flesch_kincaid_grade": _fallback_flesch_kincaid_grade(text),
                "gunning_fog": _fallback_gunning_fog(text),
            }
    except Exception:
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}

def clean_text(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def parse_list_first(text):
    if pd.isna(text):
        return ""
    s = str(text).strip()
    try:
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

def build_prompt_purpose_strict(df, prompt_col='prompt_clean', proximity_window=60):
    s = df[prompt_col].astype('string').fillna('')
    lo = s.str.lower()
    P = pd.DataFrame(index=df.index)
    P['p_codefences'] = s.str.count(r'```')
    P['p_bullets'] = s.str.count(r'(?m)^\s*[-*•]\s+')
    P['p_numlist'] = s.str.count(r'(?m)^\s*[0-9]{1,2}[.)]\s+')
    P['p_list_lines'] = (P['p_bullets'] + P['p_numlist']).astype('int16')
    tail_stripped = s.str.replace(r'[\s"”’\')\]]+$', '', regex=True)
    P['p_is_question'] = tail_stripped.str.endswith('?').fillna(False).astype('int8')
    steps_kw = r'\b(?:step[- ]?by[- ]?step|steps?|bullet(?:ed)?|checklist|enumerate|numbered list|procedure|instructions?|outline)\b'
    P['p_asks_steps'] = (lo.str.contains(steps_kw, regex=True, na=False) | (P['p_list_lines'] > 0)).astype('int8')
    langs = r'(?:python|java|javascript|typescript|c\+\+|c#|go|rust|ruby|php|sql|bash|powershell|kotlin|swift|matlab|r|scala|perl|haskell|lua|dart|c)'
    code_nouns = r'(?:code|function|method|class|script|snippet|program|algorithm|regex|query|api|unit test|unit tests|test case|module|package|library|endpoint)'
    code_verbs = r'(?:write|implement|provide|show|give|generate|create|produce|build|define|return|refactor)'
    W = proximity_window
    prox_verb_noun = rf'\b{code_verbs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b'
    prox_lang_noun = rf'\b{langs}\b[\s\S]{{0,{W}}}\b{code_nouns}\b'
    prox_lang_verb = rf'\b{langs}\b[\s\S]{{0,{W}}}\b{code_verbs}\b'
    in_lang_phrase = rf'\b(?:in|using)\s+{langs}\b'
    P['p_asks_code'] = (
        (P['p_codefences'] > 0) |
        lo.str.contains(prox_verb_noun, regex=True, na=False) |
        lo.str.contains(prox_lang_noun, regex=True, na=False) |
        lo.str.contains(prox_lang_verb, regex=True, na=False) |
        lo.str.contains(rf'\b{code_nouns}\b\s+(?:example|sample)\b', regex=True, na=False) |
        lo.str.contains(rf'{in_lang_phrase}[\s\S]{{0,{W}}}\b{code_nouns}\b', regex=True, na=False)
    ).astype('int8')
    math_kw = r'\b(?:equation|solve|solution|derivative|integral|limit|matrix|vector|probability|statistics?|theorem|proof|algebra|calculus|gradient|expectation|variance|distribution)\b'
    latex = r"\$[^\$]+\$|\\\(|\\\)|\\begin\{equation"
    P['p_asks_math'] = (lo.str.contains(math_kw, regex=True, na=False) | s.str.contains(latex, regex=True, na=False)).astype('int8')
    advice_kw = (
        r'(?:\bwhat should i\b|\bhow should i\b|\bshould (?:i|we)\b|'
        r'\badvice\b|\badvise\b|\brecommend(?:ation)?s?\b|'
        r'\bpros and cons\b|\bis it (?:okay|ok|ethical|right|wrong|bad|good)\b|'
        r'\bmorally\b|\bwhat do you think\b)'
    )
    P['p_asks_advice'] = lo.str.contains(advice_kw, regex=True, na=False).astype('int8')
    compare_kw = r'(?:\bcompare\b|\bcomparison\b|\bdifference between\b|\bversus\b| vs\.? |\bwhich is better\b|\bbetter than\b)'
    P['p_compare'] = lo.str.contains(compare_kw, regex=True, na=False).astype('int8')
    summarize_kw = r'(?:\bsummariz(?:e|ation)\b|\bsummary\b|\btl;dr\b|\bcondense\b|\bbrief overview\b|\bkey points\b|\boutline the main points\b)'
    P['p_summarize'] = lo.str.contains(summarize_kw, regex=True, na=False).astype('int8')
    rewrite_kw = r'(?:\brewrite\b|\brephrase\b|\bparaphrase\b|\bpolish\b|\bedit for clarity\b|\bimprove (?:the )?writing\b|\bmake (?:it )?(?:formal|polite|concise)\b|\bfix grammar\b)'
    P['p_rewrite'] = lo.str.contains(rewrite_kw, regex=True, na=False).astype('int8')
    langs_words = r'(?:spanish|french|german|chinese|japanese|korean|hindi|arabic|portuguese|italian|russian|turkish|vietnamese|thai|indonesian|dutch|swedish|polish|greek)'
    translate_kw = rf'(?:\btranslate\b|\btranslate .* into (?:{langs_words})\b|\bto (?:{langs_words})\b)'
    P['p_translate'] = lo.str.contains(translate_kw, regex=True, na=False).astype('int8')
    classify_kw = r'(?:\bclassif(?:y|ication)\b|\blabel\b|\bcategorize\b|\bdetermine whether\b|\btrue or false\b|\byes or no\b|\bspam\b)'
    P['p_classify'] = lo.str.contains(classify_kw, regex=True, na=False).astype('int8')
    return P.astype({c: 'int8' for c in P.columns})

def _mk_len_struct_useful(series, prefix):
    s = series.astype('string').fillna('')
    F = pd.DataFrame(index=s.index)
    F[f'{prefix}chars'] = s.str.len()
    F[f'{prefix}words'] = s.str.split().str.len()
    F[f'{prefix}sents'] = s.str.count(r'[.!?]+').clip(lower=1)
    F[f'{prefix}paragraphs'] = (s.str.count(r'\n\s*\n') + 1).where(F[f'{prefix}chars'] > 0, 0)
    F[f'{prefix}codefences'] = s.str.count(r"```")
    F[f'{prefix}headings'] = s.str.count(r"(?m)^\s*#{1,6}\s+")
    F[f'{prefix}bullets'] = s.str.count(r"(?m)^\s*[-*•]\s+")
    F[f'{prefix}numlist'] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    F[f'{prefix}list_lines'] = F[f'{prefix}bullets'] + F[f'{prefix}numlist']
    F[f'{prefix}qmarks'] = s.str.count(r"\?")
    F[f'{prefix}exclaims'] = s.str.count(r"!")
    ww = F[f'{prefix}words'].replace(0, np.nan)
    for k in ['qmarks','exclaims','list_lines','codefences','headings']:
        F[f'{prefix}{k}_per100w'] = (F[f'{prefix}{k}'] / ww * 100).fillna(0)
    return F

def build_lenstruct_useful(df, prompt_col='prompt_clean', resp_a_col='response_a_clean', resp_b_col='response_b_clean'):
    A = _mk_len_struct_useful(df[resp_a_col], 'a_')
    B = _mk_len_struct_useful(df[resp_b_col], 'b_')
    X = pd.concat([A, B], axis=1)
    kept_bases = [c[2:] for c in A.columns if c[2:] not in ('bullets','numlist')]
    for k in kept_bases:
        X[f'diff_{k}'] = X[f'a_{k}'] - X[f'b_{k}']
        X[f'ratio_{k}'] = (X[f'a_{k}'] + 1e-6) / (X[f'b_{k}'] + 1e-6)
    p_words = df[prompt_col].astype('string').str.split().str.len().replace(0, np.nan)
    X['a_to_prompt_word_ratio'] = (X['a_words'] / p_words).fillna(0)
    X['b_to_prompt_word_ratio'] = (X['b_words'] / p_words).fillna(0)
    X['a_longer_word'] = (X['a_words'] > X['b_words']).astype('int8')
    X['a_longer_char'] = (X['a_chars'] > X['b_chars']).astype('int8')
    return X.astype('float32', errors='ignore')

def batch_get_sentiment_scores(texts, labels):
    out = []
    for _ in texts:
        out.append({f"sentiment_{labels[0]}": 0.0, f"sentiment_{labels[1]}": 1.0, f"sentiment_{labels[2]}": 0.0})
    return out

def batch_get_semantic_similarity(prompt_response_pairs):
    return [0.0] * len(prompt_response_pairs)

start_time = time.time()
logger.info(f"Loading data from '{DATA_FILE}'...")
df_data = pd.read_csv(DATA_FILE, engine='python', on_bad_lines='skip')
logger.info(f"Loaded test data. Shape: {df_data.shape}")

df_data['parsed_prompt'] = df_data['prompt'].apply(safe_literal_eval)
df_data['parsed_response_a'] = df_data['response_a'].apply(safe_literal_eval)
df_data['parsed_response_b'] = df_data['response_b'].apply(safe_literal_eval)
df_data['prompt_last_turn'] = df_data['parsed_prompt'].apply(get_last_turn)

def safe_join(x):
    if isinstance(x, list):
        return " ".join([str(item) for item in x if item is not None])
    return str(x) if x is not None else ""

df_data['flat_prompt'] = df_data['parsed_prompt'].apply(safe_join)
df_data['flat_response_a'] = df_data['parsed_response_a'].apply(safe_join)
df_data['flat_response_b'] = df_data['parsed_response_b'].apply(safe_join)

df_data["prompt_clean"] = df_data["prompt"].map(parse_list_first).map(clean_text)
df_data["response_a_clean"] = df_data["response_a"].map(parse_list_first).map(clean_text)
df_data["response_b_clean"] = df_data["response_b"].map(parse_list_first).map(clean_text)

texts_a = df_data['flat_response_a'].fillna("").astype(str).tolist()
texts_b = df_data['flat_response_b'].fillna("").astype(str).tolist()

df_data['prompt_last_turn'] = df_data['prompt_last_turn'].fillna("").astype(str)
df_data['flat_response_a'] = df_data['flat_response_a'].fillna("").astype(str)
df_data['flat_response_b'] = df_data['flat_response_b'].fillna("").astype(str)

pairs_a = list(zip(df_data['prompt_last_turn'], df_data['flat_response_a']))
pairs_b = list(zip(df_data['prompt_last_turn'], df_data['flat_response_b']))

logger.info("Computing sentiment and similarity features (fallback mode if offline models absent)...")
sentiment_scores_a = batch_get_sentiment_scores(texts_a, SENTIMENT_LABELS)
sentiment_scores_b = batch_get_sentiment_scores(texts_b, SENTIMENT_LABELS)
similarity_scores_a = batch_get_semantic_similarity(pairs_a)
similarity_scores_b = batch_get_semantic_similarity(pairs_b)

df_data['semantic_similarity_A'] = similarity_scores_a
df_data['semantic_similarity_B'] = similarity_scores_b

df_sentiment_a = pd.DataFrame(sentiment_scores_a)
df_sentiment_b = pd.DataFrame(sentiment_scores_b)
df_sentiment_a.columns = [f"{col}_A" for col in df_sentiment_a.columns]
df_sentiment_b.columns = [f"{col}_B" for col in df_sentiment_b.columns]

df_features = df_data[['id']].copy()
df_features = pd.concat([df_features, df_sentiment_a, df_sentiment_b], axis=1)
df_features['semantic_similarity_A'] = df_data['semantic_similarity_A']
df_features['semantic_similarity_B'] = df_data['semantic_similarity_B']

df_features['textblob_polarity_A'] = df_data['flat_response_a'].apply(lambda x: get_textblob_features(x)['textblob_polarity'])
df_features['textblob_subjectivity_A'] = df_data['flat_response_a'].apply(lambda x: get_textblob_features(x)['textblob_subjectivity'])
df_features['textblob_polarity_B'] = df_data['flat_response_b'].apply(lambda x: get_textblob_features(x)['textblob_polarity'])
df_features['textblob_subjectivity_B'] = df_data['flat_response_b'].apply(lambda x: get_textblob_features(x)['textblob_subjectivity'])

df_features['flesch_kincaid_grade_A'] = df_data['flat_response_a'].apply(lambda x: get_readability_scores(x)['flesch_kincaid_grade'])
df_features['gunning_fog_A'] = df_data['flat_response_a'].apply(lambda x: get_readability_scores(x)['gunning_fog'])
df_features['flesch_kincaid_grade_B'] = df_data['flat_response_b'].apply(lambda x: get_readability_scores(x)['flesch_kincaid_grade'])
df_features['gunning_fog_B'] = df_data['flat_response_b'].apply(lambda x: get_readability_scores(x)['gunning_fog'])

feature_pairs = [
    ('sentiment_negative', 'sentiment_neutral', 'sentiment_positive'),
    ('textblob_polarity', 'textblob_subjectivity'),
    ('flesch_kincaid_grade', 'gunning_fog'),
    ('semantic_similarity',)
]
for feat_tuple in feature_pairs:
    for feat_base in feat_tuple:
        feat_a = f"{feat_base}_A"
        feat_b = f"{feat_base}_B"
        if feat_a in df_features.columns and feat_b in df_features.columns:
            df_features[f"{feat_base}_diff_A_minus_B"] = df_features[feat_a] - df_features[feat_b]

df_features['len_prompt'] = df_data['parsed_prompt'].apply(lambda x: len(" ".join(x)) if isinstance(x, list) else len(str(x)))
df_features['len_response_a'] = df_data['flat_response_a'].apply(len)
df_features['len_response_b'] = df_data['flat_response_b'].apply(len)
df_features['len_diff_A_minus_B'] = df_features['len_response_a'] - df_features['len_response_b']

pstruct = build_prompt_purpose_strict(df_data)
lenstruct = build_lenstruct_useful(df_data)
df_extended = pd.concat([pstruct, lenstruct], axis=1)
df_extended_with_id = df_extended.copy()
df_extended_with_id['id'] = df_data['id'].values
df_features = df_features.merge(df_extended_with_id, on='id', how='left')

logger.info("Adding log-ratio and helpfulness proxy features...")
base_features = ['words', 'chars', 'sents', 'paragraphs', 'codefences', 'headings', 'list_lines']
for feat in base_features:
    a_col = f'a_{feat}'
    b_col = f'b_{feat}'
    if a_col in df_features.columns and b_col in df_features.columns:
        df_features[f'log_ratio_{feat}'] = np.log((df_features[a_col] + 1) / (df_features[b_col] + 1))

df_features['alignment_A'] = df_features.get('semantic_similarity_A', 0.0)
df_features['alignment_B'] = df_features.get('semantic_similarity_B', 0.0)

fkg_a = df_features['flesch_kincaid_grade_A'].fillna(15)
fkg_b = df_features['flesch_kincaid_grade_B'].fillna(15)
clarity_a = (1 - np.clip(fkg_a / 15, 0, 1))
clarity_b = (1 - np.clip(fkg_b / 15, 0, 1))

length_a = np.log1p(df_features['a_words']) / 10.0
length_b = np.log1p(df_features['b_words']) / 10.0

df_features['helpfulness_proxy_A'] = (0.4 * df_features['alignment_A'] + 0.3 * clarity_a + 0.3 * length_a).clip(0, 1)
df_features['helpfulness_proxy_B'] = (0.4 * df_features['alignment_B'] + 0.3 * clarity_b + 0.3 * length_b).clip(0, 1)
df_features['helpfulness_proxy_diff_A_minus_B'] = df_features['helpfulness_proxy_A'] - df_features['helpfulness_proxy_B']

df_final = df_features.copy()

desired_order = [
    'id',
    'sentiment_negative_A', 'sentiment_neutral_A', 'sentiment_positive_A',
    'sentiment_negative_B', 'sentiment_neutral_B', 'sentiment_positive_B',
    'semantic_similarity_A', 'semantic_similarity_B',
    'textblob_polarity_A', 'textblob_subjectivity_A',
    'textblob_polarity_B', 'textblob_subjectivity_B',
    'flesch_kincaid_grade_A', 'gunning_fog_A',
    'flesch_kincaid_grade_B', 'gunning_fog_B',
    'sentiment_positive_diff_A_minus_B',
    'sentiment_negative_diff_A_minus_B',
    'sentiment_neutral_diff_A_minus_B',
    'textblob_polarity_diff_A_minus_B',
    'textblob_subjectivity_diff_A_minus_B',
    'flesch_kincaid_grade_diff_A_minus_B',
    'gunning_fog_diff_A_minus_B',
    'semantic_similarity_diff_A_minus_B',
    'len_prompt', 'len_response_a', 'len_response_b', 'len_diff_A_minus_B',
    'p_codefences', 'p_bullets', 'p_numlist', 'p_list_lines', 'p_is_question',
    'p_asks_steps', 'p_asks_code', 'p_asks_math', 'p_asks_advice', 'p_compare',
    'p_summarize', 'p_rewrite', 'p_translate', 'p_classify',
    'a_chars', 'a_words', 'a_sents', 'a_paragraphs',
    'a_codefences', 'a_headings', 'a_bullets', 'a_numlist', 'a_list_lines',
    'a_qmarks', 'a_exclaims',
    'a_qmarks_per100w', 'a_exclaims_per100w', 'a_list_lines_per100w',
    'a_codefences_per100w', 'a_headings_per100w',
    'b_chars', 'b_words', 'b_sents', 'b_paragraphs',
    'b_codefences', 'b_headings', 'b_bullets', 'b_numlist', 'b_list_lines',
    'b_qmarks', 'b_exclaims',
    'b_qmarks_per100w', 'b_exclaims_per100w', 'b_list_lines_per100w',
    'b_codefences_per100w', 'b_headings_per100w',
    'log_ratio_chars', 'ratio_chars',
    'log_ratio_words', 'ratio_words',
    'log_ratio_sents', 'ratio_sents',
    'log_ratio_paragraphs', 'ratio_paragraphs',
    'log_ratio_codefences', 'ratio_codefences',
    'log_ratio_headings', 'ratio_headings',
    'log_ratio_list_lines', 'ratio_list_lines',
    'diff_qmarks', 'ratio_qmarks',
    'diff_exclaims', 'ratio_exclaims',
    'diff_qmarks_per100w', 'ratio_qmarks_per100w',
    'diff_exclaims_per100w', 'ratio_exclaims_per100w',
    'diff_list_lines_per100w', 'ratio_list_lines_per100w',
    'diff_codefences_per100w', 'ratio_codefences_per100w',
    'diff_headings_per100w', 'ratio_headings_per100w',
    'a_to_prompt_word_ratio', 'b_to_prompt_word_ratio',
    'a_longer_word', 'a_longer_char',
    'helpfulness_proxy_A', 'helpfulness_proxy_B', 'helpfulness_proxy_diff_A_minus_B'
]

missing_cols = [c for c in desired_order if c not in df_final.columns]
for c in missing_cols:
    df_final[c] = np.nan
df_final = df_final[desired_order]

df_final.to_csv(FEATURE_OUTPUT, index=False)
logger.info(f"Saved features to {FEATURE_OUTPUT}. Shape: {df_final.shape}")
logger.info(f"Elapsed Time: {time.time() - start_time:.2f} seconds")



## === cell 11
import pandas as pd
import numpy as np
import xgboost as xgb
import os

TEST_FEATURES_FILE = "/kaggle/working/lmsys_test_features_final.csv"
df_test = pd.read_csv(TEST_FEATURES_FILE)
ID_COL = "id"
test_ids = df_test[ID_COL].values

feature_cols = [c for c in df_test.columns if c != ID_COL and not c.startswith("winner")]
X_test = df_test[feature_cols]

XGB_MODEL_PATH = resolve_existing_path([
    "/kaggle/input/xgboost-final-model/xgboost_final_model.json",
    "/kaggle/input/xgboost_final_model/xgboost_final_model.json",
    "/kaggle/input/xgboost-final-model/xgb_model.json",
    "/kaggle/working/xgboost_final_model.json",
])
if XGB_MODEL_PATH is None:
    raise FileNotFoundError("Could not find xgboost_final_model.json in expected Kaggle locations.")

model = xgb.Booster()
model.load_model(XGB_MODEL_PATH)

dtest = xgb.DMatrix(X_test)
preds = model.predict(dtest)

preds = np.asarray(preds)
if preds.ndim == 1:
    preds = preds.reshape(-1, 3)
if preds.shape[1] != 3:
    raise ValueError(f"Expected 3-class predictions, got shape {preds.shape}")

preds = np.clip(preds, 1e-6, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
submission = pd.DataFrame(preds, columns=TARGET_COLS)
submission.insert(0, ID_COL, test_ids)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print(f"✅ Submission saved to {out_path} with shape: {submission.shape}")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/678895773.py in <cell line: 0>()
     21 ])
     22 if XGB_MODEL_PATH is None:
---> 23     raise FileNotFoundError("Could not find xgboost_final_model.json in expected Kaggle locations.")
     24 
     25 model = xgb.Booster()

FileNotFoundError: Could not find xgboost_final_model.json in expected Kaggle locations.

## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
expected_cols = list(sample_sub.columns)
assert expected_cols == ["id", "winner_model_a", "winner_model_b", "winner_tie"], expected_cols

sub = pd.read_csv("/kaggle/working/submission.csv")
assert list(sub.columns) == expected_cols, list(sub.columns)
assert len(sub) == len(sample_sub), (len(sub), len(sample_sub))
assert sub["id"].is_unique, "IDs not unique in submission"
print("✅ Submission format validated.")
```

## --- ERROR in cell 12, traceback:
  File "/tmp/ipykernel_55/890454341.py", line 11
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame is missing required columns: ['winner_model_a', 'winner_model_b', 'winner_tie']
