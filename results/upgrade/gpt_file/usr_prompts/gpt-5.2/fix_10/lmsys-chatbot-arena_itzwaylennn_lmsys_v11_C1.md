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
I will remove the non-Python prose/backticks that are being executed as code (the root cause of the SyntaxErrors) by turning them into comments and ensuring there are no stray ``` tokens. Then I will make submission creation unconditional and robust by always producing the required columns (`winner_model_a`, `winner_model_b`, `winner_tie`) and writing `/kaggle/working/submission.csv` with a `.csv` suffix. To keep core modeling/feature logic unchanged, I won’t alter any feature engineering or model prediction math; I’ll only fix the column-name mismatch that caused the “missing required columns” error when the XGBoost model is absent. Finally, I’ll keep the existing input-path resolver but ensure IDs align with `test.csv` order so the submission is valid.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/1586724877.py", line 1
    I will remove the non-Python prose/backticks that are being executed as code (the root cause of the SyntaxErrors) by turning them into comments and ensuring there are no stray ``` tokens. Then I will make submission creation unconditional and robust by always producing the required columns (`winner_model_a`, `winner_model_b`, `winner_tie`) and writing `/kaggle/working/submission.csv` with a `.csv` suffix. To keep core modeling/feature logic unchanged, I won’t alter any feature engineering or model prediction math; I’ll only fix the column-name mismatch that caused the “missing required columns” error when the XGBoost model is absent. Finally, I’ll keep the existing input-path resolver but ensure IDs align with `test.csv` order so the submission is valid.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
SyntaxError: invalid character '’' (U+2019)


## === cell 1

import logging
import warnings
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow/CUDA if used

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Attempting to register.*factory")
warnings.filterwarnings("ignore", message="The 'repr' attribute.*")




## === cell 2
import types

try:
    import textstat  # type: ignore
    _HAS_TEXTSTAT = True
except Exception:
    _HAS_TEXTSTAT = False

    def _fallback_flesch_kincaid_grade(text: str) -> float:
        return float("nan")

    def _fallback_gunning_fog(text: str) -> float:
        return float("nan")

    textstat = types.SimpleNamespace(
        flesch_kincaid_grade=_fallback_flesch_kincaid_grade,
        gunning_fog=_fallback_gunning_fog,
    )

print(f"textstat available: {_HAS_TEXTSTAT}")




## === cell 3
print("Skipping bitsandbytes installation (optional dependency not available).")




## === cell 4
import pandas as pd
import numpy as np
import torch
import json
import ast
import re
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting offline feature extraction + XGBoost prediction pipeline.")

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

DATA_FILE = _first_existing([
    "/kaggle/input/lmsys-chatbot-arena/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/lmsys-chatbot-arena/test.csv",
    "/kaggle/data/test.csv",
])
if DATA_FILE is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle locations.")

CHECKPOINT_DIR = "checkpoints_llm_judge"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512

SIMILARITY_MODEL_PATH = _first_existing([
    "/kaggle/input/all-minilm-l6-v2-offline",
])
SENTIMENT_MODEL_PATH = _first_existing([
    "/kaggle/input/twitter-roberta-sentiment-offline",
])

FEATURE_OUTPUT = "/kaggle/working/lmsys_test_features_final.csv"
PREDICTION_OUTPUT = "/kaggle/working/submission.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)




## === cell 5
if SIMILARITY_MODEL_PATH and os.path.exists(SIMILARITY_MODEL_PATH):
    print("\nContents of similarity model directory:")
    for root, dirs, files in os.walk(SIMILARITY_MODEL_PATH):
        level = root.replace(SIMILARITY_MODEL_PATH, '').count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... (+{len(files)-5} more)")
else:
    print(f"Similarity model path not found (will use fallback): {SIMILARITY_MODEL_PATH}")




## === cell 6
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)




## === cell 7
sim_tokenizer = None
sim_model = None
_HAS_SIM_MODEL = False

if SIMILARITY_MODEL_PATH and os.path.exists(SIMILARITY_MODEL_PATH):
    print(f"Loading similarity model from: {SIMILARITY_MODEL_PATH}")
    sim_tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
    sim_model = AutoModel.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
    sim_model.to(device)
    sim_model.eval()
    _HAS_SIM_MODEL = True
    print("✅ Similarity model loaded successfully.")
else:
    print("Similarity model not available; semantic similarity features will default to 0.0.")




## === cell 8
test_df = pd.read_csv(DATA_FILE)
print("Test shape:", test_df.shape)
print(test_df.head())




## === cell 9
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

logger.info(f"Loading data from '{DATA_FILE}'...")
df_data = pd.read_csv(DATA_FILE, engine='python', on_bad_lines='skip')
logger.info(f"Loaded test data. Shape: {df_data.shape}")

sentiment_tokenizer = None
sentiment_model = None
_HAS_SENTIMENT_MODEL = False

if SENTIMENT_MODEL_PATH and os.path.exists(SENTIMENT_MODEL_PATH):
    logger.info(f"Loading Sentiment model from: {SENTIMENT_MODEL_PATH}")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
    sentiment_model.eval()
    sentiment_model.to(device)
    _HAS_SENTIMENT_MODEL = True
    logger.info("✅ Sentiment model loaded.")
else:
    logger.warning(f"Sentiment model path not found; using neutral fallback: {SENTIMENT_MODEL_PATH}")

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentences(sentences, model, tokenizer, batch_size=32, device=device):
    cleaned = []
    for sent in sentences:
        if sent is None or pd.isna(sent):
            cleaned.append("")
        else:
            cleaned.append(str(sent).strip())

    embeddings = []
    batches = [cleaned[i:i+batch_size] for i in range(0, len(cleaned), batch_size)]
    model.to(device)

    with torch.no_grad():
        for batch_texts in tqdm(batches, desc="Encoding Batches"):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            out = model(**encoded)
            pooled = mean_pooling(out.last_hidden_state, encoded['attention_mask'])
            pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

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
    except Exception as e:
        logger.warning(f"TextBlob error: {e}")
        return {"textblob_polarity": np.nan, "textblob_subjectivity": np.nan}

def get_readability_scores(text):
    if not isinstance(text, str) or not text.strip():
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}
    try:
        return {
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
        }
    except Exception as e:
        logger.warning(f"Readability error: {e}")
        return {"flesch_kincaid_grade": np.nan, "gunning_fog": np.nan}

class TextListDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return str(self.texts[idx])

def batch_get_sentiment_scores(texts, tokenizer, model, labels, batch_size=32, max_length=512):
    if tokenizer is None or model is None:
        neutral_scores = {f"sentiment_{label}": (1.0 if label == "neutral" else 0.0) for label in labels}
        return [neutral_scores.copy() for _ in range(len(texts))]

    dataset = TextListDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    dev = next(model.parameters()).device
    logger.info(f"Calculating sentiment for {len(texts)} texts...")
    for text_batch in tqdm(dataloader, desc="Sentiment Batches"):
        inputs = tokenizer(list(text_batch), return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        batch_scores = predictions.cpu().numpy()
        for scores in batch_scores:
            score_dict = {f"sentiment_{label}": float(score) for label, score in zip(labels, scores)}
            all_scores.append(score_dict)
    return all_scores

def batch_get_semantic_similarity(prompt_response_pairs, model, tokenizer, batch_size=32):
    if model is None or tokenizer is None:
        return [0.0] * len(prompt_response_pairs)
    prompts, responses = zip(*prompt_response_pairs)
    prompt_emb = encode_sentences(list(prompts), model, tokenizer, batch_size)
    resp_emb = encode_sentences(list(responses), model, tokenizer, batch_size)
    return list(np.sum(prompt_emb * resp_emb, axis=1))

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
    advice_kw = (r'(?:\bwhat should i\b|\bhow should i\b|\bshould (?:i|we)\b|'
                 r'\badvice\b|\badvise\b|\brecommend(?:ation)?s?\b|'
                 r'\bpros and cons\b|\bis it (?:okay|ok|ethical|right|wrong|bad|good)\b|'
                 r'\bmorally\b|\bwhat do you think\b)')
    P['p_asks_advice'] = lo.str.contains(advice_kw, regex=True, na=False).astype('int8')
    compare_kw = r'(?:\bcompare\b|\bcomparison\b|\bdifference between\b|\bversus\b|\bwhich is better\b|\bbetter than\b)'
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
    return P.astype({
        'p_codefences': 'int8', 'p_bullets': 'int8', 'p_numlist': 'int8', 'p_list_lines': 'int8',
        'p_is_question': 'int8', 'p_asks_steps': 'int8', 'p_asks_code': 'int8', 'p_asks_math': 'int8',
        'p_asks_advice': 'int8', 'p_compare': 'int8', 'p_summarize': 'int8', 'p_rewrite': 'int8',
        'p_translate': 'int8', 'p_classify': 'int8'
    })

def _mk_len_struct_useful(series, prefix):
    s = series.astype('string').fillna('')
    Fdf = pd.DataFrame(index=s.index)
    Fdf[f'{prefix}chars'] = s.str.len()
    Fdf[f'{prefix}words'] = s.str.split().str.len()
    Fdf[f'{prefix}sents'] = s.str.count(r'[.!?]+').clip(lower=1)
    Fdf[f'{prefix}paragraphs'] = (s.str.count(r'\n\s*\n') + 1).where(Fdf[f'{prefix}chars'] > 0, 0)
    Fdf[f'{prefix}codefences'] = s.str.count(r"```")
    Fdf[f'{prefix}headings'] = s.str.count(r"(?m)^\s*#{1,6}\s+")
    Fdf[f'{prefix}bullets'] = s.str.count(r"(?m)^\s*[-*•]\s+")
    Fdf[f'{prefix}numlist'] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    Fdf[f'{prefix}list_lines'] = Fdf[f'{prefix}bullets'] + Fdf[f'{prefix}numlist']
    Fdf[f'{prefix}qmarks'] = s.str.count(r"\?")
    Fdf[f'{prefix}exclaims'] = s.str.count(r"!")
    ww = Fdf[f'{prefix}words'].replace(0, np.nan)
    for k in ['qmarks', 'exclaims', 'list_lines', 'codefences', 'headings']:
        Fdf[f'{prefix}{k}_per100w'] = (Fdf[f'{prefix}{k}'] / ww * 100).fillna(0)
    return Fdf

def build_lenstruct_useful(df, prompt_col='prompt_clean', resp_a_col='response_a_clean', resp_b_col='response_b_clean'):
    A = _mk_len_struct_useful(df[resp_a_col], 'a_')
    B = _mk_len_struct_useful(df[resp_b_col], 'b_')
    X = pd.concat([A, B], axis=1)
    kept_bases = [c[2:] for c in A.columns if c[2:] not in ('bullets', 'numlist')]
    for k in kept_bases:
        X[f'diff_{k}'] = X[f'a_{k}'] - X[f'b_{k}']
        X[f'ratio_{k}'] = (X[f'a_{k}'] + 1e-6) / (X[f'b_{k}'] + 1e-6)
    p_words = df[prompt_col].astype('string').str.split().str.len().replace(0, np.nan)
    X['a_to_prompt_word_ratio'] = (X['a_words'] / p_words).fillna(0)
    X['b_to_prompt_word_ratio'] = (X['b_words'] / p_words).fillna(0)
    X['a_longer_word'] = (X['a_words'] > X['b_words']).astype('int8')
    X['a_longer_char'] = (X['a_chars'] > X['b_chars']).astype('int8')
    return X.astype('float32', errors='ignore')

start_time = time.time()

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

pairs_a = list(zip(df_data['prompt_last_turn'].fillna("").astype(str), df_data['flat_response_a'].fillna("").astype(str)))
pairs_b = list(zip(df_data['prompt_last_turn'].fillna("").astype(str), df_data['flat_response_b'].fillna("").astype(str)))

sentiment_scores_a = batch_get_sentiment_scores(texts_a, sentiment_tokenizer, sentiment_model, SENTIMENT_LABELS, BATCH_SIZE, MAX_SEQ_LENGTH)
sentiment_scores_b = batch_get_sentiment_scores(texts_b, sentiment_tokenizer, sentiment_model, SENTIMENT_LABELS, BATCH_SIZE, MAX_SEQ_LENGTH)

similarity_scores_a = batch_get_semantic_similarity(pairs_a, sim_model, sim_tokenizer, BATCH_SIZE)
similarity_scores_b = batch_get_semantic_similarity(pairs_b, sim_model, sim_tokenizer, BATCH_SIZE)

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

tb_a = df_data['flat_response_a'].apply(get_textblob_features)
tb_b = df_data['flat_response_b'].apply(get_textblob_features)
rd_a = df_data['flat_response_a'].apply(get_readability_scores)
rd_b = df_data['flat_response_b'].apply(get_readability_scores)

df_features['textblob_polarity_A'] = tb_a.apply(lambda d: d['textblob_polarity'])
df_features['textblob_subjectivity_A'] = tb_a.apply(lambda d: d['textblob_subjectivity'])
df_features['textblob_polarity_B'] = tb_b.apply(lambda d: d['textblob_polarity'])
df_features['textblob_subjectivity_B'] = tb_b.apply(lambda d: d['textblob_subjectivity'])

df_features['flesch_kincaid_grade_A'] = rd_a.apply(lambda d: d['flesch_kincaid_grade'])
df_features['gunning_fog_A'] = rd_a.apply(lambda d: d['gunning_fog'])
df_features['flesch_kincaid_grade_B'] = rd_b.apply(lambda d: d['flesch_kincaid_grade'])
df_features['gunning_fog_B'] = rd_b.apply(lambda d: d['gunning_fog'])

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

length_a = np.log1p(df_features['a_words'].fillna(0)) / 10.0
length_b = np.log1p(df_features['b_words'].fillna(0)) / 10.0

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

missing_cols = [col for col in desired_order if col not in df_final.columns]
if missing_cols:
    logger.warning(f"Missing columns in df_final (filled with NaN): {missing_cols[:10]}{'...' if len(missing_cols)>10 else ''}")
    for col in missing_cols:
        df_final[col] = np.nan

df_final = df_final[desired_order]
df_final.to_csv(FEATURE_OUTPUT, index=False)

end_time = time.time()
logger.info(f"✅ Features saved to {FEATURE_OUTPUT}. Shape: {df_final.shape}. Elapsed: {end_time - start_time:.2f}s")




## === cell 10
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_FEATURES_FILE = FEATURE_OUTPUT
ID_COL = 'id'

XGB_MODEL_FILE = _first_existing([
    "/kaggle/input/xgboost-final-model/xgboost_final_model.json",
    "/kaggle/data/xgboost-final-model/xgboost_final_model.json",
])

logger.info("Loading test features...")
df_test = pd.read_csv(TEST_FEATURES_FILE)
if ID_COL not in df_test.columns:
    raise KeyError(f"{ID_COL} not found in test features at {TEST_FEATURES_FILE}")

df_raw_test = pd.read_csv(DATA_FILE, usecols=[ID_COL])
raw_test_ids = df_raw_test[ID_COL].astype(np.int64).values

df_test_ids = df_test[ID_COL].astype(np.int64).values
if len(df_test_ids) != len(raw_test_ids) or not np.array_equal(df_test_ids, raw_test_ids):
    logger.warning("Feature file id order differs from test.csv; realigning features to test.csv order.")
    df_test = df_test.set_index(ID_COL).reindex(raw_test_ids).reset_index()

test_ids = df_test[ID_COL].astype(np.int64)

feature_cols = [col for col in df_test.columns if col != ID_COL and not col.startswith('winner')]
X_test = df_test[feature_cols].copy()

for c in X_test.columns:
    if X_test[c].dtype == "object":
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
X_test = X_test.astype(np.float32, errors="ignore")

TARGET_COLS = ['winner_model_a', 'winner_model_b', 'winner_tie']

if XGB_MODEL_FILE and os.path.exists(XGB_MODEL_FILE):
    logger.info(f"Loading XGBoost model from: {XGB_MODEL_FILE}")
    dtest = xgb.DMatrix(X_test)
    model = xgb.Booster()
    model.load_model(XGB_MODEL_FILE)

    preds = model.predict(dtest)
    if preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    preds = preds / preds.sum(axis=1, keepdims=True)
else:
    logger.warning("XGBoost model file not found; writing a valid uniform-probability submission.")
    n = len(df_test)
    preds = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)

submission = pd.DataFrame(preds, columns=TARGET_COLS)
submission.insert(0, ID_COL, test_ids.values)
submission = submission[[ID_COL] + TARGET_COLS]

for c in TARGET_COLS:
    submission[c] = submission[c].astype(float)

row_sums = submission[TARGET_COLS].sum(axis=1).values
row_sums = np.where(row_sums == 0, 1.0, row_sums)
submission[TARGET_COLS] = submission[TARGET_COLS].div(row_sums, axis=0)

missing_req = [c for c in [ID_COL] + TARGET_COLS if c not in submission.columns]
if missing_req:
    raise RuntimeError(f"Invalid submission: missing required columns: {missing_req}")

submission.to_csv(PREDICTION_OUTPUT, index=False)

print(submission.head())
print(f"✅ Submission saved to {PREDICTION_OUTPUT} with shape: {submission.shape}")
print("Columns:", list(submission.columns))
print("Sum check (first 5 rows):", submission[TARGET_COLS].sum(axis=1).head().tolist())
print("Submission path exists:", os.path.exists(PREDICTION_OUTPUT))
print("Submission size (bytes):", os.path.getsize(PREDICTION_OUTPUT) if os.path.exists(PREDICTION_OUTPUT) else None)
```

## --- ERROR in cell 10, traceback:
  File "/tmp/ipykernel_55/3561161290.py", line 82
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame is missing required columns: ['winner_model_a', 'winner_model_b', 'winner_tie']
