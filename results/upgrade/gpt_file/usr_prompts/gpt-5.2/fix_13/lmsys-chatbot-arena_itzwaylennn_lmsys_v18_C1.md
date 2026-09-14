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
joblib==1.5.2
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

1.0825928949255894

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the stray non-Python prose and triple-backticks that are currently being executed as code (causing the `SyntaxError` in the first and last cells), without changing the feature extraction or MLP inference logic. I will also ensure the pipeline always writes a valid `submission.csv` with the exact required columns by building the submission from `sample_submission.csv` and aligning predictions by `id`. To prevent the “missing required columns” submission error, I’ll add a strict final validation step that checks columns/order, row count, and that probabilities are finite and row-normalized before saving. These changes are execution/format fixes and should be score-neutral (the model outputs remain the same).

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/2308069542.py", line 1
    I will remove the stray non-Python prose and triple-backticks that are currently being executed as code (causing the `SyntaxError` in the first and last cells), without changing the feature extraction or MLP inference logic. I will also ensure the pipeline always writes a valid `submission.csv` with the exact required columns by building the submission from `sample_submission.csv` and aligning predictions by `id`. To prevent the “missing required columns” submission error, I’ll add a strict final validation step that checks columns/order, row count, and that probabilities are finite and row-normalized before saving. These changes are execution/format fixes and should be score-neutral (the model outputs remain the same).
                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
SyntaxError: invalid character '“' (U+201C)


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
import importlib.util

_HAS_TEXTSTAT = importlib.util.find_spec("textstat") is not None
if _HAS_TEXTSTAT:
    import textstat  # noqa: F401
    print("textstat is available.")
else:
    print("textstat not available; will use a fallback readability implementation.")



## === cell 3
import importlib.util

_HAS_BNB = importlib.util.find_spec("bitsandbytes") is not None
if _HAS_BNB:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes is available.")
else:
    print("bitsandbytes not available (OK). Proceeding without it.")



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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "/kaggle/input/lmsys-chatbot-arena/test.csv"
CHECKPOINT_DIR = "checkpoints_llm_judge"
CHUNK_SIZE = 50
BATCH_SIZE_JUDGE = 1
MAX_NEW_TOKENS = 150
MAX_SEQ_LENGTH = 512
PREDICTION_OUTPUT = "submission.csv"
FEATURE_OUTPUT = "lmsys_test_features_final.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logger.info("Starting feature extraction + distilled MLP inference pipeline.")



## === cell 5
import os

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
        if level >= 2:
            break
else:
    print(f"Not found: {SIM_MODEL_PATH}")



## === cell 6
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
SENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



## === cell 7
import os
import torch
from transformers import AutoTokenizer, AutoModel

SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"

sim_tokenizer = None
sim_model = None
if os.path.exists(SIM_MODEL_PATH):
    try:
        print(f"Loading similarity model from: {SIM_MODEL_PATH}")
        sim_tokenizer = AutoTokenizer.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
        sim_model = AutoModel.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
        sim_model.to("cuda" if torch.cuda.is_available() else "cpu")
        sim_model.eval()
        print("Similarity model loaded successfully.")
    except Exception as e:
        print(f"Failed to load similarity model; using fallback similarity. Error: {e}")
        sim_tokenizer = None
        sim_model = None
else:
    print(f"Similarity model path not found; using fallback similarity: {SIM_MODEL_PATH}")



## === cell 8
import pandas as pd
test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
print("Test shape:", test_df.shape)
print(test_df.head(2))



## === cell 9
import pandas as pd
import numpy as np
from textblob import TextBlob
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import ast
import logging
import time
import re
import json
import warnings
import os
import importlib.util

warnings.filterwarnings("ignore", category=SyntaxWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HAS_LABELS = False  # Test data has no labels

SENTIMENT_MODEL_PATH = "/kaggle/input/twitter-roberta-sentiment-offline"
SIMILARITY_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
SENTIMENT_LABELS = ['negative', 'neutral', 'positive']

BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_HAS_TEXTSTAT = importlib.util.find_spec("textstat") is not None
if _HAS_TEXTSTAT:
    import textstat as _textstat  # type: ignore
else:
    _textstat = None

def _simple_syllable_count(word: str) -> int:
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_v = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def _fallback_fk_grade(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"\b\w+\b", text)
    n_words = max(1, len(words))
    syll = sum(_simple_syllable_count(w) for w in words)
    return float(0.39 * (n_words / sents) + 11.8 * (syll / n_words) - 15.59)

def _fallback_gunning_fog(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"\b\w+\b", text)
    n_words = max(1, len(words))
    complex_words = sum(1 for w in words if _simple_syllable_count(w) >= 3)
    return float(0.4 * ((n_words / sents) + 100.0 * (complex_words / n_words)))

logger.info(f"Loading data from '{DATA_FILE}'...")
df_data = pd.read_csv(DATA_FILE, engine='python', on_bad_lines='skip')
logger.info(f"Loaded '{DATA_FILE}'. Shape: {df_data.shape}")

sentiment_tokenizer = None
sentiment_model = None
if os.path.exists(SENTIMENT_MODEL_PATH):
    try:
        logger.info(f"Loading Sentiment Analysis Model from: {SENTIMENT_MODEL_PATH}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, local_files_only=True)
        sentiment_model.eval()
        if torch.cuda.is_available():
            sentiment_model = sentiment_model.to('cuda')
            logger.info("Sentiment model moved to GPU.")
    except Exception as e:
        logger.warning(f"Failed to load sentiment model; using fallback neutral. Error: {e}")
        sentiment_tokenizer = None
        sentiment_model = None
else:
    logger.warning(f"Sentiment model path missing; using fallback neutral: {SENTIMENT_MODEL_PATH}")

sim_tokenizer = None
sim_model = None
if os.path.exists(SIMILARITY_MODEL_PATH):
    try:
        logger.info(f"Loading Semantic Similarity Model from: {SIMILARITY_MODEL_PATH}")
        sim_tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
        sim_model = AutoModel.from_pretrained(SIMILARITY_MODEL_PATH, local_files_only=True)
        sim_model.eval()
        if torch.cuda.is_available():
            sim_model = sim_model.to('cuda')
            logger.info("Similarity model moved to GPU.")
    except Exception as e:
        logger.warning(f"Failed to load similarity model; using fallback similarity. Error: {e}")
        sim_tokenizer = None
        sim_model = None
else:
    logger.warning(f"Similarity model path missing; using fallback similarity: {SIMILARITY_MODEL_PATH}")

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentences(sentences, model, tokenizer, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    cleaned = []
    for sent in sentences:
        if sent is None or pd.isna(sent):
            cleaned.append("")
        elif isinstance(sent, (int, float)):
            cleaned.append(str(int(sent)) if not pd.isna(sent) else "")
        else:
            try:
                cleaned.append(str(sent).strip())
            except Exception:
                cleaned.append("")

    embeddings = []
    dataset = [cleaned[i:i+batch_size] for i in range(0, len(cleaned), batch_size)]
    model.to(device)

    with torch.no_grad():
        for batch_texts in tqdm(dataset, desc="Encoding Batches"):
            batch_texts = [txt if isinstance(txt, str) and len(txt.strip()) > 0 else "" for txt in batch_texts]
            try:
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
            except Exception as e:
                logger.error(f"Tokenization/encoding failed on batch: {e}")
                dummy = np.zeros((len(batch_texts), 384), dtype=np.float32)
                embeddings.append(dummy)

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
        if _textstat is not None:
            return {
                "flesch_kincaid_grade": float(_textstat.flesch_kincaid_grade(text)),
                "gunning_fog": float(_textstat.gunning_fog(text)),
            }
        return {
            "flesch_kincaid_grade": _fallback_fk_grade(text),
            "gunning_fog": _fallback_gunning_fog(text),
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
        neutral = {f"sentiment_{label}": (1.0 if label == 'neutral' else 0.0) for label in labels}
        return [neutral.copy() for _ in range(len(texts))]

    dataset = TextListDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_scores = []
    device = next(model.parameters()).device
    logger.info(f"Calculating sentiment for {len(texts)} texts...")
    for text_batch in tqdm(dataloader, desc="Sentiment Batches"):
        try:
            inputs = tokenizer(list(text_batch), return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = predictions.cpu().numpy()
            for scores in batch_scores:
                score_dict = {f"sentiment_{label}": float(score) for label, score in zip(labels, scores)}
                all_scores.append(score_dict)
        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            for _ in text_batch:
                neutral_scores = {f"sentiment_{label}": 0.0 if label != 'neutral' else 1.0 for label in labels}
                all_scores.append(neutral_scores)
    return all_scores

def _fallback_similarity(prompt: str, response: str) -> float:
    p = set(re.findall(r"\w+", (prompt or "").lower()))
    r = set(re.findall(r"\w+", (response or "").lower()))
    if not p and not r:
        return 0.0
    inter = len(p & r)
    union = len(p | r)
    return float(inter / max(1, union))

def batch_get_semantic_similarity(prompt_response_pairs, model, tokenizer, batch_size=32):
    if model is None or tokenizer is None:
        return [_fallback_similarity(p, r) for p, r in prompt_response_pairs]
    prompts, responses = zip(*prompt_response_pairs)
    prompt_emb = encode_sentences(list(prompts), model, tokenizer, batch_size)
    resp_emb = encode_sentences(list(responses), model, tokenizer, batch_size)
    cos_sims = np.sum(prompt_emb * resp_emb, axis=1)
    return list(cos_sims)

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
    return P.astype({'p_codefences': 'int8', 'p_bullets': 'int8', 'p_numlist': 'int8', 'p_list_lines': 'int8',
                     'p_is_question': 'int8', 'p_asks_steps': 'int8', 'p_asks_code': 'int8', 'p_asks_math': 'int8',
                     'p_asks_advice': 'int8', 'p_compare': 'int8', 'p_summarize': 'int8', 'p_rewrite': 'int8',
                     'p_translate': 'int8', 'p_classify': 'int8'})

def _mk_len_struct_useful(series, prefix):
    s = series.astype('string').fillna('')
    Ff = pd.DataFrame(index=s.index)
    Ff[f'{prefix}chars'] = s.str.len()
    Ff[f'{prefix}words'] = s.str.split().str.len()
    Ff[f'{prefix}sents'] = s.str.count(r'[.!?]+').clip(lower=1)
    Ff[f'{prefix}paragraphs'] = (s.str.count(r'\n\s*\n') + 1).where(Ff[f'{prefix}chars'] > 0, 0)
    Ff[f'{prefix}codefences'] = s.str.count(r"```")
    Ff[f'{prefix}headings'] = s.str.count(r"(?m)^\s*#{1,6}\s+")
    Ff[f'{prefix}bullets'] = s.str.count(r"(?m)^\s*[-*•]\s+")
    Ff[f'{prefix}numlist'] = s.str.count(r"(?m)^\s*[0-9]{1,2}[.)]\s+")
    Ff[f'{prefix}list_lines'] = Ff[f'{prefix}bullets'] + Ff[f'{prefix}numlist']
    Ff[f'{prefix}qmarks'] = s.str.count(r"\?")
    Ff[f'{prefix}exclaims'] = s.str.count(r"!")
    ww = Ff[f'{prefix}words'].replace(0, np.nan)
    for k in ['qmarks', 'exclaims', 'list_lines', 'codefences', 'headings']:
        Ff[f'{prefix}{k}_per100w'] = (Ff[f'{prefix}{k}'] / ww * 100).fillna(0)
    return Ff

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
df_data['prompt_last_turn'] = df_data['prompt_last_turn'].fillna("").astype(str)
df_data['flat_response_a'] = df_data['flat_response_a'].fillna("").astype(str)
df_data['flat_response_b'] = df_data['flat_response_b'].fillna("").astype(str)

pairs_a = list(zip(df_data['prompt_last_turn'], df_data['flat_response_a']))
pairs_b = list(zip(df_data['prompt_last_turn'], df_data['flat_response_b']))

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

df_final = df_features.merge(df_extended_with_id, on='id', how='left')
logger.info(f"Final feature shape: {df_final.shape}")

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
    'b_codefences', 'b_bullets', 'b_numlist', 'b_list_lines',
    'b_qmarks', 'b_exclaims',
    'b_qmarks_per100w', 'b_exclaims_per100w', 'b_list_lines_per100w',
    'b_codefences_per100w', 'b_headings_per100w',
    'diff_chars', 'ratio_chars',
    'diff_words', 'ratio_words',
    'diff_sents', 'ratio_sents',
    'diff_paragraphs', 'ratio_paragraphs',
    'diff_codefences', 'ratio_codefences',
    'diff_headings', 'ratio_headings',
    'diff_list_lines', 'ratio_list_lines',
    'diff_qmarks', 'ratio_qmarks',
    'diff_exclaims', 'ratio_exclaims',
    'diff_qmarks_per100w', 'ratio_qmarks_per100w',
    'diff_exclaims_per100w', 'ratio_exclaims_per100w',
    'diff_list_lines_per100w', 'ratio_list_lines_per100w',
    'diff_codefences_per100w', 'ratio_codefences_per100w',
    'diff_headings_per100w', 'ratio_headings_per100w',
    'a_to_prompt_word_ratio', 'b_to_prompt_word_ratio',
    'a_longer_word', 'a_longer_char'
]
missing_cols = [c for c in desired_order if c not in df_final.columns]
if missing_cols:
    logger.warning(f"Missing columns in df_final (will fill NaN): {missing_cols}")
    for c in missing_cols:
        df_final[c] = np.nan

df_final = df_final[desired_order]
df_final.to_csv(FEATURE_OUTPUT, index=False)
logger.info(f"Features saved to {FEATURE_OUTPUT}. Shape: {df_final.shape}")

end_time = time.time()
logger.info(f"Elapsed Time: {end_time - start_time:.2f} seconds")



## === cell 10
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 11
df_features = pd.read_csv(FEATURE_OUTPUT)
X_test = df_features.drop(columns=['id', 'winner_model_a', 'winner_model_b', 'winner_tie'], errors='ignore').values
ids = df_features['id'].values
print(f"Features shape: {X_test.shape}, ids: {ids.shape}")



## === cell 12
class StudentMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = torch.clamp(x, -1e3, 1e3)
        return self.net(x)



## === cell 13
pass



## === cell 14
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CSV = FEATURE_OUTPUT
df_test = pd.read_csv(TEST_CSV)

if "id" not in df_test.columns:
    raise ValueError(f"'id' column missing from features file: {TEST_CSV}")

ids = df_test['id'].values

scaler_path = '/kaggle/input/distillation/scaler.pkl'
model_path = '/kaggle/input/distillation/distilled_mlp.pth'
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Missing scaler at {scaler_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing model weights at {model_path}")

scaler = joblib.load(scaler_path)

if hasattr(scaler, "feature_names_in_"):
    expected_all = list(scaler.feature_names_in_)
    feature_cols = [c for c in expected_all if c != "id"]
else:
    feature_cols = [c for c in df_test.columns if c != "id"]

missing = [c for c in feature_cols if c not in df_test.columns]
if missing:
    for c in missing:
        df_test[c] = np.nan

X_test = df_test[feature_cols].values.astype(np.float32)

print(f"Loaded test features for scaling: {X_test.shape}")
print("Before imputation - NaN:", int(np.isnan(X_test).sum()), "| Inf:", int(np.isinf(X_test).sum()))

col_medians = np.nanmedian(X_test, axis=0)
nan_mask = np.isnan(X_test)
if nan_mask.sum() > 0:
    X_test[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

X_test = np.where(np.isposinf(X_test),  1e6, X_test)
X_test = np.where(np.isneginf(X_test), -1e6, X_test)

print("After imputation - NaN:", int(np.isnan(X_test).sum()), "| Inf:", int(np.isinf(X_test).sum()))

X_scaled = scaler.transform(X_test)

if np.isnan(X_scaled).any():
    raise ValueError("Scaling introduced NaN")
if np.isinf(X_scaled).any():
    raise ValueError("Scaling introduced Inf")

model = StudentMLP(input_dim=X_scaled.shape[1]).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

preds = []
batch_size = 512
for i in range(0, len(X_scaled), batch_size):
    x_batch = torch.from_numpy(X_scaled[i:i+batch_size]).float().to(device)
    with torch.no_grad():
        logits = model(x_batch)
        batch_probs = F.softmax(logits, dim=1).cpu().numpy()
        preds.append(batch_probs)

preds = np.concatenate(preds, axis=0)
preds = np.nan_to_num(preds, nan=1/3, posinf=1/3, neginf=1/3)
preds = np.clip(preds, 1e-7, 1 - 1e-7)
preds = preds / preds.sum(axis=1, keepdims=True)

if preds.shape != (len(ids), 3):
    raise ValueError(f"Bad preds shape: {preds.shape}, expected ({len(ids)}, 3)")

submission_pred = pd.DataFrame({
    'id': ids,
    'winner_model_a': preds[:, 0],
    'winner_model_b': preds[:, 1],
    'winner_tie': preds[:, 2]
})

sample_path = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
if not os.path.exists(sample_path):
    raise FileNotFoundError("sample_submission.csv not found in expected Kaggle paths.")

sample = pd.read_csv(sample_path)
required_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie']
missing_in_sample = [c for c in required_cols if c not in sample.columns]
if missing_in_sample:
    raise ValueError(f"sample_submission.csv missing required columns: {missing_in_sample}")

submission = sample[['id']].copy()
pred_map = submission_pred.set_index('id')[['winner_model_a', 'winner_model_b', 'winner_tie']]
aligned = pred_map.reindex(submission['id'].values)

for c in ['winner_model_a', 'winner_model_b', 'winner_tie']:
    submission[c] = aligned[c].to_numpy()

submission[['winner_model_a', 'winner_model_b', 'winner_tie']] = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].fillna(1.0/3.0)

P = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].to_numpy(dtype=np.float64)
P = np.nan_to_num(P, nan=1/3, posinf=1/3, neginf=1/3)
P = np.clip(P, 1e-7, 1 - 1e-7)
P = P / P.sum(axis=1, keepdims=True)
submission[['winner_model_a', 'winner_model_b', 'winner_tie']] = P

submission = submission[required_cols]

if list(submission.columns) != required_cols:
    raise ValueError(f"Submission columns wrong. Got {list(submission.columns)} expected {required_cols}")
if submission.shape[0] != sample.shape[0]:
    raise ValueError(f"Submission row count mismatch: {submission.shape[0]} vs sample {sample.shape[0]}")
if submission[required_cols[1:]].isna().any().any():
    raise ValueError("Submission contains NaN probabilities")
row_sums = submission[required_cols[1:]].sum(axis=1).to_numpy()
if not np.all(np.isfinite(row_sums)):
    raise ValueError("Submission contains non-finite probabilities")
if not np.allclose(row_sums, 1.0, atol=1e-6):
    raise ValueError("Submission probabilities are not row-normalized to 1")

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
print(submission.head())
print(f"Shape: {submission.shape}")
print("Columns:", list(submission.columns))
```

## --- ERROR in cell 14, traceback:
  File "/tmp/ipykernel_55/1227929310.py", line 138
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame is missing required columns: ['winner_model_a', 'winner_model_b', 'winner_tie']
