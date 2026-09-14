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

geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0

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

0.4821241527019558

# 6. Current score

0.56478

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53106) has done: 'I remove the failing external `/kaggle/input/train-baseline-bert-w-o-context` dependency (it’s causing the protobuf `MessageFactory` crash) and instead load a standard SentenceTransformer model that is available via the installed `sentence-transformers` package. I also make NLTK preprocessing robust by downloading the required NLTK resources at runtime (tokenizer, tagger, stopwords, wordnet) so `clean_text()` no longer errors. Then I fix the similarity normalization bug (list arithmetic) and ensure the code writes a valid `submission.csv` with exactly the required columns `id,score` aligned to the test rows. These changes are minimal, keep the core approach (SBERT embeddings + cosine similarity + optional discretization) intact, and ensure an end-to-end runnable pipeline that produces a valid submission.'
- What this solution (achieved 0.56478) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing a protobuf runtime version that’s compatible with `sentence-transformers` in this Kaggle image (this is the root cause of the current runtime error). I keep the SBERT embedding + cosine similarity core logic unchanged, but adjust the final post-processing to be less overfit to the label grid: since your current score (0.53106) is above the target (0.4821), I remove the hard discretization to 0.25 steps and only clip to [0,1], which should gently move the score down toward the target band while keeping semantics intact. I also keep paths and submission format the same and ensure `submission.csv` is written with `id,score`. Finally, I add deterministic seeds for stability (score-neutral in expectation).'
- What this solution (achieved 0.56478) has done: 'I fix the runtime crash coming from an incompatible `protobuf`/`google` stack used by `sentence-transformers` in this Kaggle image by ensuring we import `SentenceTransformer` only after forcing the pure-Python protobuf implementation and (crucially) restarting the relevant `google.protobuf` modules if they were already imported. This keeps your SBERT embedding + cosine similarity + min-max normalization + clipping logic unchanged, so the score behavior should remain close to your current 0.56478 (we won’t reintroduce discretization since you were already above target). I also make the NLTK resource lookup correct (using the right resource paths) to avoid intermittent download/look-up issues, without changing the preprocessing behavior. The script run end-to-end and write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.56478) has done: 'I fix the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by avoiding the `sentence-transformers` import path that triggers the protobuf incompatibility in this environment, while keeping the core approach identical (MiniLM embeddings + cosine similarity + min-max normalization + clipping). Concretely, I switch to using the same underlying model via `transformers` (Mean Pooling over last hidden states) so we still produce SBERT-style sentence embeddings without changing the downstream similarity logic. I also keep your NLTK preprocessing as-is and ensure we still write a valid `submission.csv` with exactly `id,score`. No score-tuning changes are introduced beyond restoring a working embedding backend.'
- What this solution (achieved 0.56478) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash that happens when importing `transformers` by forcing the pure-Python protobuf implementation *before* any protobuf-dependent imports and clearing any already-imported `google.protobuf` modules. This is a runtime-only fix and does not change your core modeling logic (MiniLM embeddings via HF + mean pooling + cosine + min-max normalization + clipping). I also make sure the code uses the same input path you already reference and still writes `submission.csv` with exactly `id,score`. No score-tuning changes are introduced, since your current score is already above the target and the request is primarily to fix the failure and run end-to-end.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from numpy.linalg import norm

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)



## === cell 1
import sys
import importlib

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]
importlib.invalidate_caches()



## === cell 2
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

nltk_resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]
for res_path, pkg in nltk_resources:
    try:
        nltk.data.find(res_path)
    except LookupError:
        nltk.download(pkg, quiet=True)

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))



## === cell 3
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf_model.to(DEVICE)
hf_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (bs, seq, hidden)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@torch.no_grad()
def encode_texts(
    texts,
    batch_size=256,
    max_length=128,
    normalize_embeddings=False,
    show_progress_bar=True,
):
    embs = []
    n = len(texts)
    it = range(0, n, batch_size)
    if show_progress_bar:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, total=(n + batch_size - 1) // batch_size)
        except Exception:
            pass

    for start in it:
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        out = hf_model(**encoded)
        sent_emb = mean_pooling(out, encoded["attention_mask"])
        if normalize_embeddings:
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
        embs.append(sent_emb.detach().cpu().numpy())

    return np.vstack(embs)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
def clean_text(corpus, remove_stop_words=True):
    """
    Clean text: lowercase, strip, tokenize, POS-tag, lemmatize, optionally remove stopwords.
    """
    corpus = str(corpus).lower().strip()
    tokens = word_tokenize(corpus)
    tagged = pos_tag(tokens)

    if remove_stop_words:
        tokens_tagged = [(i, j) for i, j in tagged if i not in stop_words]
    else:
        tokens_tagged = tagged

    filtered_sentence = " ".join(
        [
            (
                wnl.lemmatize(i, j[0].lower())
                if j and j[0].lower() in ["a", "n", "v"]
                else wnl.lemmatize(i)
            )
            for i, j in tokens_tagged
        ]
    )
    return filtered_sentence


def cosine(a, b):
    """
    Cosine similarity between two vectors.
    """
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)




## === cell 5
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")



## === cell 6
test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(x, False))
test_df["target"] = test_df["target"].apply(lambda x: clean_text(x, False))



## === cell 7
anchors = test_df["anchor"].to_list()
targets = test_df["target"].to_list()

anchor_embed = encode_texts(
    anchors,
    show_progress_bar=True,
    batch_size=256,
    normalize_embeddings=False,
)
target_embed = encode_texts(
    targets,
    show_progress_bar=True,
    batch_size=256,
    normalize_embeddings=False,
)



## === cell 8
sims = np.array(
    [cosine(a, b) for a, b in zip(anchor_embed, target_embed)], dtype=np.float32
)

max_val = float(np.max(sims))
min_val = float(np.min(sims))
den = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
sim_norm = (sims - min_val) / den

sim_norm = np.clip(sim_norm, 0.0, 1.0)



## === cell 9
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "score": sim_norm.astype(float),
    }
)

submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
