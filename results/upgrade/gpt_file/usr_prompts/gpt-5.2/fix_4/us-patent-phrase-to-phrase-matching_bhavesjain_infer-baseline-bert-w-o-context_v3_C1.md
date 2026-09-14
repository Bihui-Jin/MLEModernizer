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

0.56511

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56511) has done: 'I remove the external dataset copy/pip-install step that’s causing the `MessageFactory.GetPrototype` error and instead load a standard `sentence-transformers` model that is available via the installed package. I also add the required NLTK downloads (tokenizer, tagger, stopwords, wordnet) so `clean_text()` can run without `LookupError`. Then I ensure embeddings are computed, cosine similarities are generated safely, and the submission dataframe includes both required columns (`id`, `score`) with the correct row alignment before writing `submission.csv`. These fixes are necessary for end-to-end execution and to yield a valid submission; the overall modeling approach (SentenceTransformer embeddings + cosine similarity) is preserved.'
- What this solution (achieved 0.56511) has done: 'We fix the `MessageFactory.GetPrototype` crash by forcing a compatible `protobuf` implementation at runtime before importing `sentence_transformers` (this is a common Kaggle environment mismatch). The rest of the pipeline (NLTK cleaning → SentenceTransformer embeddings → cosine similarity → clip to [0,1]) stays the same to preserve evaluation semantics and keep score changes minimal. I also make the data path resolution more robust (fallback to `/kaggle/data/...` if needed) and ensure the submission is always written as `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.56511) has done: 'I fix the crash in `SentenceTransformer` import by forcing a protobuf implementation that is compatible with the Kaggle runtime, before `sentence_transformers` is imported. This is a pure runtime compatibility fix and keeps the same model (`all-MiniLM-L6-v2`) and the same cosine-similarity scoring logic, so it should be score-neutral (and not intentionally push performance further away from your target). I also add a safe fallback that tries a second protobuf setting only if the first attempt fails, so the notebook reliably runs end-to-end and always writes a valid `submission.csv`. Paths, preprocessing, embedding, and submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



## === cell 1
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # some NLTK versions require this
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)




## === cell 2
def _load_st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except AttributeError as e:
        if "GetPrototype" in str(e):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name)
        raise


model_name = "all-MiniLM-L6-v2"
model = _load_st_model(model_name)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))




## === cell 4
def clean_text(corpus, remove_stop_words=True):
    """
    Clean text: lowercase, strip, tokenize, POS-tag, lemmatize; optionally remove stopwords.
    """
    corpus = str(corpus).lower().strip()
    tokens = word_tokenize(corpus)

    if remove_stop_words:
        tokens = [t for t in tokens if t not in stop_words]

    tagged = pos_tag(tokens)
    lemmas = [
        (
            wnl.lemmatize(tok, tag[0].lower())
            if tag and tag[0].lower() in ["a", "n", "v"]
            else wnl.lemmatize(tok)
        )
        for tok, tag in tagged
    ]
    return " ".join(lemmas)


def cosine(a, b):
    """
    Cosine similarity of two vectors.
    """
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)




## === cell 5
test_path_primary = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
test_path_fallback = "/kaggle/data/us-patent-phrase-to-phrase-matching/test.csv"

test_path = (
    test_path_primary if os.path.exists(test_path_primary) else test_path_fallback
)
if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Could not find test.csv at {test_path_primary} or {test_path_fallback}"
    )

test_df = pd.read_csv(test_path)
test_df.head()



## === cell 6
test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(x, False))
test_df["target"] = test_df["target"].apply(lambda x: clean_text(x, False))



## === cell 7
anchors = test_df["anchor"].tolist()
targets = test_df["target"].tolist()

anchor_embed = model.encode(anchors, show_progress_bar=True, batch_size=256)
target_embed = model.encode(targets, show_progress_bar=True, batch_size=256)



## === cell 8
sims = [cosine(a, b) for a, b in zip(anchor_embed, target_embed)]
sims = np.clip(np.array(sims, dtype=np.float32), 0.0, 1.0)



## === cell 9
submission_df = pd.DataFrame({"id": test_df["id"].values, "score": sims})
submission_df.head()



## === cell 10
submission_df.to_csv("submission.csv", index=False)

assert list(submission_df.columns) == ["id", "score"]
assert submission_df.shape[0] == test_df.shape[0]
print("Wrote submission.csv with shape:", submission_df.shape)
