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

3.13

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

0.8275192406827523

# 6. Current score

0.66607

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45106) has done: 'The fix changes the Ridge regression to use the `lsqr` solver, which avoids the SciPy `cg` incompatibility that caused the training to fail. After fitting, predictions are clipped to [0, 1] and then rounded to the allowed 0.25 step values, matching the competition’s label granularity and likely improving Pearson correlation. The rest of the pipeline (data loading, text concatenation, and submission writing) remains unchanged, guaranteeing a valid `submission.csv` file.'
- What this solution (achieved 0.6056) has done: 'I keep the overall pipeline (concatenating context + anchor + target, TF‑IDF feature extraction, linear Ridge regression) but add a character‑level TF‑IDF via a `FeatureUnion` and lower the regularisation strength (α=0.5). This richer text representation usually boosts Pearson correlation. I also remove the forced rounding to 0.25 steps, leaving the predictions continuous (only clipping to [0, 1]), which better matches the Pearson metric.'
- What this solution (achieved 0.62267) has done: 'I increase the expressiveness of the text features and reduce the regularisation strength, which are small, targeted tweaks that usually raise the Pearson correlation for this type of linear model. Specifically, I double the `max_features` for both the word‑level and character‑level TF‑IDF vectorisers and set the Ridge regularisation `alpha` to 0.1. These changes keep the overall pipeline unchanged while giving the model more signal to learn from, moving the score closer to the target.'
- What this solution (achieved 0.61709) has done: 'I enhance the text representation by expanding the n‑gram ranges, increasing the number of features, enabling sub‑linear term frequency scaling, and removing stop‑word removal – all of which give the linear Ridge model more useful signal without altering its core structure. I also lower the Ridge regularisation slightly (α = 0.05) to let the richer features influence the predictions more. These targeted parameter tweaks should raise the Pearson correlation toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.08769) has done: 'I swap the Ridge regressor for an unregularized LinearRegression, which removes the extra bias from the α = 0.05 regularisation and usually yields a higher Pearson correlation for this TF‑IDF feature set. The rest of the pipeline (concatenated text, word‑ and char‑level TF‑IDF, clipping to [0, 1]) stays unchanged, ensuring a valid submission.csv is still produced.'
- What this solution (achieved 0.60142) has done: 'I fix the training crash by switching the Ridge regression solver to “lsqr”, which works with the current SciPy version, and I slightly enlarge the TF‑IDF vocabularies (more features) to give the model a bit more signal. These minimal changes unblock the pipeline, allow the model to fit, and should move the Pearson score closer to the target while preserving the original workflow and submission format.'
- What this solution (achieved 0.60619) has done: 'I slightly adjust the linear pipeline to be more appropriate for Pearson scoring: use English stop‑word removal in the word‑level TF‑IDF (helps focus on informative terms) and raise the Ridge regularisation strength back to 0.1 (the previous 0.01 was too low and hurt correlation). These are minimal, targeted tweaks that keep the overall model unchanged while likely moving the score upward toward the target.'
- What this solution (achieved 0.61114) has done: 'I lower the Ridge regularisation (α = 0.05) and remove stop‑word filtering from the word‑level TF‑IDF, letting the model use all tokens. I also increase the vocabulary size for both word‑ and character‑level vectorisers (800 k features each) to capture more signal. These minimal tweaks keep the overall pipeline unchanged while providing the model with richer features, which should raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.57706) has done: 'I make two minimal tweaks that are known to improve Pearson for this task: (1) increase the Ridge regularisation slightly (α = 0.1) to reduce over‑fitting given the very large TF‑IDF vocabularies, and (2) round the final predictions to the allowed 0.25 increments after clipping to [0, 1]. Both changes keep the original pipeline intact while aligning the output more closely with the discrete label granularity, which should move the score nearer the target.'
- What this solution (achieved 0.60386) has done: 'I lower the Ridge regularisation to α=0.05 (to let the richer TF‑IDF features influence the model more) and add English stop‑word removal for the word‑level vectoriser (reducing noisy terms). I also stop rounding the predictions to 0.25 steps – keeping the continuous output better matches the Pearson correlation metric. These minimal tweaks keep the overall pipeline unchanged while expected to raise the score toward the target.'
- What this solution (achieved 0.6332) has done: 'I add a lightweight sentence‑transformer embedding feature (converted to a sparse matrix) and combine it with the existing word‑ and character‑level TF‑IDF features via a `FeatureUnion`. This enriches the representation while keeping the linear Ridge model unchanged, and I lower the Ridge regularisation to α=0.01 to let the extra signal improve the Pearson correlation. The rest of the pipeline (data loading, prediction clipping and CSV writing) stays the same.'
- What this solution (achieved 0.53224) has done: 'The fix patches the protobuf API so that `SentenceTransformer` loads without the “MessageFactory” error, then lowers the Ridge regularisation (α = 0.001) to let the richer TF‑IDF + embedding features improve Pearson correlation. No core logic is changed, and the pipeline still writes a valid `submission.csv`.'
- What this solution (achieved 0.66607) has done: 'The fix adds a robust fallback for the SentenceTransformer so the pipeline never crashes due to the protobuf incompatibility, switches to a stronger embedding model, and restores the Ridge regularisation (α = 0.01) that was shown to give a higher Pearson score in earlier experiments. These minimal changes keep the overall architecture unchanged while improving the feature representation and stabilising training, moving the validation score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith(".csv"):
            print(os.path.join(dirname, filename))



## === cell 1
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass
except Exception:
    pass


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for a SentenceTransformer model.
    If the model cannot be loaded (e.g., protobuf issues), it falls back to a zero‑vector,
    ensuring the pipeline stays functional.
    """

    def __init__(
        self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embed_dim = None  # will be set after a successful load

    def fit(self, X, y=None):
        try:
            from sentence_transformers import SentenceTransformer

            self.model_ = SentenceTransformer(self.model_name)
            test_vec = self.model_.encode(["test"], batch_size=1, convert_to_numpy=True)
            self.embed_dim = test_vec.shape[1]
        except Exception as e:
            self.model_ = None
            self.embed_dim = 768
        return self

    def transform(self, X):
        if self.model_ is None:
            return sparse.csr_matrix(
                np.zeros((len(X), self.embed_dim), dtype=np.float32)
            )
        embeddings = self.model_.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return sparse.csr_matrix(embeddings)


word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=800000,
    sublinear_tf=True,
    stop_words="english",
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 7),
    max_features=800000,
    sublinear_tf=True,
)

combined_vectorizer = FeatureUnion(
    [
        ("word", word_vectorizer),
        ("char", char_vectorizer),
        ("embed", EmbeddingTransformer()),
    ]
)

model = make_pipeline(
    combined_vectorizer,
    Ridge(alpha=0.01, solver="lsqr", random_state=42),
)

train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
train_df = pd.read_csv(train_path)

train_text = (
    train_df["context"].astype(str)
    + " "
    + train_df["anchor"].astype(str)
    + " "
    + train_df["target"].astype(str)
)

y_train = train_df["score"].astype(float)

model.fit(train_text, y_train)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
test_df = pd.read_csv(test_path)

test_text = (
    test_df["context"].astype(str)
    + " "
    + test_df["anchor"].astype(str)
    + " "
    + test_df["target"].astype(str)
)

pred = model.predict(test_text)
pred = np.clip(pred, 0.0, 1.0)



## === cell 3
submission = pd.DataFrame({"id": test_df["id"], "score": pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
