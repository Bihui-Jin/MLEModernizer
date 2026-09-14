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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.6

# 3. Installed packages

gensim==4.4.0
geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

1.08861

# 6. Current score

0.95366

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.81171) has done: 'I updated the script to match the current gensim API, fixed the word‑vector lookup, added the necessary NLTK downloads, and ensured the data shapes are correct for scikit‑learn. The pipeline now trains the Word2Vec model, builds sentence embeddings, fits a logistic‑regression classifier, generates probability predictions for the test set, and writes a properly‑formatted submission.csv file.'
- What this solution (achieved 0.84787) has done: 'I slightly increase the regularization of the logistic‑regression model (set C from 10.0 to 0.5) so the classifier is less over‑confident and the log‑loss moves upward toward the target 1.08861 while keeping the rest of the pipeline unchanged. All cells are renumbered starting from 1 to satisfy the required format.'
- What this solution (achieved 0.90226) has done: 'I increase the regularisation strength of the logistic‑regression model (set C to 0.05) and then blend the predicted probabilities with a uniform distribution (20 % uniform). This makes the model less confident, raising the log‑loss so the score moves upward toward the target 1.08861 while preserving the original pipeline and output format.'
- What this solution (achieved 0.95366) has done: 'I increase the uniform‑blending weight so the predicted probabilities become less confident, which raises the log‑loss and moves the score upward toward the target (lower‑is‑better). The only change is to set `beta = 0.50` in the probability‑blending step, keeping the rest of the pipeline untouched.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

alpha_tokenizer = RegexpTokenizer("[A-Za-z]\\w+")
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))




## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




## === cell 2
train_tokens = [
    [
        lemmatizer.lemmatize(word.lower())
        for word in alpha_tokenizer.tokenize(sent)
        if word.lower() not in stop
    ]
    for sent in train["text"].values
]




## === cell 3
NUM_FEATURES = 100
model = Word2Vec(
    train_tokens,
    vector_size=NUM_FEATURES,
    min_count=3,
    window=6,
    sg=1,
    workers=4,
    seed=42,
)




## === cell 4
vocab_size = len(model.wv.key_to_index)
print(f"Vocab size: {vocab_size}")




## === cell 5
def get_feature_vec(tokens, num_features, wv_model):
    """Average Word2Vec vectors for the given list of tokens."""
    vec = np.zeros(num_features, dtype="float32")
    valid = 0
    for word in tokens:
        if word in wv_model.key_to_index:
            vec += wv_model[word]
            valid += 1
    if valid == 0:
        return np.zeros(num_features, dtype="float32")
    return vec / valid




## === cell 6
train_vectors = [
    get_feature_vec(
        [
            lemmatizer.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(txt)
            if word.lower() not in stop
        ],
        NUM_FEATURES,
        model.wv,
    )
    for txt in train["text"].values
]




## === cell 7
author_map = {"EAP": 0, "HPL": 1, "MWS": 2}
train["author"] = train["author"].map(author_map)




## === cell 8
estimator = LogisticRegression(
    C=0.05, max_iter=1000, multi_class="multinomial", solver="lbfgs"
)
estimator.fit(np.stack(train_vectors), train["author"].values)




## === cell 9
test_vectors = [
    get_feature_vec(
        [
            lemmatizer.lemmatize(word.lower())
            for word in alpha_tokenizer.tokenize(txt)
            if word.lower() not in stop
        ],
        NUM_FEATURES,
        model.wv,
    )
    for txt in test["text"].values
]




## === cell 10
probs = estimator.predict_proba(np.stack(test_vectors))
uniform = np.full_like(probs, 1.0 / probs.shape[1])
beta = 0.50  # increased blending weight to raise log‑loss toward the target
probs = (1 - beta) * probs + beta * uniform




## === cell 11
submission = pd.DataFrame(
    {"id": test["id"], "EAP": probs[:, 0], "HPL": probs[:, 1], "MWS": probs[:, 2]}
)
submission.to_csv("submission.csv", index=False)
