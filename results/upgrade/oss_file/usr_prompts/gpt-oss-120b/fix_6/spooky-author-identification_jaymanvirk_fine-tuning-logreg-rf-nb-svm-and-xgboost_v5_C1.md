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

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
xgboost==2.0.3

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

0.4013

# 6. Current score

0.49515

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50526) has done: 'I add a fallback download for the spaCy model to ensure it loads, switch the classifier to ComplementNB (which often yields lower log‑loss for text data), and reorder the submission columns so the id appears first, matching the expected format. These minimal changes keep the overall pipeline intact while aiming to move the log‑loss closer to the target score.'
- What this solution (achieved 0.47791) has done: 'I lower the ComplementNB smoothing parameter from 0.1 to 0.01, which usually yields sharper probability estimates for text classification and therefore reduces the log‑loss, moving the score closer to the target 0.4013. No other parts of the pipeline are changed, so the core logic and submission format remain intact.'
- What this solution (achieved 9.07431) has done: 'I add a small validation split to choose a better ComplementNB smoothing parameter (alpha) and switch the model to use the scaled TF‑IDF features, which are better conditioned for Naïve Bayes. This minor change keeps the overall pipeline and model type unchanged while aiming to lower the log‑loss toward the target. The script now imports `log_loss`, selects the best α from a tiny grid, retrains on the full data, and predicts with the scaled features before writing the submission CSV.'
- What this solution (achieved 0.49515) has done: 'Implemented a minimal fix by training and predicting with the raw TF‑IDF matrix instead of the unnecessarily scaled version. Scaling distorted the probabilistic assumptions of ComplementNB, leading to a very high log‑loss. Using the original TF‑IDF features aligns with the Naïve Bayes model expectations and is expected to substantially lower the loss, moving it toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)




## === cell 1
list_l = []
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        list_l.append(os.path.join(dirname, filename))




## === cell 2
train_path = next(
    p for p in list_l if p.endswith("train.csv") and "spooky-author-identification" in p
)
test_path = next(
    p for p in list_l if p.endswith("test.csv") and "spooky-author-identification" in p
)
sample_path = next(
    p
    for p in list_l
    if p.endswith("sample_submission.csv") and "spooky-author-identification" in p
)

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_data = pd.read_csv(sample_path)




## === cell 3
def print_short_summary(name, data):
    print(name)
    print("\n1. Data head:")
    print(data.head())
    print("\n2 Data shape: {}".format(data.shape))
    print("\n3. Data info:")
    data.info()
    if "text" in data.columns:
        avg = np.mean(np.vectorize(len)(data["text"]))
        print("\n4. Average number of characters per text: {:.0f}".format(avg))




## === cell 4
print_short_summary("Train data", train_data)
print_short_summary("Test data", test_data)
print_short_summary("Sample data", sample_data)




## === cell 5
plt.figure(figsize=(16, 9))
tmp = train_data["author"].value_counts()
sns.barplot(y=tmp.index.values, x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Author")
plt.title("Number of records per author")
plt.show()




## === cell 6
pass




## === cell 7
def plot_word_dist_author(labels, top_n_words=10):
    n = len(labels)
    default_palette = sns.color_palette("deep")
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(16, 9))
    for i in range(n):
        col = i % n
        indexes = train_data["author"] == labels[i]
        w = train_data["text"][indexes].str.split(expand=True).unstack().value_counts()
        l = w[:top_n_words] / np.sum(w) * 100
        axes[col].bar(l.index, l.values, color=default_palette[i])
        axes[col].set_title(labels[i])
        axes[col].set_xlabel("Words")
        axes[col].set_ylabel("Percentage of total word count (%)")
    plt.tight_layout()
    plt.show()




## === cell 8
plot_word_dist_author(["EAP", "MWS", "HPL"])




## === cell 9
pass




## === cell 10
try:
    spacy_process = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    spacy.cli.download("en_core_web_sm")
    spacy_process = spacy.load("en_core_web_sm", disable=["parser", "ner"])
pattern = re.compile(r"\b([a-zA-Z])\b|\d+|[.,!?()-:;]")
stop_words = set(stopwords.words("english"))




## === cell 11
def get_processed_text(text):
    text = pattern.sub("", text.lower())
    lemmas = spacy_process(text)
    lemmas = [token.lemma_ for token in lemmas if token.text not in stop_words]
    return " ".join(lemmas)


def get_clean_text(texts):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor() as executor:
        clean_texts = list(executor.map(get_processed_text, texts))
    return np.array(clean_texts)




## === cell 12
train_clean_data = get_clean_text(train_data["text"].values)
test_clean_data = get_clean_text(test_data["text"].values)




## === cell 13
pass




## === cell 14
vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
tfidf_vect = vectorizer.fit(train_clean_data)

X_train_tfidf = tfidf_vect.transform(train_clean_data)
X_test_tfidf = tfidf_vect.transform(test_clean_data)




## === cell 15
svd = TruncatedSVD(n_components=100, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)




## === cell 16
scaler = StandardScaler(with_mean=False)
X_train_stand = scaler.fit_transform(X_train_tfidf)
X_test_stand = scaler.transform(X_test_tfidf)

X_train_stand_svd = scaler.fit_transform(X_train_svd)
X_test_stand_svd = scaler.transform(X_test_svd)




## === cell 17
dict_map = {"EAP": 0, "MWS": 1, "HPL": 2}
y_train = train_data["author"].map(dict_map).values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)
candidate_alphas = [0.001, 0.005, 0.01]
best_alpha = candidate_alphas[0]
best_loss = np.inf
for a in candidate_alphas:
    tmp_model = ComplementNB(alpha=a).fit(X_tr, y_tr)
    val_pred = tmp_model.predict_proba(X_val)
    loss = log_loss(y_val, val_pred)
    if loss < best_loss:
        best_loss = loss
        best_alpha = a

model = ComplementNB(alpha=best_alpha).fit(X_train_tfidf, y_train)




## === cell 18
results = model.predict_proba(X_test_tfidf)




## === cell 19
submission = pd.DataFrame(
    {
        "id": sample_data["id"],
        "EAP": results[:, 0],
        "HPL": results[:, 2],
        "MWS": results[:, 1],
    }
)




## === cell 20
submission.head()




## === cell 21
submission.to_csv("submission.csv", index=False)




## === cell 22
pass
