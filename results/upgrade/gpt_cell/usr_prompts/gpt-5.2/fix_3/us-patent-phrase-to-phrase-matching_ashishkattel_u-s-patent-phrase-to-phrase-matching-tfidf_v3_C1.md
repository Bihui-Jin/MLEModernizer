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

0.4719

# 6. Current score

0.52481

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.52481) has done: 'Diagnosis: Cell 21 crashes because `test_anchor` and `test_target` are SciPy sparse `csr_matrix` objects, and in this environment they don’t expose the `.A` shorthand attribute. The training code used `.toarray()` earlier, so converting the test sparse matrices the same way preserves identical feature construction semantics. The fix is to replace `.A` with `.toarray()` (or an equivalent dense conversion) so `np.hstack` receives NumPy arrays.

Patch summary: Update cell 21 to use `toarray()` for both sparse matrices and keep the remaining concatenation unchanged, ensuring `test_x` stays a dense NumPy array with the same layout as `train_x`.

Updated cells: (cell 21 only)

Compatibility notes for cell k+1: `test_x` remains a dense `ndarray`, so `clf.predict(test_x)` in the next cell continues to work without any interface changes.

Assumptions: SciPy is available as a dependency of scikit-learn and the returned matrices support `.toarray()`; memory usage is acceptable since the original notebook already densifies training features similarly.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## === cell 2
train_data = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")


## === cell 3
lengths_nachor = train_data["anchor"].apply(lambda x:len(x.split(" ")))
lengths_target = train_data["target"].apply(lambda x:len(x.split(" ")))
print(max(lengths_nachor),np.argmax(lengths_target))


## === cell 4
from nltk.stem import PorterStemmer
ps = PorterStemmer()


## === cell 5
def find_common_word(words):
    
    word1 = words[0].lower()
    word2 = words[1].lower()
    common = set()
    w1 = []
    w2 = []
    for w in word1.split(" "):
        w1.append(ps.stem(w))
    for w in word2.split(" "):
        w2.append(ps.stem(w))
       
    common.update(w1)
    common.update(w2)
    if len(common) == len(w1)+len(w2):
        return 0
    else:
        value = len(common) - (len(w1)+len(w2))
        return abs(value)/(len(w1)+len(w2))
    
train_data["common"] = train_data[["anchor","target"]].apply(find_common_word,axis=1)


## === cell 6
train_data['modifed_score'] = train_data["score"].map(
{0. : 1,
0.25 : 2,
0.5 : 3,
0.75 : 4,
1 : 5
}
)


## === cell 7
y = train_data["modifed_score"]
x = train_data.drop(columns=["modifed_score","score"])


## === cell 8
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_anchor = TfidfVectorizer()
anchor_tfid = vectorizer_anchor.fit_transform(x.anchor.values)


## === cell 9
vectorizer_target = TfidfVectorizer()
target_tfid = vectorizer_target.fit_transform(x.target.values)


## === cell 10
x.drop(columns=["anchor","target"],inplace=True)


## === cell 11
x.drop(columns=["id","context"],inplace=True)


## === cell 12
train_x = np.hstack([anchor_tfid.toarray(), target_tfid.toarray(), x.values])


## === cell 13
from sklearn.linear_model import LogisticRegression


## === cell 14
clf = LogisticRegression()
clf.fit(train_x,y.values)
pred = clf.predict(train_x)
from sklearn.metrics import accuracy_score
accuracy_score(y.values,pred)


## === cell 16
test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
test.head()


## === cell 17
test.drop(columns=["id","context"],inplace=True)


## === cell 18
test["common"] = test[["anchor","target"]].apply(find_common_word,axis=1)


## === cell 19
test_anchor = vectorizer_anchor.transform(test["anchor"])


## === cell 20
test_target = vectorizer_target.transform(test["target"])


## === cell 21
test_x = np.hstack(
    [
        test_anchor.toarray(),
        test_target.toarray(),
        np.reshape(test["common"].values, (test.shape[0], 1)),
    ]
)


## === cell 23
ls_pred = clf.predict(test_x)


## === cell 24
test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
submission = pd.DataFrame({
    "id":test.id,
    "score":ls_pred
})


## === cell 25
submission.score=submission.score.map({
    1 : 0,
     2: 0.25,
3: 0.5,
4 : 0.75,
5 : 1
})


## === cell 26
submission.to_csv("submission.csv",index=False)
