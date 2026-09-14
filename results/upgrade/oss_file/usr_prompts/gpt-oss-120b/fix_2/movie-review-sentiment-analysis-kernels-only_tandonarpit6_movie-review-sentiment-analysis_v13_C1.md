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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.64062

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.tsv", sep="\t")
test = pd.read_csv("../input/test.tsv", sep="\t")
submission = pd.read_csv("../input/sampleSubmission.csv")



## === cell 2
ytrain = train["Sentiment"].values
from keras.utils import to_categorical

ytrain_cat = to_categorical(ytrain, num_classes=5)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
from keras.preprocessing.text import Tokenizer

MAX_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train["Phrase"].astype(str))

xtrain = tokenizer.texts_to_matrix(train["Phrase"].astype(str), mode="binary")
xtest = tokenizer.texts_to_matrix(test["Phrase"].astype(str), mode="binary")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/526442483.py in <cell line: 0>()
      1 # Text tokenization → dense matrix (binary presence)
----> 2 from keras.preprocessing.text import Tokenizer
      3 
      4 MAX_WORDS = 20000
      5 tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

ModuleNotFoundError: No module named 'keras.preprocessing.text'

## === cell 4
from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_shape=(xtrain.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(xtrain, ytrain_cat, epochs=8, batch_size=512, verbose=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4112273629.py in <cell line: 0>()
      3 
      4 model = models.Sequential()
----> 5 model.add(layers.Dense(256, activation="relu", input_shape=(xtrain.shape[1],)))
      6 model.add(layers.Dropout(0.2))
      7 model.add(layers.Dense(256, activation="relu"))

NameError: name 'xtrain' is not defined

## === cell 5
ypred_nn = model.predict(xtest, verbose=0)
pred_nn = np.argmax(ypred_nn, axis=1)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2411937179.py in <cell line: 0>()
      1 # Predict on test set
----> 2 ypred_nn = model.predict(xtest, verbose=0)
      3 pred_nn = np.argmax(ypred_nn, axis=1)
      4 

NameError: name 'xtest' is not defined

## === cell 6
submission["Sentiment"] = pred_nn
submission.to_csv("sampleSubmission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2262574871.py in <cell line: 0>()
      1 # Prepare submission file
----> 2 submission["Sentiment"] = pred_nn
      3 submission.to_csv("sampleSubmission.csv", index=False)

NameError: name 'pred_nn' is not defined
