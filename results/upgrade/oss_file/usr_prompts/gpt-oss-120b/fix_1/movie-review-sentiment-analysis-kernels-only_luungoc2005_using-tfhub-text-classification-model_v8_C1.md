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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.58945

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
seed = 197

import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
print(tf.__version__)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import pandas as pd

train_df = pd.read_csv('../input/train.tsv',  sep="\t")
test_df = pd.read_csv('../input/test.tsv',  sep="\t")


## === cell 3
train_df.head()


## === cell 4
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/374247031.py in <cell line: 0>()
----> 1 train_input_fn = tf.estimator.inputs.pandas_input_fn(
      2     train_df, train_df["Sentiment"], num_epochs=None, shuffle=True)
      3 
      4 # Prediction on the whole training set.
      5 predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 5
embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1")


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2933501043.py in <cell line: 0>()
----> 1 embedded_text_feature_column = hub.text_embedding_column(
      2     key="Phrase",
      3     module_spec="https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1")

AttributeError: module 'tensorflow_hub' has no attribute 'text_embedding_column'

## === cell 6
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=5,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/784820771.py in <cell line: 0>()
----> 1 estimator = tf.estimator.DNNClassifier(
      2     hidden_units=[500, 100],
      3     feature_columns=[embedded_text_feature_column],
      4     n_classes=5,
      5     optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 7
steps = 10000


## === cell 8
estimator.train(input_fn=train_input_fn, steps=steps)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3489762622.py in <cell line: 0>()
----> 1 estimator.train(input_fn=train_input_fn, steps=steps)

NameError: name 'estimator' is not defined

## === cell 9
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1647668735.py in <cell line: 0>()
----> 1 train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
      2 
      3 print("Training set accuracy: {accuracy}".format(**train_eval_result))

NameError: name 'estimator' is not defined

## === cell 10
sub = pd.read_csv('../input/sampleSubmission.csv')
sub.head()


## === cell 11
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["Phrase"], shuffle=False)

def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

sub['Sentiment'] = get_predictions(estimator, predict_test_input_fn)

sub.to_csv('sub_tfhub.csv', index=False)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3199482873.py in <cell line: 0>()
----> 1 predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
      2     test_df, test_df["Phrase"], shuffle=False)
      3 
      4 def get_predictions(estimator, input_fn):
      5     return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 12
sub.head()
