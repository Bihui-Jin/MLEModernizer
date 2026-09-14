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

No external packages required in the script and installed.

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

0.6226844868159054

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/movie-review-sentiment-analysis-kernels-only/train.tsv"
test_path = "../input/movie-review-sentiment-analysis-kernels-only/test.tsv"

train_data = pd.read_csv(train_path, sep="\t")
test_data = pd.read_csv(test_path, sep="\t")




## === cell 2
def tokenize(x):
    return TweetTokenizer().tokenize(x) if len(x) > 1 else [x]


vectorizer = TfidfVectorizer(tokenizer=tokenize)
corpus = pd.concat([train_data["Phrase"], test_data["Phrase"]]).values
tf_idf_total = vectorizer.fit_transform(corpus).toarray()

X_train_full = tf_idf_total[: train_data.shape[0]]
X_test = tf_idf_total[train_data.shape[0] :]
y_full = train_data["Sentiment"].values


## === cell 3
x_train, x_val, y_train, y_val = train_test_split(
    X_train_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)




## === cell 4
def get_batch(index, batch_size=2000):
    index = shuffle(index)
    batch = []
    for i in index:
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch




## === cell 5
tf.compat.v1.reset_default_graph()
n_features = X_train_full.shape[1]

input_sentence = tf.compat.v1.placeholder(
    tf.float32, shape=[None, n_features], name="input"
)
labels_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="labels")

hidden1 = tf.compat.v1.layers.dense(
    input_sentence, 50, activation=tf.nn.relu, name="fc1"
)
hidden1 = tf.compat.v1.layers.dropout(hidden1, rate=0.3, name="drop1")  # keep_prob=0.7
hidden2 = tf.compat.v1.layers.dense(hidden1, 30, activation=tf.nn.relu, name="fc2")
hidden2 = tf.compat.v1.layers.dropout(hidden2, rate=0.2, name="drop2")  # keep_prob=0.8
logits = tf.compat.v1.layers.dense(hidden2, 5, activation=None, name="out")

one_hot = tf.one_hot(labels_ph, depth=5)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logits)
)
optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

predictions = tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int32)
accuracy, acc_op = tf.compat.v1.metrics.accuracy(
    labels=labels_ph, predictions=predictions
)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2253723582.py in <cell line: 0>()
      8 labels_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="labels")
      9 
---> 10 hidden1 = tf.compat.v1.layers.dense(
     11     input_sentence, 50, activation=tf.nn.relu, name="fc1"
     12 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `dense` is not available with Keras 3.

## === cell 6
epochs = 30
batch_size = 2000
index_all = np.arange(x_train.shape[0])

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    for epoch in range(epochs):
        sess.run(tf.compat.v1.local_variables_initializer())  # reset metric
        for batch_idx in get_batch(index_all, batch_size):
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            sess.run(optimizer, feed_dict={input_sentence: batch_x, labels_ph: batch_y})
        acc_val, _ = sess.run(
            [accuracy, acc_op], feed_dict={input_sentence: x_val, labels_ph: y_val}
        )
        print(f"Epoch {epoch+1}/{epochs} – validation accuracy: {acc_val:.4f}")

    pred_test = sess.run(predictions, feed_dict={input_sentence: X_test})


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/404487777.py in <cell line: 0>()
     13             batch_x = x_train[batch_idx]
     14             batch_y = y_train[batch_idx]
---> 15             sess.run(optimizer, feed_dict={input_sentence: batch_x, labels_ph: batch_y})
     16         acc_val, _ = sess.run(
     17             [accuracy, acc_op], feed_dict={input_sentence: x_val, labels_ph: y_val}

NameError: name 'optimizer' is not defined

## === cell 7
submission = pd.DataFrame({"PhraseId": test_data["PhraseId"], "Sentiment": pred_test})
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3573732777.py in <cell line: 0>()
      1 # Create submission file
----> 2 submission = pd.DataFrame({"PhraseId": test_data["PhraseId"], "Sentiment": pred_test})
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission written to submission.csv")

NameError: name 'pred_test' is not defined
