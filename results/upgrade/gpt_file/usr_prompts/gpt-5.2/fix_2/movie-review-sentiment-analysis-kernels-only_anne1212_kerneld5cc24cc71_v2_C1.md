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
import os
import numpy as np
import pandas as pd

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

try:
    from nltk.tokenize import TweetTokenizer

    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, scope=None):
    return tf.compat.v1.layers.dense(
        inputs, units=num_outputs, activation=activation_fn, name=scope
    )


def dropout(inputs, keep_prob=0.5, scope=None):
    rate = 1.0 - keep_prob
    return tf.compat.v1.layers.dropout(inputs, rate=rate, training=True, name=scope)


np.random.seed(42)
tf.compat.v1.set_random_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pass



## === cell 2
CANDIDATE_BASES = [
    "/kaggle/input/movie-review-sentiment-analysis-kernels-only",
    "/kaggle/data/movie-review-sentiment-analysis-kernels-only",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {CANDIDATE_BASES}"
    )


train_path = find_file("train.tsv")
test_path = find_file("test.tsv")

train_data = pd.read_csv(train_path, sep="\t")
test_data = pd.read_csv(test_path, sep="\t")

print(train_data.shape, test_data.shape)
print(train_data.columns.tolist())



## === cell 3
assert {"PhraseId", "Phrase", "Sentiment"}.issubset(set(train_data.columns))
assert {"PhraseId", "Phrase"}.issubset(set(test_data.columns))
assert train_data["Sentiment"].between(0, 4).all()



## === cell 4
train_data.head()



## === cell 5
train_data["SentenceId"].unique().size



## === cell 6
corpus = pd.concat(
    [train_data["Phrase"], test_data["Phrase"]], axis=0, ignore_index=True
).values



## === cell 7
assert corpus.shape[0] == train_data["Phrase"].shape[0] + test_data["Phrase"].shape[0]




## === cell 8
def tokenize(x):
    if x is None:
        return []
    x = str(x)
    if len(x) == 1:
        return [x]
    if _HAS_NLTK:
        return TweetTokenizer().tokenize(x)
    return x.split()


tfVectorizer = TfidfVectorizer(tokenizer=tokenize)
tf_idf_total = tfVectorizer.fit_transform(corpus)



## === cell 9
tf_idf_total = tf_idf_total.todense()
tf_idf_total.shape



## === cell 10
tf_idf_dense = tf_idf_total[: train_data["Phrase"].shape[0]]
tf_idf_dense.shape



## === cell 11
tf_idf_dense.shape



## === cell 12
tfVectorizer.get_params()



## === cell 13
train_vocab = getattr(tfVectorizer, "vocabulary_", None)
type(train_vocab), (len(train_vocab) if train_vocab is not None else None)



## === cell 14
if hasattr(tfVectorizer, "get_feature_names_out"):
    feature_names = tfVectorizer.get_feature_names_out()
else:
    feature_names = tfVectorizer.get_feature_names()
feature_names[:10]




## === cell 15
def top_tfidf_words(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ["feature", "tfidf"]
    return df


def top_words_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(np.asarray(Xtr[row_id]))
    return top_tfidf_words(row, features, top_n)


def top_mean_words(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = np.asarray(Xtr[grp_ids])
    else:
        D = np.asarray(Xtr)

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_words(tfidf_means, features, top_n)


def top_words_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y == label)
        feats_df = top_mean_words(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classWords_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(num_class, 1, i + 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.barh(x, df.tfidf, align="center")
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1] + 1])
        ax.set_yticklabels(df.feature)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()




## === cell 16
def get_batch(index, batch_size=2000):
    index = shuffle(index)
    batch_index = []
    for sample in index:
        batch_index.append(sample)
        if len(batch_index) == batch_size:
            yield batch_index
            batch_index = []
    if len(batch_index) > 0:
        yield batch_index




## === cell 17
tf_idf_dense2 = tf_idf_total[train_data["Phrase"].shape[0] :]
assert len(tf_idf_dense) + len(tf_idf_dense2) == len(tf_idf_total)
tf_idf_dense2.shape



## === cell 18
test_data.head()



## === cell 19
tf_idf_dense2 = tf_idf_total[train_data["Phrase"].shape[0] :]
tf_idf_dense2.shape



## === cell 20
assert len(tf_idf_dense) + len(tf_idf_dense2) == len(tf_idf_total)



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
n_sentance, n_words = tf_idf_dense.shape

input_sentence = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_words])
labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])

hidden_layer1 = fully_connected(input_sentence, num_outputs=50)
hidden_layer1 = dropout(hidden_layer1, keep_prob=0.7)

hidden_layer2 = fully_connected(hidden_layer1, num_outputs=30)
hidden_layer2 = dropout(hidden_layer2, keep_prob=0.8)

output = fully_connected(hidden_layer2, activation_fn=None, num_outputs=5)

label_onehot = tf.one_hot(labels, depth=5)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_onehot, logits=output)
loss = tf.reduce_mean(loss)

y_ = tf.nn.softmax(output)
optimize = tf.compat.v1.train.AdamOptimizer().minimize(loss)

predictions = tf.cast(tf.argmax(y_, 1), tf.int32)
acc, acc_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)
batch_acc = 1 - tf.reduce_mean(
    tf.cast(tf.cast(labels - predictions, tf.bool), tf.float32)
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2428326673.py in <cell line: 0>()
      5 labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
      6 
----> 7 hidden_layer1 = fully_connected(input_sentence, num_outputs=50)
      8 hidden_layer1 = dropout(hidden_layer1, keep_prob=0.7)
      9 

/tmp/ipykernel_11/3781430051.py in fully_connected(inputs, num_outputs, activation_fn, scope)
     27 def fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, scope=None):
     28     # Equivalent to tf.contrib.layers.fully_connected with default relu
---> 29     return tf.compat.v1.layers.dense(
     30         inputs, units=num_outputs, activation=activation_fn, name=scope
     31     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `dense` is not available with Keras 3.

## === cell 25
tf_idf_dense = np.asarray(tf_idf_dense, dtype=np.float32)
tf_idf_dense2 = np.asarray(tf_idf_dense2, dtype=np.float32)
y_all = train_data["Sentiment"].values.astype(np.int32)

print(tf_idf_dense.shape, tf_idf_dense2.shape, y_all.shape)



## === cell 26
x_train, x_vad, y_train, y_vad = train_test_split(
    tf_idf_dense, y_all, test_size=0.2, random_state=42, stratify=y_all
)



## === cell 27
print(x_train.shape)
print(x_vad.shape)



## === cell 28
(
    i,
    j,
    max_epoch,
    patience,
    loss_max,
) = (
    0,
    0,
    50,
    5,
    np.inf,
)
n = 1
optimal_i = i



## === cell 29
optimal_i = 38

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    index = np.arange(len(tf_idf_dense))
    for i in range(optimal_i):
        sess.run(tf.compat.v1.local_variables_initializer())
        for batch_index in list(get_batch(index)):
            batch_x = tf_idf_dense[batch_index]
            batch_y = y_all[batch_index]
            _, loss_value, accuracy = sess.run(
                [optimize, loss, acc_op],
                feed_dict={input_sentence: batch_x, labels: batch_y},
            )
        print("accuracy for epoch {} is {}, loss is {}".format(i, accuracy, loss_value))

    sess.run(tf.compat.v1.local_variables_initializer())
    pred_value = sess.run(predictions, feed_dict={input_sentence: tf_idf_dense2})
    print(pred_value.shape)

    sentiment = test_data[["PhraseId"]].copy()
    sentiment["Sentiment"] = pred_value.astype(np.int32)
    sentiment.to_csv("submission.csv", index=False)

tf.compat.v1.reset_default_graph()

print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/707335255.py in <cell line: 0>()
     15             batch_y = y_all[batch_index]
     16             _, loss_value, accuracy = sess.run(
---> 17                 [optimize, loss, acc_op],
     18                 feed_dict={input_sentence: batch_x, labels: batch_y},
     19             )

NameError: name 'optimize' is not defined
