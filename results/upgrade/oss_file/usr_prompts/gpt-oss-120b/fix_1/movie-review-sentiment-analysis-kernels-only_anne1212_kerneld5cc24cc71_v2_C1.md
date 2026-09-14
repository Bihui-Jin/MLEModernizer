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
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.contrib.layers import fully_connected, dropout
from nltk.tokenize import TweetTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
submission = pd.read_csv('../input/submission/submission.csv')
submission.to_csv('submission.csv',index = False)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/330785863.py in <cell line: 0>()
----> 1 submission = pd.read_csv('../input/submission/submission.csv')
      2 submission.to_csv('submission.csv',index = False)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/submission/submission.csv'

## === cell 3
train_data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
test_data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')

## === cell 5
train_data.head()

## === cell 6
train_data["SentenceId"].unique().size

## === cell 7
corpus = train_data['Phrase'].append(test_data['Phrase']).values

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3638349411.py in <cell line: 0>()
----> 1 corpus = train_data['Phrase'].append(test_data['Phrase']).values

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'Series' object has no attribute 'append'

## === cell 8
assert corpus.shape[0] == train_data['Phrase'].shape[0] + test_data['Phrase'].shape[0]

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4255736062.py in <cell line: 0>()
----> 1 assert corpus.shape[0] == train_data['Phrase'].shape[0] + test_data['Phrase'].shape[0]

NameError: name 'corpus' is not defined

## === cell 9
def tokenize(x):
    return [x] if len(x) == 1 else TweetTokenizer().tokenize(x)
     
tfVectorizer = TfidfVectorizer(tokenizer=tokenize)
tf_idf_total = tfVectorizer.fit_transform(corpus)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3529689008.py in <cell line: 0>()
      3 
      4 tfVectorizer = TfidfVectorizer(tokenizer=tokenize)
----> 5 tf_idf_total = tfVectorizer.fit_transform(corpus)

NameError: name 'corpus' is not defined

## === cell 10
tf_idf_total = tf_idf_total.todense()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/713548823.py in <cell line: 0>()
----> 1 tf_idf_total = tf_idf_total.todense()

NameError: name 'tf_idf_total' is not defined

## === cell 11
tf_idf_dense = tf_idf_total[:train_data['Phrase'].shape[0]]

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1483678331.py in <cell line: 0>()
----> 1 tf_idf_dense = tf_idf_total[:train_data['Phrase'].shape[0]]

NameError: name 'tf_idf_total' is not defined

## === cell 12
tf_idf_dense.shape

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1526820381.py in <cell line: 0>()
----> 1 tf_idf_dense.shape

NameError: name 'tf_idf_dense' is not defined

## === cell 13
tfVectorizer.get_params()

## === cell 14
train_vocab = tfVectorizer.vocabulary_
train_vocab

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2760221937.py in <cell line: 0>()
----> 1 train_vocab = tfVectorizer.vocabulary_
      2 train_vocab

AttributeError: 'TfidfVectorizer' object has no attribute 'vocabulary_'

## === cell 15
tfVectorizer.get_feature_names()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4161934352.py in <cell line: 0>()
----> 1 tfVectorizer.get_feature_names()

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 16
def top_tfidf_words(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_words_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_words(row, features, top_n)

def top_mean_words(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_words(tfidf_means, features, top_n)

def top_words_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_words(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classWords_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

## === cell 17
def get_batch(index, batch_size = 2000):
    
    index = shuffle(index)
    
    batch_index = []
    for sample in index:
        batch_index.append(sample)
        
        if len(batch_index) == batch_size:
            yield batch_index
            batch_index= []
        
    if len(batch_index) > 0:
        yield batch_index

## === cell 19
test_data.head()

## === cell 20
tf_idf_dense2 = tf_idf_total[train_data['Phrase'].shape[0]:]
tf_idf_dense2.shape

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3686579787.py in <cell line: 0>()
----> 1 tf_idf_dense2 = tf_idf_total[train_data['Phrase'].shape[0]:]
      2 tf_idf_dense2.shape

NameError: name 'tf_idf_total' is not defined

## === cell 21
assert len(tf_idf_dense) + len(tf_idf_dense2) == len(tf_idf_total)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2439333648.py in <cell line: 0>()
----> 1 assert len(tf_idf_dense) + len(tf_idf_dense2) == len(tf_idf_total)

NameError: name 'tf_idf_dense' is not defined

## === cell 22
tfVectorizer2.get_params()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2770381358.py in <cell line: 0>()
----> 1 tfVectorizer2.get_params()

NameError: name 'tfVectorizer2' is not defined

## === cell 23
test_vocab = tfVectorizer2.vocabulary_

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3964175713.py in <cell line: 0>()
----> 1 test_vocab = tfVectorizer2.vocabulary_

NameError: name 'tfVectorizer2' is not defined

## === cell 24
tfVectorizer2.get_feature_names()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775731170.py in <cell line: 0>()
----> 1 tfVectorizer2.get_feature_names()

NameError: name 'tfVectorizer2' is not defined

## === cell 25
n_sentance, n_words = tf_idf_dense.shape

input_sentence = tf.placeholder(dtype=tf.float32, shape = [None, n_words])
labels = tf.placeholder(dtype=tf.int32, shape = [None])
hidden_layer1 = fully_connected(input_sentence, num_outputs =  50)
hidden_layer1 = dropout(hidden_layer1, keep_prob=0.7)
hidden_layer2 = fully_connected(hidden_layer1, num_outputs =  30)
hidden_layer2 = dropout(hidden_layer2, keep_prob=0.8)
output = fully_connected(hidden_layer2, activation_fn=None, num_outputs =  5)

label_onehot = tf.one_hot(labels, depth = 5)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = label_onehot, logits = output)
loss = tf.reduce_mean(loss)
y_ = tf.nn.softmax(output)

optimize = tf.train.AdamOptimizer().minimize(loss)

predictions = tf.cast(tf.argmax(y_,1), tf.int32)
acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
batch_acc = 1 - tf.reduce_mean(tf.cast(tf.cast(labels - predictions, tf.bool), tf.float32))

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3283689035.py in <cell line: 0>()
----> 1 n_sentance, n_words = tf_idf_dense.shape
      2 
      3 input_sentence = tf.placeholder(dtype=tf.float32, shape = [None, n_words])
      4 labels = tf.placeholder(dtype=tf.int32, shape = [None])
      5 hidden_layer1 = fully_connected(input_sentence, num_outputs =  50)

NameError: name 'tf_idf_dense' is not defined

## === cell 27
x_train, x_vad, y_train, y_vad = train_test_split(tf_idf_dense, train_data['Sentiment'], test_size=0.2)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2382538141.py in <cell line: 0>()
----> 1 x_train, x_vad, y_train, y_vad = train_test_split(tf_idf_dense, train_data['Sentiment'], test_size=0.2)

NameError: name 'train_test_split' is not defined

## === cell 28
print(x_train.shape)
print(x_vad.shape)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/5909890.py in <cell line: 0>()
----> 1 print(x_train.shape)
      2 print(x_vad.shape)

NameError: name 'x_train' is not defined

## === cell 29
i, j, max_epoch, patience, loss_max, = 0, 0, 50, 5, np.inf 
n = 1
optimal_i = i

## === cell 30

optimal_i = 38
gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='0')
with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    index = np.arange(len(tf_idf_dense))
    for i in range(optimal_i):
        sess.run(tf.local_variables_initializer())
        for batch_index in list(get_batch(index)):
            batch_x, batch_y = tf_idf_dense[batch_index], train_data['Sentiment'].values[batch_index]
            _, loss_value, accuracy = sess.run([optimize, loss, acc_op], feed_dict= {input_sentence: batch_x,  labels: batch_y})
        print('accuracy for epoch {} is {}, loss is {}'.format(i, accuracy, loss_value)) 
        
    sess.run(tf.local_variables_initializer())
    pred_value = sess.run(predictions, feed_dict = {input_sentence: tf_idf_dense2}) 
    print(pred_value.shape)
    sentiment = test_data[['PhraseId']]
    sentiment['Sentiment'] = pred_value
    sentiment.to_csv('submission.csv', index = False)
    
tf.reset_default_graph()

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2361951454.py in <cell line: 0>()
      1 optimal_i = 38
----> 2 gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='0')
      3 with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
      4     # retrain model by using all data with optimal_i
      5     sess.run(tf.global_variables_initializer())

AttributeError: module 'tensorflow' has no attribute 'GPUOptions'
