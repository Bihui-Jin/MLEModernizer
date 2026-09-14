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
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.4999971921565668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt     
import seaborn as sns

import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer   
from wordcloud import WordCloud, STOPWORDS

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
from keras.layers import Convolution1D, GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, Adam

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

import gc
import warnings
warnings.filterwarnings("ignore")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

## === cell 6
train.isnull().sum()

## === cell 7
print(train.shape, '\n')
train.info()

## === cell 8
train.head()

## === cell 10
plt.figure(figsize=(10,6))
graph_1 = sns.distplot(train[train['target'] > 0]['target'], color = 'red')
plt.title('Toxicity (Target) Distribution')
plt.xlabel("Toxicity Rate")
plt.ylabel("Distribution") 
plt.show()

## === cell 12
comment_adjective = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

plt.figure(figsize=(15,6))

for col in comment_adjective:
    graph_2 = sns.distplot(train[train[col] > 0][col], label=col, hist=False)
    plt.xlabel("Rate", fontsize=16)
    plt.ylabel("Distribution", fontsize=16)
    plt.legend(loc=1, prop={'size': 14})

plt.show()

## === cell 17
!mkdir '/root/.kaggle'

## === cell 18
!cp '../input/my-json/kaggle.json' '/root/.kaggle'

## === cell 19
!kaggle datasets download -d danielwillgeorge/glove6b100dtxt

## === cell 20
!unzip "./glove6b100dtxt.zip"

## === cell 22
""" GLOVE_EMBEDDING_PATH = "./glove.6B.100d.txt"

import pickle

def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr

glove_embeddings = load_embeddings(GLOVE_EMBEDDING_PATH)
print('Found and loaded {} word vectors'.format(len(glove_embeddings)))

#!rm "./glove.840B.300d.pkl"

"check_coverage" goes through a given vocabulary and tries to find word vectors in embedding matrix. "build_vocab" builds a ordered dictionary of words and their frequency in the text corpus.

import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def build_vocab(sentences, verbose =  True):
    """ """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """ """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

vocab = build_vocab(list(train['comment_text'].apply(lambda x:x.split())))
oov = check_coverage(vocab,glove_embeddings)
oov[:10]

### Symbols in GloVe

import string
letter_digit_list = string.ascii_letters + string.digits + ' '
letter_digit_list += "'"

Symbols that have embedding vectors in GloVe:

glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])
glove_symbols = ''.join([c for c in glove_chars if not c in letter_digit_list])
glove_symbols

Symbols that have no embedding vectors in GloVe:

jigsaw_chars = build_vocab(list(train["comment_text"]))

jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in letter_digit_list])
jigsaw_symbols

symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])
symbols_to_delete

symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])
symbols_to_isolate

del glove_embeddings
del vocab
del glove_chars
del glove_symbols

gc.collect() """

## === cell 23




""" def handle_punctuation(x):
    x = x.replace(remove_dict)
    x = x.replace(isolate_dict)
    return x """

"""def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x """
def cleaning_text(x):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    x = clean_special_chars(x, punct)
    return x

"""def handle_contractions(x):
    x = tokenizer.tokenize(x)
    return x """

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def preprocess(x):
    x = cleaning_text(x)
    x = handle_contractions(x)
    x = fix_quote(x)
    return x

## === cell 24
train['comment_text'] = train['comment_text'].progress_apply(lambda x:preprocess(x))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4107855393.py in <cell line: 0>()
----> 1 train['comment_text'] = train['comment_text'].progress_apply(lambda x:preprocess(x))

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in inner(df, func, *args, **kwargs)
    915                 # on the df using our wrapper (which provides bar updating)
    916                 try:
--> 917                     return getattr(df, df_function)(wrapper, **kwargs)
    918                 finally:
    919                     t.close()

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in wrapper(*args, **kwargs)
    910                     # take a fast or slow code path; so stop when t.total==t.n
    911                     t.update(n=1 if not t.total or t.n < t.total else 0)
--> 912                     return func(*args, **kwargs)
    913 
    914                 # Apply the provided function (in **kwargs)

/tmp/ipykernel_11/4107855393.py in <lambda>(x)
----> 1 train['comment_text'] = train['comment_text'].progress_apply(lambda x:preprocess(x))

/tmp/ipykernel_11/2519369907.py in preprocess(x)
     38     #x = handle_punctuation(x)
     39     x = cleaning_text(x)
---> 40     x = handle_contractions(x)
     41     x = fix_quote(x)
     42     return x

NameError: name 'handle_contractions' is not defined

## === cell 26
MAX_NUM_WORDS = 10000
TARGET_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train[TEXT_COLUMN])

MAX_SEQUENCE_LENGTH = 250
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3589352414.py in <cell line: 0>()
      5 # Create a text tokenizer.
      6 tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
----> 7 tokenizer.fit_on_texts(train[TEXT_COLUMN])
      8 
      9 # All comments must be truncated or padded to be the same length.

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/text.py in fit_on_texts(self, texts)
    131             else:
    132                 if self.analyzer is None:
--> 133                     seq = text_to_word_sequence(
    134                         text,
    135                         filters=self.filters,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/text.py in text_to_word_sequence(input_text, filters, lower, split)
     20     """DEPRECATED."""
     21     if lower:
---> 22         input_text = input_text.lower()
     23 
     24     translate_dict = {c: split for c in filters}

AttributeError: 'float' object has no attribute 'lower'

## === cell 27
identity_columns = ['asian', 'atheist',
       'bisexual', 'black', 'buddhist', 'christian', 'female',
       'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
       'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
       'muslim', 'other_disability', 'other_gender',
       'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']
weights = np.ones((len(train),)) / 4
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()
train_label = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/758906886.py in <cell line: 0>()
     10 weights = np.ones((len(train),)) / 4
     11 # Subgroup
---> 12 weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
     13 # Background Positive, Subgroup Negative
     14 weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

## === cell 30
"""
!kaggle datasets download -d fizzbuzz/cleaned-toxic-comments

!unzip "./cleaned-toxic-comments.zip"

pre_data = "./train_preprocessed.csv"


## Get the Corpus of all the comments and related Toxicity fields

data = pd.read_csv(pre_data)
data.head()

## Dividing the dataset into features and labels:
Features = "comment"            
Labels = "toxicity"

Features = data['comment_text']
Labels = np.array([0 if y == 0 else 1 for y in data['toxicity']])

### Tokenizing and preprocessing the data

NUM_WORDS = 40000 # Maximum number of unique words which need to be tokenized
MAXLEN = 50 # Maximum length of a sentence/ comment
PADDING = 'post' # The type of padding done for sentences shorter than the Max len

tokenizer = Tokenizer(num_words=NUM_WORDS)

# Fit the tokenizer on the comments 
tokenizer.fit_on_texts(Features)

# Get the word index of the top 20000 words from the dataset
word_idx = tokenizer.word_index

# Convert the string sentence to a sequence of their numerical values
Feature_sequences = tokenizer.texts_to_sequences(Features)

# Pad the sequences to make them of uniform length
padded_sequences = pad_sequences(Feature_sequences, maxlen = MAXLEN, padding = PADDING)

print("The Transformation of sentence::")
print("\n\nThe normal Sentencen:\n")
print(Features[2])
print("\n\nThe tokenized sequence:\n")
print(Feature_sequences[2])
print("\n\nThe padded sequence:\n")
print(padded_sequences[2])

# Convert to array for passing through the model
X = np.array(padded_sequences)
"""

## === cell 33
EMBEDDINGS_PATH = './glove.6B.100d.txt'
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 10
BATCH_SIZE = 128

train_text = pad_text(train[TEXT_COLUMN], tokenizer)

del train
gc.collect()

X_train, X_test, y_train, y_test=train_test_split(train_text, train_label, test_size=0.20, random_state=42)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2388380271.py in <cell line: 0>()
      7 
      8 # Prepare data
----> 9 train_text = pad_text(train[TEXT_COLUMN], tokenizer)
     10 
     11 del train

NameError: name 'pad_text' is not defined

## === cell 34
gc.collect()

## === cell 36
print('loading embeddings')
embeddings_index = {}
with open(EMBEDDINGS_PATH) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,
                                 EMBEDDINGS_DIMENSION))
num_words_in_embedding = 0
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        num_words_in_embedding += 1
        embedding_matrix[i] = embedding_vector

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/271813590.py in <cell line: 0>()
      2 print('loading embeddings')
      3 embeddings_index = {}
----> 4 with open(EMBEDDINGS_PATH) as f:
      5     for line in f:
      6         values = line.split()

FileNotFoundError: [Errno 2] No such file or directory: './glove.6B.100d.txt'

## === cell 37
model=Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1,100,input_length = 100,weights = [embedding_matrix],trainable = False))
model.add(Bidirectional(LSTM(100,return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(2,activation='softmax'))

print('Compiling model...')
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=LEARNING_RATE),
                  metrics=['acc'])
print("Compiled model!")

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2786884314.py in <cell line: 0>()
      1 # Create model layers.
      2 model=Sequential()
----> 3 model.add(Embedding(len(tokenizer.word_index) + 1,100,input_length = 100,weights = [embedding_matrix],trainable = False))
      4 model.add(Bidirectional(LSTM(100,return_sequences=True)))
      5 model.add(GlobalAveragePooling1D())

NameError: name 'embedding_matrix' is not defined

## === cell 40
"""
history = model.fit(
            X, 
            Labels,
            batch_size = 128,
            epochs = 10,
            validation_split = 0.2, # 20 percent data reserved for validation to avoid or monitor overfitting/ underfitting
            verbose = verbose,
        )
"""

## === cell 42
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_lstm_toxic.h5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2091486119.py in <cell line: 0>()
----> 1 model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      2     filepath="best_lstm_toxic.h5",
      3     save_weights_only=True,
      4     monitor='val_accuracy',
      5     mode='max',

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=best_lstm_toxic.h5

## === cell 43
import time

## === cell 44
print('Training model...')
start = time.time()
history = model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE,
              epochs= 2,
              validation_data=(X_test, y_test),
              verbose=1, callbacks = [model_checkpoint_callback])

end = time.time()
print("Training duration: {}".format(str(end-start)))

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649580532.py in <cell line: 0>()
      2 print('Training model...')
      3 start = time.time()
----> 4 history = model.fit(X_train,
      5               y_train,
      6               batch_size=BATCH_SIZE,

NameError: name 'X_train' is not defined

## === cell 45
print("Training duration: {} minutes".format(str((end-start)/60)))

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1920069209.py in <cell line: 0>()
----> 1 print("Training duration: {} minutes".format(str((end-start)/60)))

NameError: name 'end' is not defined

## === cell 46
del X_train, y_train, X_test, y_test
gc.collect()

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600831509.py in <cell line: 0>()
----> 1 del X_train, y_train, X_test, y_test
      2 gc.collect()

NameError: name 'X_train' is not defined

## === cell 48
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1529017351.py in <cell line: 0>()
      1 # Plot training & validation accuracy values
----> 2 plt.plot(history.history['acc'])
      3 plt.plot(history.history['val_acc'])
      4 plt.title('Model accuracy')
      5 plt.ylabel('Accuracy')

NameError: name 'history' is not defined

## === cell 49
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1212291865.py in <cell line: 0>()
      1 # Plot training & validation loss values
----> 2 plt.plot(history.history['loss'])
      3 plt.plot(history.history['val_loss'])
      4 plt.title('Model loss')
      5 plt.ylabel('Loss')

NameError: name 'history' is not defined

## === cell 50
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

## === cell 51
submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]
submission.to_csv('submission.csv')

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1061430037.py in <cell line: 0>()
----> 1 submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]
      2 submission.to_csv('submission.csv')

NameError: name 'pad_text' is not defined
