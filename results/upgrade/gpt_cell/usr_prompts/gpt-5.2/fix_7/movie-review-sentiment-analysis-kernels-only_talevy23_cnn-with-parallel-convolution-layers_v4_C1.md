# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
seed = 12345
import random
import numpy as np

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(seed)
np.random.seed(seed)


## === cell 1
train,test,sampleSubmission = pd.read_csv('../input/train.tsv', sep = '\t'),pd.read_csv('../input/test.tsv', sep = '\t'),pd.read_csv('../input/sampleSubmission.csv')

train.head(3)


## === cell 2
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

phrases = [clean_str(s) for s in train['Phrase']]


## === cell 3
type(train['Phrase'])
len(phrases[2].split())
phrases[2].split()


## === cell 4
sent = train["Sentiment"].values.astype(int)
num_classes = int(sent.max()) + 1
Y = np.eye(num_classes, dtype=np.float32)[sent]

print(Y[155:165])
print(train["Sentiment"].values[155:165])


## === cell 5
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

t = Tokenizer()
t.fit_on_texts(phrases)
vocab_size = len(t.word_index) + 1
X = t.texts_to_sequences(phrases)
max_length = max([len(test.split()) for test in phrases])
X = pad_sequences(X, maxlen=max_length, padding="post")
print(X.shape)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,merge, concatenate, Input

def model1(output_dim=8, max_length=50, y_dim=5, filter_sizes = [3]):
    model = Sequential()
    model.add(Embedding(vocab_size,output_dim,input_length=max_length))
    model.add(Flatten())
    model.add(Dense(y_dim, activation = 'sigmoid'))

    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())
    return model

def model2(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5]):
    model = Sequential()
    model.add(Embedding(vocab_size,output_dim,input_length=max_length))

    model.add(SpatialDropout1D(0.2))

    model.add(Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=max_length-filter_sizes[0]+1, strides=1, padding='valid'))

    model.add(Flatten())
    model.add(Dense(y_dim, activation = 'sigmoid'))

    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())
    return model

def model3(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5]):
    embed_input = Input(shape=(max_length,))
    x = Embedding(vocab_size,output_dim,input_length=max_length)(embed_input)
    x = SpatialDropout1D(0.2)(x)
    conv1 = Conv1D(num_filters, kernel_size=filter_sizes[0], padding='valid', activation='relu')(x)
    conv1 = MaxPooling1D(pool_size=max_length-filter_sizes[0]+1, strides=1, padding='valid')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=filter_sizes[1], padding='valid', activation='relu')(x)
    conv2 = MaxPooling1D(pool_size=max_length-filter_sizes[1]+1, strides=1, padding='valid')(conv2)    
    merge = concatenate([conv1,conv2])

    x = Flatten()(merge)
    predictions = Dense(y_dim, activation = 'sigmoid')(x)
    
    model = Model(inputs=embed_input,outputs=predictions)

    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())
    
    from keras.utils import plot_model
    plot_model(model, to_file='shared_input_layer.png')
    from IPython.display import Image
    Image(filename='shared_input_layer.png') 
    
    return model

model = model3(output_dim=8, max_length=max_length,y_dim=5,filter_sizes = [3,5])
