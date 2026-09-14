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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
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
from keras import layers, models, optimizers, datasets, preprocessing
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

base_dir='../input'
train_dir=os.path.join(base_dir, '')
test_dir=os.path.join(base_dir, '')


def load_data():
    data = pd.read_csv(os.path.join(train_dir, 'train.tsv'), sep='\t')
    x_train = [line['Phrase'] for i, line in data.iterrows()]
    y_train = [line['Sentiment'] for i, line in data.iterrows()]
    return x_train, y_train

def get_model():
    main_input = layers.Input(shape=(max_len,))
    embedded = layers.Embedding(max_features, 8, input_length=max_len)(main_input)
    flattened = layers.Flatten()(embedded)
    dense1 = layers.Dense(5, activation='softmax')(flattened)
    model = models.Model(inputs=main_input, outputs=dense1)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model


max_features = 10000
max_len = 100

x_train, y_train = load_data()
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
y_train = np_utils.to_categorical(y_train)
model = get_model()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    )


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_data = pd.read_csv(os.path.join(test_dir, 'test.tsv'), sep='\t')
x_test = [line['Phrase'] for i, line in test_data.iterrows()]
x_test = tokenizer.texts_to_sequences(x_test)
x_test= preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
test_pred = model.predict(x_test)
test_pred = test_pred.argmax(axis=1)
pred_df = pd.DataFrame({'PhraseId':test_data['PhraseId'].values, 'Sentiment': test_pred})
pred_df.to_csv('submit.csv', index=False)
