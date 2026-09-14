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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.54102

# 6. Current score

0.43189

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.43189) has done: 'I fix the TensorFlow import error by setting the protobuf implementation flag before importing TensorFlow, correct the cleaning logic, rewrite `get_phrase` to return a clean string instead of a NumPy array, and adjust the prediction loop so that the final `selected_text` column contains properly formatted strings. These changes resolve the runtime crashes and ensure a valid `submission.csv` is produced, while keeping the original model architecture and training approach intact.'

# 9. Code solution

## === cell 0
import os, re, string, numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_path = "../input/tweet-sentiment-extraction/train.csv"
data = pd.read_csv(train_path)



## === cell 4
print("Rows:", len(data), "Unique IDs:", len(data.textID.unique()))
print("Sample row:", data.head(1))



## === cell 5
data = data[pd.notnull(data.selected_text)].reset_index(drop=True)



## === cell 6
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data, test_size=0.25, random_state=42, stratify=data.sentiment
)



## === cell 7
vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower=True, split=" ")
tokenizer.fit_on_texts(train_df.text)
word_index = tokenizer.word_index



## === cell 8
max_len = 50
trunc_type = "post"
pad_type = "post"


def texts_to_padded(texts, maxlen=None):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=maxlen, truncating=trunc_type, padding=pad_type)


train_x = texts_to_padded(train_df.text, maxlen=max_len)
val_x = texts_to_padded(val_df.text, maxlen=max_len)


def texts_to_mask(texts, selected_texts):
    masks = []
    for txt, sel in zip(texts, selected_texts):
        txt_seq = tokenizer.texts_to_sequences([txt])[0]
        sel_seq = set(tokenizer.texts_to_sequences([sel])[0])
        mask = [1 if token in sel_seq else 0 for token in txt_seq]
        if len(mask) < max_len:
            mask = mask + [0] * (max_len - len(mask))
        else:
            mask = mask[:max_len]
        masks.append(mask)
    return np.array(masks)


train_y = texts_to_mask(train_df.text, train_df.selected_text)
val_y = texts_to_mask(val_df.text, val_df.selected_text)



## === cell 9
rev_word_index = {v: k for k, v in word_index.items()}


def decode_phrase(padded_seq, mask_seq):
    """Return a space‑separated string of tokens where mask==1."""
    indices = padded_seq[mask_seq.astype(bool)]
    words = [rev_word_index.get(i, "") for i in indices if i != 0]
    return " ".join(words).strip()




## === cell 10
def build_model():
    model = Sequential(
        [
            Embedding(vocab_size, 16, input_length=max_len),
            Dropout(0.5),
            Bidirectional(LSTM(20)),
            Dropout(0.5),
            Dense(max_len, activation="softmax"),
        ]
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(
        loss=loss_fn, optimizer=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"]
    )
    return model


positive_model = build_model()
negative_model = build_model()



## === cell 11
pos_train_mask = train_y[train_df.sentiment == "positive"]
pos_val_mask = val_y[val_df.sentiment == "positive"]
pos_train_x = train_x[train_df.sentiment == "positive"]
pos_val_x = val_x[val_df.sentiment == "positive"]

positive_history = positive_model.fit(
    pos_train_x.astype(float),
    pos_train_mask.astype(float),
    epochs=20,
    batch_size=32,
    verbose=2,
    validation_data=(pos_val_x.astype(float), pos_val_mask.astype(float)),
)



## === cell 12
neg_train_mask = train_y[train_df.sentiment == "negative"]
neg_val_mask = val_y[val_df.sentiment == "negative"]
neg_train_x = train_x[train_df.sentiment == "negative"]
neg_val_x = val_x[val_df.sentiment == "negative"]

negative_history = negative_model.fit(
    neg_train_x.astype(float),
    neg_train_mask.astype(float),
    epochs=20,
    batch_size=32,
    verbose=2,
    validation_data=(neg_val_x.astype(float), neg_val_mask.astype(float)),
)



## === cell 13
test_path = "../input/tweet-sentiment-extraction/test.csv"
test = pd.read_csv(test_path)




## === cell 14
def clean_series(series):
    cleaned = []
    for txt in series:
        if isinstance(txt, str):
            txt = txt.lower()
            txt = re.sub(r"^https?://\S+", "", txt)
            txt = txt.translate(str.maketrans("", "", string.punctuation))
            txt = txt.strip()
        cleaned.append(txt)
    return cleaned


test["text"] = clean_series(test["text"])



## === cell 15
test_x = texts_to_padded(test.text, maxlen=max_len)



## === cell 16
preds_masks = []
for idx, row in test.iterrows():
    seq = test_x[idx][np.newaxis]  # shape (1, max_len)
    if row.sentiment == "positive":
        pred = np.round(positive_model.predict(seq))
    elif row.sentiment == "negative":
        pred = np.round(negative_model.predict(seq))
    else:  # neutral – use full text later
        pred = np.zeros_like(seq)
    preds_masks.append(pred.squeeze())

preds_masks = np.array(preds_masks)



## === cell 17
selected_texts = []
for i, row in test.iterrows():
    if row.sentiment == "neutral":
        selected_texts.append(row.text)
    else:
        mask = preds_masks[i]
        phrase = decode_phrase(test_x[i], mask)
        selected_texts.append(phrase if phrase else row.text)

test["prediction"] = selected_texts



## === cell 18
submission = test[["textID", "prediction"]].rename(
    columns={"prediction": "selected_text"}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
