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

0.51636

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.51636) has done: 'I remove the problematic TensorFlow Hub import and replace the hub embedding with a lightweight TextVectorization + Embedding pipeline that works with the current library versions. This fixes the import and layer‑call errors, restores the model definition, and allows the script to train and generate a valid CSV submission. The rest of the workflow (data loading, splitting, training loop, and CSV creation) stays unchanged.'

# 9. Code solution

## === cell 0
import os, random
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

seed = 197
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF version:", tf.__version__)
print("Contents of ../input:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/train.tsv"
test_path = "../input/test.tsv"
train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print(train_df.head())
print(test_df.head())




## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=seed, stratify=train_df["Sentiment"]
)

train_text = train_df["Phrase"].astype(str).values
train_labels = train_df["Sentiment"].values
val_text = val_df["Phrase"].astype(str).values
val_labels = val_df["Sentiment"].values




## === cell 3
max_tokens = 20000  # size of vocabulary
output_seq_len = 20  # truncate / pad sequences
embedding_dim = 50  # match original hub output dimension

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=output_seq_len,
    standardize="lower_and_strip_punctuation",
)

vectorizer.adapt(train_text)

inputs = tf.keras.Input(shape=(), dtype=tf.string, name="Phrase")
x = vectorizer(inputs)  # (batch, output_seq_len)
x = tf.keras.layers.Embedding(
    input_dim=max_tokens, output_dim=embedding_dim, name="embedding"
)(
    x
)  # (batch, seq, dim)
x = tf.keras.layers.GlobalAveragePooling1D()(x)  # (batch, dim)
x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
batch_size = 128
epochs = 5  # modest number to stay within time limits
model.fit(
    x=train_text,
    y=train_labels,
    validation_data=(val_text, val_labels),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
)




## === cell 5
val_loss, val_acc = model.evaluate(x=val_text, y=val_labels, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")




## === cell 6
test_text = test_df["Phrase"].astype(str).values
pred_probs = model.predict(test_text, batch_size=batch_size, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)

sub = pd.read_csv("../input/sampleSubmission.csv")
sub["Sentiment"] = pred_classes
sub.to_csv("sub_tfhub.csv", index=False)
print("Submission saved to sub_tfhub.csv")
