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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.0934370701962233

# 6. Current score

1.27819

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.27819) has done: 'I replace the missing tokenizer and pretrained GRU models with a lightweight end‑to‑end pipeline: build a Keras tokenizer on the training texts, train a small Embedding‑GRU‑softmax network directly on the provided training data, and use it to generate probabilities for the test set. This removes the file‑not‑found errors, ensures the predictions have three columns that sum to 1, and writes a correctly formatted `submission.csv`. The changes keep the original feature construction and evaluation logic while fixing all runtime crashes.'
- What this solution (achieved 1.27819) has done: 'The TensorFlow import was failing due to a protobuf version mismatch. Adding an environment‑variable line before importing TensorFlow forces TensorFlow to use the pure‑Python protobuf implementation, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error. No other logic changes are needed, so the model training and submission generation remain the same, keeping the score improvement path intact.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings, random

warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/lmsys-chatbot-arena/train.csv"
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
sample_sub_path = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print("Train shape:", train.shape)
print("Test shape :", test.shape)




## === cell 2
def combine(row):
    return f"{row['prompt']} [SEP] {row['response_a']} [SEP] {row['response_b']}"


train["combined_text"] = train.apply(combine, axis=1)
test["combined_text"] = test.apply(combine, axis=1)




## === cell 3
MAX_LEN = 256
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train["combined_text"].tolist())

X_train_seq = tokenizer.texts_to_sequences(train["combined_text"])
X_test_seq = tokenizer.texts_to_sequences(test["combined_text"])

X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)




## === cell 4
y_train = train[["winner_model_a", "winner_model_b", "winner_tie"]].values.astype(
    np.float32
)

assert np.allclose(y_train.sum(axis=1), 1, atol=1e-3), "Target rows must sum to 1"




## === cell 5
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 64

model = keras.Sequential(
    [
        keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, input_length=MAX_LEN
        ),
        keras.layers.GRU(64, return_sequences=False),
        keras.layers.Dense(3, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()




## === cell 6
model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=256, verbose=2)




## === cell 7
avg_predictions = model.predict(X_test, batch_size=256, verbose=0)
print("Predictions shape:", avg_predictions.shape)




## === cell 8
submission = sample_submission.copy()
submission["winner_model_a"] = avg_predictions[:, 0]
submission["winner_model_b"] = avg_predictions[:, 1]
submission["winner_tie"] = avg_predictions[:, 2]

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print("Saved submission to:", output_path)
print("Submission shape:", submission.shape)




## === cell 9
print("\nPrediction statistics per class:")
for i, name in enumerate(["winner_model_a", "winner_model_b", "winner_tie"]):
    col = avg_predictions[:, i]
    print(
        f"{name}: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}"
    )

prob_sums = avg_predictions.sum(axis=1)
print(
    "\nProbability sums -> min:",
    prob_sums.min(),
    "max:",
    prob_sums.max(),
    "mean:",
    prob_sums.mean(),
)




## === cell 10
print("\n=== INFERENCE COMPLETED ===")
print("Submission file saved at:", output_path)
