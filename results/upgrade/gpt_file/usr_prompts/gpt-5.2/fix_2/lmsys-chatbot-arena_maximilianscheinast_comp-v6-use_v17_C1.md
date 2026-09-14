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

3.12

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

1.1029188742467997

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/lmsys-chatbot-arena"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = (
    "/kaggle/input/chat_predict_v2/tensorflow2/v4_keras_use/1/class_v4_am(9).keras"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_data = pd.read_csv(TEST_PATH)
test_data["combined_response"] = (
    test_data["response_a"].fillna("").astype(str)
    + " "
    + test_data["response_b"].fillna("").astype(str)
)


def hashing_embed_texts(texts, dim=512, ngram_range=(1, 2), max_tokens=2000):
    """
    Deterministic, lightweight text embedding:
    - tokenizes on whitespace
    - uses hashed n-grams into a fixed-dim vector
    - L2 normalizes each row
    """
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = (t or "").lower()
        toks = t.split()
        if max_tokens is not None and len(toks) > max_tokens:
            toks = toks[:max_tokens]

        for tok in toks:
            h = hash(tok) % dim
            X[i, h] += 1.0

        if ngram_range[1] >= 2:
            for j in range(len(toks) - 1):
                bg = toks[j] + "_" + toks[j + 1]
                h = hash(bg) % dim
                X[i, h] += 1.0

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X /= norms
    return X


embeddings = hashing_embed_texts(test_data["combined_response"].tolist(), dim=512)
print("Embedding shape:", embeddings.shape)



## === cell 2
model = load_model(MODEL_PATH, compile=False)
model.summary()

in_shape = model.inputs[0].shape
expected_dim = None
if len(in_shape) == 2:
    expected_dim = int(in_shape[1])
elif len(in_shape) == 3:
    expected_dim = int(in_shape[2])
else:
    raise ValueError(f"Unsupported model input shape: {in_shape}")

X = embeddings
if X.shape[1] != expected_dim:
    if X.shape[1] > expected_dim:
        X = X[:, :expected_dim]
    else:
        pad = np.zeros((X.shape[0], expected_dim - X.shape[1]), dtype=X.dtype)
        X = np.concatenate([X, pad], axis=1)

if len(in_shape) == 3:
    X = np.expand_dims(X, axis=1)

predictions = model.predict(X, batch_size=256, verbose=1)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3287806618.py in <cell line: 0>()
      2 # NOTE: If the model expects a different embedding width, we adapt by simple truncation/padding to match
      3 # the input layer shape (this preserves the same pipeline and prevents shape runtime errors).
----> 4 model = load_model(MODEL_PATH, compile=False)
      5 model.summary()
      6 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/input/chat_predict_v2/tensorflow2/v4_keras_use/1/class_v4_am(9).keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 3
predictions = np.asarray(predictions, dtype=np.float64)
predictions = np.clip(predictions, 0.0, np.inf)
row_sums = predictions.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
predictions = (predictions / row_sums).astype(np.float32)

submission = pd.DataFrame(
    {
        "id": test_data["id"].values,
        "winner_model_a": predictions[:, 0],
        "winner_model_b": predictions[:, 1],
        "winner_tie": predictions[:, 2],
    }
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/484351885.py in <cell line: 0>()
      2 # - clamp to >= 0
      3 # - row-normalize to sum to 1
----> 4 predictions = np.asarray(predictions, dtype=np.float64)
      5 predictions = np.clip(predictions, 0.0, np.inf)
      6 row_sums = predictions.sum(axis=1, keepdims=True)

NameError: name 'predictions' is not defined
