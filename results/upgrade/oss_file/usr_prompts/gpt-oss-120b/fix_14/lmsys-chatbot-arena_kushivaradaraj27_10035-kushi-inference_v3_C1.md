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

1.66657

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.27819) has done: 'I replace the missing tokenizer and pretrained GRU models with a lightweight end‑to‑end pipeline: build a Keras tokenizer on the training texts, train a small Embedding‑GRU‑softmax network directly on the provided training data, and use it to generate probabilities for the test set. This removes the file‑not‑found errors, ensures the predictions have three columns that sum to 1, and writes a correctly formatted `submission.csv`. The changes keep the original feature construction and evaluation logic while fixing all runtime crashes.'
- What this solution (achieved 1.27819) has done: 'The TensorFlow import was failing due to a protobuf version mismatch. Adding an environment‑variable line before importing TensorFlow forces TensorFlow to use the pure‑Python protobuf implementation, which resolves the `'MessageFactory' object has no attribute 'GetPrototype'` error. No other logic changes are needed, so the model training and submission generation remain the same, keeping the score improvement path intact.'
- What this solution (achieved 1.71046) has done: 'I added a safe import for TensorFlow – if it fails the script falls back to a scikit‑learn TF‑IDF + LogisticRegression model, guaranteeing the notebook runs without the protobuf error. I also increased the Keras training epochs slightly (to 5) to squeeze a bit more performance when TensorFlow is available. The rest of the pipeline (text concatenation, tokenisation, submission formatting) stays unchanged, so the core logic is preserved while the score should move closer to the target.'
- What this solution (achieved 1.62169) has done: 'I keep the overall pipeline unchanged but fix the TensorFlow import issue by ensuring the fallback to scikit‑learn is always used, and I improve the fallback model’s performance with a stronger TF‑IDF configuration and a LogisticRegression that uses balanced class weights and more iterations. These changes are minimal, preserve the core logic, and are expected to lower the log‑loss toward the target score.'
- What this solution (achieved 1.74789) has done: 'I keep the overall pipeline unchanged but fix the TensorFlow import fallback (already handled) and improve the scikit‑learn fallback model, which is used when TensorFlow cannot be loaded. By increasing the TF‑IDF feature space (larger max_features and a broader n‑gram range) and using a less‑regularized LogisticRegression (higher C, more iterations), the model’s predictive power is enhanced, which should lower the log‑loss from the current 1.62 toward the target ≈ 1.09 while still producing a correctly formatted `submission.csv`.'
- What this solution (achieved 1.71027) has done: 'The fix replaces the fallback LogisticRegression with a calibrated LinearSVC, which generally gives better probability estimates for text classification and therefore lowers the log‑loss. The TF‑IDF settings are also slightly expanded. No other logic changes are made, preserving the original pipeline and ensuring a correct CSV submission is written.'
- What this solution (achieved 1.65223) has done: 'I replace the fallback LinearSVC + CalibratedClassifierCV with a stronger TF‑IDF + LogisticRegression pipeline (multinomial, balanced class weights, more iterations). This keeps the overall structure unchanged while giving better calibrated probability estimates, which should lower the log‑loss toward the target. No other parts of the script are altered.'
- What this solution (achieved 1.73133) has done: 'The fix keeps the original TF‑fallback pipeline but improves the scikit‑learn path: it removes the artificial feature limit, increases LogisticRegression iterations and regularisation strength, and adds a calibrated classifier to produce better probability estimates. These changes are minimal, preserve the overall architecture, and are expected to lower the log‑loss toward the target while still writing a correct `submission.csv`.'
- What this solution (achieved 1.6762) has done: 'The fix switches the scikit‑learn fallback from a LogisticRegression‑based pipeline to a LinearSVC model calibrated with CalibratedClassifierCV (using 5‑fold CV). This change keeps the overall workflow intact while providing stronger text classification performance, which is expected to lower the log‑loss toward the target. Additional imports are added, and the TF‑IDF vectorizer is given a larger feature limit for richer representations.'
- What this solution (achieved 1.67202) has done: 'Implemented a stronger scikit‑learn fallback: switched from a calibrated LinearSVC to a well‑tuned multinomial LogisticRegression with balanced class weights and higher regularisation. This change stays within the original fallback logic, eliminates unnecessary calibration, and is expected to lower the log‑loss toward the target while still producing a correctly formatted CSV submission.'
- What this solution (achieved 1.68345) has done: 'Implemented a calibrated LinearSVC fallback (instead of LogisticRegression) with a larger TF‑IDF feature space. This keeps the original TensorFlow path untouched, fixes the previous lower‑performing scikit‑learn pipeline, and is expected to lower the log‑loss toward the target while still producing a correctly formatted CSV submission.'
- What this solution (achieved 1.71428) has done: 'I replace the scikit‑learn fallback model with a tuned multinomial LogisticRegression (balanced classes, higher C, more iterations) which directly provides calibrated probabilities, removing the unnecessary CalibratedClassifierCV wrapper. This minor change improves predictive quality and therefore lowers the log‑loss toward the target while keeping all other logic intact. I also add the required import.'
- What this solution (achieved 1.66657) has done: 'The fix adds the required scikit‑learn imports for a calibrated LinearSVC fallback, replaces the LogisticRegression fallback with a `LinearSVC` wrapped in `CalibratedClassifierCV` (using balanced class weights and higher iterations), and retains the existing TensorFlow path. This improves probability calibration while keeping the overall pipeline unchanged, leading to a lower log‑loss and a correctly formatted submission file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import warnings, random

warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)
random.seed(seed)

use_tf = True
try:
    import tensorflow as tf
    from tensorflow import keras

    print("TensorFlow imported successfully. Version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("TensorFlow import failed, switching to scikit‑learn fallback.")
    print("Error:", e)
    use_tf = False
    from sklearn.feature_extraction.text import TfidfVectorizer

    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression  # retained for compatibility



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
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>") if use_tf else None

if use_tf:
    tokenizer.fit_on_texts(train["combined_text"].tolist())
    X_train_seq = tokenizer.texts_to_sequences(train["combined_text"])
    X_test_seq = tokenizer.texts_to_sequences(test["combined_text"])
    X_train = keras.preprocessing.sequence.pad_sequences(
        X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test = keras.preprocessing.sequence.pad_sequences(
        X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
else:
    X_train = train["combined_text"].values
    X_test = test["combined_text"].values

print("Prepared X_train shape:", X_train.shape if use_tf else X_train.shape)
print("Prepared X_test  shape:", X_test.shape if use_tf else X_test.shape)



## === cell 4
y_train = train[["winner_model_a", "winner_model_b", "winner_tie"]].values.astype(
    np.float32
)
assert np.allclose(y_train.sum(axis=1), 1, atol=1e-3), "Target rows must sum to 1"

y_int = np.argmax(y_train, axis=1)

if use_tf:
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
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
else:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=300000,
        sublinear_tf=True,
        lowercase=True,
        stop_words=None,
    )
    svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
    calibrated_clf = CalibratedClassifierCV(svc, cv=5, method="sigmoid")
    model = (tfidf, calibrated_clf)



## === cell 5
if use_tf:
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=256,
        verbose=2,
    )
else:
    tfidf, clf = model
    X_train_tfidf = tfidf.fit_transform(X_train)
    clf.fit(X_train_tfidf, y_int)
    model = (tfidf, clf)



## === cell 6
if use_tf:
    avg_predictions = model.predict(X_test, batch_size=256, verbose=0)
else:
    tfidf, clf = model
    X_test_tfidf = tfidf.transform(X_test)
    avg_predictions = clf.predict_proba(X_test_tfidf)

print("Predictions shape:", avg_predictions.shape)



## === cell 7
submission = sample_submission.copy()
submission["winner_model_a"] = avg_predictions[:, 0]
submission["winner_model_b"] = avg_predictions[:, 1]
submission["winner_tie"] = avg_predictions[:, 2]

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print("Saved submission to:", output_path)
print("Submission shape:", submission.shape)



## === cell 8
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



## === cell 9
print("\n=== INFERENCE COMPLETED ===")
print("Submission file saved at:", output_path)
