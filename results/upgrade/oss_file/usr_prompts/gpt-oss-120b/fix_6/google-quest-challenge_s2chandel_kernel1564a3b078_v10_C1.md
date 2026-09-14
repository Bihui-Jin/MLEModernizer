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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
scipy==1.15.3
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.0095

# 6. Current score

0.09513

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09576) has done: 'I added a proper preprocessing step that encodes **all** non‑target columns (including the text fields) on the combined train + test data so no unseen labels appear at inference time, then scales every feature to [0, 1]. The buggy directory walk and the original encoder that left raw strings in the feature matrix have been removed. All later cells now use the new numeric feature matrices, allowing the model to train, evaluate and generate a valid `submission.csv` file.'
- What this solution (achieved 0.09316) has done: 'The import of the standalone `keras` package clashes with the TensorFlow version in this environment, causing an AttributeError during module loading. Switching to `tensorflow.keras` resolves the protobuf incompatibility while keeping the same model definition and behavior. No other logic changes are needed, so the pipeline run end‑to‑end and produce a valid `submission.csv` with a score still well above the target.'
- What this solution (achieved 0.09798) has done: 'I add a protobuf compatibility fix by setting the environment variable before any TensorFlow imports, and ensure predictions stay within the required [0, 1] range. No other logic changes are needed, and the model already scores well above the target.'
- What this solution (achieved 0.09513) has done: 'Implemented a safe import order fix: set the protobuf environment variable first, then import TensorFlow before any Keras sub‑modules to eliminate the `MessageFactory` attribute error. The rest of the pipeline (preprocessing, scaling, model training, prediction, and submission creation) is unchanged, preserving the existing high score that already exceeds the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf  # Must be imported before any tf.keras sub‑modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import matplotlib.pyplot as plt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/google-quest-challenge/train.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"
data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

target_cols = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]


def preprocess_features(train_df, test_df, target_columns):
    """Encode all non‑target columns on the combined train+test data and return
    numeric train and test feature DataFrames along with the fitted encoders."""
    feature_cols = [c for c in train_df.columns if c not in target_columns + ["qa_id"]]
    combined = pd.concat(
        [train_df[feature_cols], test_df[feature_cols]], ignore_index=True
    )
    encoders = {}
    for col in feature_cols:
        if combined[col].dtype == object:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
            encoders[col] = le
        else:
            combined[col] = combined[col]
    train_enc = combined.iloc[: len(train_df)].reset_index(drop=True)
    test_enc = combined.iloc[len(train_df) :].reset_index(drop=True)
    return train_enc, test_enc, encoders


x_enc, test_enc, encoders = preprocess_features(data, test_data, target_cols)




## === cell 2
def scale_features(train_feat, test_feat):
    """Scale all feature columns to [0,1] using MinMaxScaler fitted on the train set."""
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feat), columns=train_feat.columns
    )
    test_scaled = pd.DataFrame(scaler.transform(test_feat), columns=test_feat.columns)
    return train_scaled, test_scaled, scaler


x, test_scaled, scaler = scale_features(x_enc, test_enc)

print("Feature shape (train):", x.shape)
print("Feature shape (test):", test_scaled.shape)



## === cell 3
y = data[target_cols]



## === cell 4
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)




## === cell 5
def build_model(input_dim):
    """Simple feed‑forward network for regression on 30 targets."""
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(30, activation="sigmoid"))  # outputs in [0,1]
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    return model


model = build_model(input_dim=x_train.shape[1])



## === cell 6
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_data=(x_val, y_val),
    verbose=1,
)



## === cell 7
val_loss, val_mse = model.evaluate(x_val, y_val, verbose=0)
print(f"Validation MSE: {val_mse:.4f}")


def plot_history(hist):
    plt.style.use("ggplot")
    epochs = range(1, len(hist.history["mse"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.history["mse"], "b", label="Train MSE")
    plt.plot(epochs, hist.history["val_mse"], "r", label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE over epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.history["loss"], "b", label="Train loss")
    plt.plot(epochs, hist.history["val_loss"], "r", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()

    plt.show()


plot_history(history)



## === cell 8
y_pred_val = model.predict(x_val)
spearman_val = spearmanr(y_val.values.ravel(), y_pred_val.ravel()).correlation
print(f"Validation Spearman (approx.): {spearman_val:.4f}")



## === cell 9
test_pred = model.predict(test_scaled)
test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 10
submission = pd.DataFrame(
    test_pred,
    columns=target_cols,
)
submission.insert(0, "qa_id", test_data["qa_id"].values)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
