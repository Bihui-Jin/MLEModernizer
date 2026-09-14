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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
data.head()




## === cell 2
def encoder(df):
    """
    Encode categorical string columns with LabelEncoder.
    Returns the transformed DataFrame and a dict of fitted encoders.
    """
    encoders = {}
    cat_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders




## === cell 3
def scale(df):
    """
    Scale the numeric/categorical columns to [0,1] with MinMaxScaler.
    Returns the scaled DataFrame and the fitted scaler.
    """
    scaler = MinMaxScaler()
    cols_to_scale = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    df_scaled = df_scaled.drop(columns="qa_id")
    return df_scaled, scaler




## === cell 4
x_enc, encoders = encoder(data)
x, scaler = scale(x_enc)

print("Feature shape:", x.shape)




## === cell 5
y = data[
    [
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
]




## === cell 6
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)




## === cell 7
def build_model(input_dim):
    """
    Simple feed‑forward network for regression on 30 targets.
    """
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(30, activation="sigmoid"))  # output in [0,1]
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    return model


model = build_model(input_dim=x_train.shape[1])




## === cell 8
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_data=(x_val, y_val),
    verbose=1,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1127353901.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x_train,
      3     y_train,
      4     epochs=10,
      5     batch_size=256,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    242             new_values = values
    243         else:
--> 244             raise
    245 
    246     return new_values

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: "I'm using dash(sh) or bash or zsh. If possible, I would prefer to put in common place.\n\nI want to put proper PS1 settings when I interactive with shell, so these situation should be considered\n\n\nlogin (show PS1)\nsu\nsudo\nscript (Don't show PS1)\n\n\nI was put PS1 on .bashrc, but it seems not always workable.\n"

## === cell 9
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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2450564067.py in <cell line: 0>()
      1 # Evaluate on validation set
----> 2 val_loss, val_mse = model.evaluate(x_val, y_val, verbose=0)
      3 print(f"Validation MSE: {val_mse:.4f}")
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    242             new_values = values
    243         else:
--> 244             raise
    245 
    246     return new_values

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: "I have the differential equation\n\n$$y''y+n(y')^2=0\\tag 1$$\n\nI haven't found any references to equations of this type by searching.  Does anyone have a suggestion as to how to proceed?\n\nIt comes about from the following progression:\n\n$$y=(y')^n-\\frac 1n\\tag 2\\\\\ny'=ny''(y')^{n-1}\\implies1=ny''(y')^{n-2}\\\\\n0=n(n-2)(y'')^2(y')^{n-3}+ny'''(y')^{n-2}\\\\\n0=(n-2)(y'')^2+y'''y'$$\n\nTaking $y$ in place of $y'$ and $n$ in place of $n-2$ I get the form shown in $(1)$.  Are there any good ways to solve any of the other intermediate forms, or even the initial form shown in $(2)$?\n\nFurther constraints:\n\n$y(0)\\approx 1.27$ is an otherwise-unknown constant that I am attempting to build an expression for.\n\nThe initial problem is the recurrence relation with $a_1\\approx 1.27$ the unknown constant and\n\n$$a_{n+1}=(a_n)^{n+1}-\\frac 1{n+1}$$\n\nwith the additional constraint $\\lim_{n\\to\\infty}a_n=1$.\n"

## === cell 10
y_pred_val = model.predict(x_val)
spearman_val = spearmanr(y_val.values.ravel(), y_pred_val.ravel()).correlation
print(f"Validation Spearman (approx.): {spearman_val:.4f}")




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2903450462.py in <cell line: 0>()
      1 # Predict on validation set and compute Spearman correlation (approximate score)
----> 2 y_pred_val = model.predict(x_val)
      3 spearman_val = spearmanr(y_val.values.ravel(), y_pred_val.ravel()).correlation
      4 print(f"Validation Spearman (approx.): {spearman_val:.4f}")
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/optree/ops.py in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764     leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765     flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766     return treespec.unflatten(map(func, *flat_args))
    767 
    768 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    242             new_values = values
    243         else:
--> 244             raise
    245 
    246     return new_values

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: "I have the differential equation\n\n$$y''y+n(y')^2=0\\tag 1$$\n\nI haven't found any references to equations of this type by searching.  Does anyone have a suggestion as to how to proceed?\n\nIt comes about from the following progression:\n\n$$y=(y')^n-\\frac 1n\\tag 2\\\\\ny'=ny''(y')^{n-1}\\implies1=ny''(y')^{n-2}\\\\\n0=n(n-2)(y'')^2(y')^{n-3}+ny'''(y')^{n-2}\\\\\n0=(n-2)(y'')^2+y'''y'$$\n\nTaking $y$ in place of $y'$ and $n$ in place of $n-2$ I get the form shown in $(1)$.  Are there any good ways to solve any of the other intermediate forms, or even the initial form shown in $(2)$?\n\nFurther constraints:\n\n$y(0)\\approx 1.27$ is an otherwise-unknown constant that I am attempting to build an expression for.\n\nThe initial problem is the recurrence relation with $a_1\\approx 1.27$ the unknown constant and\n\n$$a_{n+1}=(a_n)^{n+1}-\\frac 1{n+1}$$\n\nwith the additional constraint $\\lim_{n\\to\\infty}a_n=1$.\n"

## === cell 11
test_data = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
test_ids = test_data["qa_id"]

test_enc = test_data.copy()
for col, le in encoders.items():
    test_enc[col] = le.transform(test_enc[col].astype(str))

test_scaled = test_enc.copy()
cols_to_scale = [
    "question_title",
    "question_user_name",
    "question_user_page",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]
test_scaled[cols_to_scale] = scaler.transform(test_scaled[cols_to_scale])
test_scaled = test_scaled.drop(columns="qa_id")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    223         try:
--> 224             return _map_to_integer(values, uniques)
    225         except KeyError as e:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _map_to_integer(values, uniques)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in <listcomp>(.0)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in __missing__(self, key)
    157             return self.nan_value
--> 158         raise KeyError(key)
    159 

KeyError: 'What is a word for somebody who lies to themselves'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3658883613.py in <cell line: 0>()
      6 test_enc = test_data.copy()
      7 for col, le in encoders.items():
----> 8     test_enc[col] = le.transform(test_enc[col].astype(str))
      9 
     10 # Scale using the fitted scaler

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in transform(self, y)
    137             return np.array([])
    138 
--> 139         return _encode(y, uniques=self.classes_)
    140 
    141     def inverse_transform(self, y):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    224             return _map_to_integer(values, uniques)
    225         except KeyError as e:
--> 226             raise ValueError(f"y contains previously unseen labels: {str(e)}")
    227     else:
    228         if check_unknown:

ValueError: y contains previously unseen labels: 'What is a word for somebody who lies to themselves'

## === cell 12
test_pred = model.predict(test_scaled)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/188458801.py in <cell line: 0>()
      1 # Predict on test set
----> 2 test_pred = model.predict(test_scaled)
      3 
      4 

NameError: name 'test_scaled' is not defined

## === cell 13
submission = pd.DataFrame(
    test_pred,
    columns=[
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
    ],
)
submission.insert(0, "qa_id", test_ids.values)

submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2584993077.py in <cell line: 0>()
      1 # Build submission DataFrame with correct column order
      2 submission = pd.DataFrame(
----> 3     test_pred,
      4     columns=[
      5         "question_asker_intent_understanding",

NameError: name 'test_pred' is not defined
