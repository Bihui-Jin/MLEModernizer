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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

1.08084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
BASE_PATH = "/kaggle/input/lmsys-chatbot-arena"




## === cell 1
class CFG:
    seed = 42  # Random seed
    label2name = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
    name2label = {v: k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())




## === cell 2
import numpy as np
import pandas as pd


def _parse_first_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    import json, ast

    s_json = s.replace("null", '""')
    try:
        obj = json.loads(s_json)
    except Exception:
        try:
            obj = ast.literal_eval(s_json)
        except Exception:
            return s  # last resort: raw string
    if isinstance(obj, list) and len(obj) > 0:
        return "" if obj[0] is None else str(obj[0])
    return "" if obj is None else str(obj)


train_df = pd.read_csv(f"{BASE_PATH}/train.csv")

train_df["prompt"] = train_df["prompt"].map(_parse_first_text)
train_df["response_a"] = train_df["response_a"].map(_parse_first_text)
train_df["response_b"] = train_df["response_b"].map(_parse_first_text)

train_df["class_name"] = train_df[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].idxmax(axis=1)
train_df["class_label"] = train_df["class_name"].map(CFG.name2label).astype(int)



## === cell 3
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def count_stop_words(text):
    words = str(text).lower().split()
    return sum(1 for word in words if word in ENGLISH_STOP_WORDS)


def count_chars(text):
    return len(str(text))


def count_words(text):
    return len(str(text).split())


def count_lines(text):
    text = str(text)
    return text.count("\n") + 1


def is_markdown(text):
    text = str(text)
    return bool(re.search(r"(#+|\*\*|\[.*\]\(.*\)|`.*`)", text))


def is_html(text):
    text = str(text)
    return bool(re.search(r"(<[^>]+>)", text))


def count_bullet_points(text):
    text = str(text)
    return text.count("- ") + text.count("* ") + text.count("+ ")


def count_headlines(text):
    text = str(text)
    return text.count("# ")


def count_bold_tokens(text):
    text = str(text)
    return text.count("**") // 2




## === cell 4
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

text_feature_extraction_pipeline = Pipeline(
    [
        (
            "hashing_vectorizer",
            HashingVectorizer(stop_words="english", n_features=50_000),
        ),
        ("tfidf_transformer", TfidfTransformer()),
        ("svd", TruncatedSVD(n_components=50, random_state=CFG.seed)),
        ("normalizer", Normalizer(copy=False)),
    ]
)

text_feature_extraction_pipeline.fit(
    train_df["prompt"] + train_df["response_a"] + train_df["response_b"]
)


def calculate_percentage_difference(value_a, value_b):
    return abs(value_a - value_b) / ((value_a + value_b + 1) / 2) * 100


def avg_line_length_chars(text):
    lines = str(text).splitlines()
    return sum(len(line) for line in lines) / len(lines) if lines else 0.0


def avg_line_length_words(text):
    lines = str(text).splitlines()
    return sum(len(line.split()) for line in lines) / len(lines) if lines else 0.0


def avg_line_length_non_stopwords(text):
    lines = str(text).splitlines()
    non_stopword_counts = [
        sum(1 for word in line.lower().split() if word not in ENGLISH_STOP_WORDS)
        for line in lines
    ]
    return sum(non_stopword_counts) / len(lines) if lines else 0.0




## === cell 5
from sklearn.cluster import KMeans


def cluster_feature(df, text_vectors, output_field, k=64):
    kmeans = KMeans(n_clusters=k, random_state=CFG.seed, n_init=10).fit(text_vectors)
    df[output_field] = kmeans.labels_
    return df, kmeans


def predict_cluster(df, kmeans_model, x, output_field):
    df[output_field] = kmeans_model.predict(x)
    return df




## === cell 6
def calculate_fieldwise_features(df, field):
    df[f"{field}_char_length"] = df[field].apply(count_chars)
    df[f"{field}_word_count"] = df[field].apply(count_words)
    df[f"{field}_line_count"] = df[field].apply(count_lines)
    df[f"{field}_stop_word_count"] = df[field].apply(count_stop_words)
    df[f"{field}_is_markdown"] = df[field].apply(is_markdown)
    df[f"{field}_is_html"] = df[field].apply(is_html)
    df[f"{field}_bullet_point_count"] = df[field].apply(count_bullet_points)
    df[f"{field}_headline_count"] = df[field].apply(count_headlines)
    df[f"{field}_bold_token_count"] = df[field].apply(count_bold_tokens)

    df[f"{field}_avg_line_length_chars"] = df[field].apply(avg_line_length_chars)
    df[f"{field}_avg_line_length_words"] = df[field].apply(avg_line_length_words)
    df[f"{field}_avg_line_length_non_stopwords"] = df[field].apply(
        avg_line_length_non_stopwords
    )
    return df


def diff_features(df, field1, field2):
    df["diff_length_ab"] = df[f"{field1}_char_length"] - df[f"{field2}_char_length"]
    df["diff_words_ab"] = df[f"{field1}_word_count"] - df[f"{field2}_word_count"]
    df["diff_lines_ab"] = df[f"{field1}_line_count"] - df[f"{field2}_line_count"]
    df["diff_stop_words_ab"] = (
        df[f"{field1}_stop_word_count"] - df[f"{field2}_stop_word_count"]
    )

    df["both_markdown_ab"] = df[f"{field1}_is_markdown"] & df[f"{field2}_is_markdown"]
    df["both_html_ab"] = df[f"{field1}_is_html"] & df[f"{field2}_is_html"]
    df["diff_bullet_points_ab"] = (
        df[f"{field1}_bullet_point_count"] - df[f"{field2}_bullet_point_count"]
    )
    df["diff_headlines_ab"] = (
        df[f"{field1}_headline_count"] - df[f"{field2}_headline_count"]
    )
    df["diff_bold_tokens_ab"] = (
        df[f"{field1}_bold_token_count"] - df[f"{field2}_bold_token_count"]
    )

    df["diff_percentage_length_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_char_length"], row[f"{field2}_char_length"]
        ),
        axis=1,
    )
    df["diff_percentage_words_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_word_count"], row[f"{field2}_word_count"]
        ),
        axis=1,
    )
    df["diff_percentage_lines_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_line_count"], row[f"{field2}_line_count"]
        ),
        axis=1,
    )
    df["diff_percentage_stop_words_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_stop_word_count"], row[f"{field2}_stop_word_count"]
        ),
        axis=1,
    )
    df["diff_percentage_bullet_points_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_bullet_point_count"], row[f"{field2}_bullet_point_count"]
        ),
        axis=1,
    )
    df["diff_percentage_headlines_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_headline_count"], row[f"{field2}_headline_count"]
        ),
        axis=1,
    )
    df["diff_percentage_bold_tokens_ab"] = df.apply(
        lambda row: calculate_percentage_difference(
            row[f"{field1}_bold_token_count"], row[f"{field2}_bold_token_count"]
        ),
        axis=1,
    )

    df["diff_avg_line_length_chars_ab"] = (
        df[f"{field1}_avg_line_length_chars"] - df[f"{field2}_avg_line_length_chars"]
    )
    df["diff_avg_line_length_words_ab"] = (
        df[f"{field1}_avg_line_length_words"] - df[f"{field2}_avg_line_length_words"]
    )
    df["diff_avg_line_length_non_stopwords_ab"] = (
        df[f"{field1}_avg_line_length_non_stopwords"]
        - df[f"{field2}_avg_line_length_non_stopwords"]
    )
    return df


def ratio_features(df):
    pw = df["prompt_word_count"].replace(0, 1)
    pl = df["prompt_line_count"].replace(0, 1)
    pc = df["prompt_char_length"].replace(0, 1)

    df["ratio_words_a"] = df["response_a_word_count"] / pw
    df["ratio_words_b"] = df["response_b_word_count"] / pw

    df["ratio_lines_a"] = df["response_a_line_count"] / pl
    df["ratio_lines_b"] = df["response_b_line_count"] / pl

    df["ratio_len_a"] = df["response_a_char_length"] / pc
    df["ratio_len_b"] = df["response_b_char_length"] / pc
    return df


def compute_text_vectors(df, field):
    return text_feature_extraction_pipeline.transform(df[field])


def text_similarity(df, text1, text2, output_field):
    df[output_field] = np.asarray(text1.multiply(text2).sum(axis=1)).ravel()
    return df


def add_text_vectors_to_df(df, text_vectors, prefix):
    arr = (
        text_vectors.toarray()
        if hasattr(text_vectors, "toarray")
        else np.asarray(text_vectors)
    )
    vector_df = pd.DataFrame(
        arr, columns=[f"{prefix}_vector_{i}" for i in range(arr.shape[1])]
    )
    return pd.concat(
        [df.reset_index(drop=True), vector_df.reset_index(drop=True)], axis=1
    )


def generate_features(
    df,
    kmeans_prompt=None,
    kmeans_response_a=None,
    kmeans_response_b=None,
    train_flag=True,
):
    df = df.copy()
    df = calculate_fieldwise_features(df, "prompt")
    df = calculate_fieldwise_features(df, "response_a")
    df = calculate_fieldwise_features(df, "response_b")

    df = diff_features(df, "response_a", "response_b")
    df = ratio_features(df)

    prompt_vectors = compute_text_vectors(df, "prompt")
    response_a_vectors = compute_text_vectors(df, "response_a")
    response_b_vectors = compute_text_vectors(df, "response_b")

    df = add_text_vectors_to_df(df, prompt_vectors, "prompt")
    df = add_text_vectors_to_df(df, response_a_vectors, "response_a")
    df = add_text_vectors_to_df(df, response_b_vectors, "response_b")

    df = text_similarity(df, prompt_vectors, response_a_vectors, "sim_a")
    df = text_similarity(df, prompt_vectors, response_b_vectors, "sim_b")
    df = text_similarity(df, response_a_vectors, response_b_vectors, "sim_ab")

    if train_flag:
        df, kmeans_prompt = cluster_feature(df, prompt_vectors, "cluster_prompt")
        df, kmeans_response_a = cluster_feature(df, response_a_vectors, "cluster_a")
        df, kmeans_response_b = cluster_feature(df, response_b_vectors, "cluster_b")
    else:
        df = predict_cluster(df, kmeans_prompt, prompt_vectors, "cluster_prompt")
        df = predict_cluster(df, kmeans_response_a, response_a_vectors, "cluster_a")
        df = predict_cluster(df, kmeans_response_b, response_b_vectors, "cluster_b")

    df["cluster_prompt_cat"] = (
        df["cluster_prompt"].astype("category").cat.as_ordered().cat.codes + 1
    )
    df["cluster_a_cat"] = (
        df["cluster_a"].astype("category").cat.as_ordered().cat.codes + 1
    )
    df["cluster_b_cat"] = (
        df["cluster_b"].astype("category").cat.as_ordered().cat.codes + 1
    )

    return df, kmeans_prompt, kmeans_response_a, kmeans_response_b




## === cell 7
train_df, kmeans_prompt, kmeans_response_a, kmeans_response_b = generate_features(
    train_df
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1168752657.py in <cell line: 0>()
----> 1 train_df, kmeans_prompt, kmeans_response_a, kmeans_response_b = generate_features(
      2     train_df
      3 )
      4 

/tmp/ipykernel_11/2023644019.py in generate_features(df, kmeans_prompt, kmeans_response_a, kmeans_response_b, train_flag)
    161     df = add_text_vectors_to_df(df, response_b_vectors, "response_b")
    162 
--> 163     df = text_similarity(df, prompt_vectors, response_a_vectors, "sim_a")
    164     df = text_similarity(df, prompt_vectors, response_b_vectors, "sim_b")
    165     df = text_similarity(df, response_a_vectors, response_b_vectors, "sim_ab")

/tmp/ipykernel_11/2023644019.py in text_similarity(df, text1, text2, output_field)
    119     # Change (speed, same semantics): vectorized cosine since vectors are L2-normalized by the pipeline.
    120     # cosine(u,v) = dot(u,v) when ||u||=||v||=1
--> 121     df[output_field] = np.asarray(text1.multiply(text2).sum(axis=1)).ravel()
    122     return df
    123 

AttributeError: 'numpy.ndarray' object has no attribute 'multiply'

## === cell 8
label = ["class_label"]


def get_feature_list(df):
    exclude_cols = [
        "id",
        "prompt",
        "response_a",
        "response_b",
        "model_a",
        "model_b",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
        "class_name",
        "class_label",
        "cluster_prompt",
        "cluster_a",
        "cluster_b",
        "model_a_cat",
        "model_b_cat",
    ]
    return [col for col in df.columns if col not in exclude_cols]


feature_list = get_feature_list(train_df)
training_features = train_df[feature_list]
training_labels = train_df[label].values.ravel().astype(int)



## === cell 9
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    training_features,
    training_labels,
    test_size=0.2,
    random_state=CFG.seed,
    stratify=training_labels,
)



## === cell 10
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [200],
    "max_depth": [3],
    "learning_rate": [0.2],
}

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    random_state=CFG.seed,
    tree_method="hist",
    eval_metric="mlogloss",
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,  # keep runtime within limits
    scoring="neg_log_loss",
    verbose=2,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Neg Log Loss:", grid_search.best_score_)

best_xgb_model = grid_search.best_estimator_



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2740605279.py in <cell line: 0>()
     25     n_jobs=-1,
     26 )
---> 27 grid_search.fit(X_train, y_train)
     28 
     29 print("Best Parameters:", grid_search.best_params_)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
   1386     def _run_search(self, evaluate_candidates):
   1387         """Search all candidates in param_grid"""
-> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
   1389 
   1390 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
    849                     )
    850 
--> 851                 _warn_or_raise_about_fit_failures(out, self.error_score)
    852 
    853                 # For callable self.scoring, the return type is only know after

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
    365                 f"Below are more details about the failures:\n{fit_errors_summary}"
    366             )
--> 367             raise ValueError(all_fits_failed_message)
    368 
    369         else:

ValueError: 
All the 3 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/xgboost/core.py", line 730, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py", line 1500, in fit
    train_dmatrix, evals = _wrap_evaluation_matrices(
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py", line 521, in _wrap_evaluation_matrices
    train_dmatrix = create_dmatrix(
                    ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py", line 958, in _create_dmatrix
    return QuantileDMatrix(
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/xgboost/core.py", line 730, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/xgboost/core.py", line 1529, in __init__
    self._init(
  File "/usr/local/lib/python3.11/dist-packages/xgboost/core.py", line 1590, in _init
    _check_call(ret)
  File "/usr/local/lib/python3.11/dist-packages/xgboost/core.py", line 282, in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
xgboost.core.XGBoostError: [04:17:04] /workspace/src/data/iterative_dmatrix.cc:202: Check failed: n_features >= 1 (0 vs. 1) : Data must has at least 1 column.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fffbb1fdfba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f59b7) [0x7fffbb2039b7]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f8858) [0x7fffbb206858]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3a2a07) [0x7fffbb1b0a07]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGQuantileDMatrixCreateFromCallback+0x2b0) [0x7fffbaf73c40]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff753ee2e]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff753b493]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff751a4d8]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff7519c8e]




## === cell 11
from xgboost import plot_importance
from matplotlib import pyplot

ax = plot_importance(best_xgb_model)
ax.figure.set_size_inches(20, 30)
pyplot.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4237294989.py in <cell line: 0>()
      2 from matplotlib import pyplot
      3 
----> 4 ax = plot_importance(best_xgb_model)
      5 ax.figure.set_size_inches(20, 30)
      6 pyplot.show()

NameError: name 'best_xgb_model' is not defined

## === cell 12
from sklearn.metrics import classification_report, log_loss

y_pred = best_xgb_model.predict(X_val)
print(classification_report(y_val, y_pred, target_names=CFG.class_names))

y_pred_proba = best_xgb_model.predict_proba(X_val)
nll = log_loss(y_val, y_pred_proba)
print("Validation Log Loss:", nll)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/806022910.py in <cell line: 0>()
      1 from sklearn.metrics import classification_report, log_loss
      2 
----> 3 y_pred = best_xgb_model.predict(X_val)
      4 print(classification_report(y_val, y_pred, target_names=CFG.class_names))
      5 

NameError: name 'best_xgb_model' is not defined

## === cell 13
test_df = pd.read_csv(f"{BASE_PATH}/test.csv")
test_df["prompt"] = test_df["prompt"].map(_parse_first_text)
test_df["response_a"] = test_df["response_a"].map(_parse_first_text)
test_df["response_b"] = test_df["response_b"].map(_parse_first_text)

test_df, _, _, _ = generate_features(
    test_df, kmeans_prompt, kmeans_response_a, kmeans_response_b, train_flag=False
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/7260345.py in <cell line: 0>()
      5 
      6 test_df, _, _, _ = generate_features(
----> 7     test_df, kmeans_prompt, kmeans_response_a, kmeans_response_b, train_flag=False
      8 )
      9 

NameError: name 'kmeans_prompt' is not defined

## === cell 14
feature_list = get_feature_list(train_df)
test_features = test_df[feature_list]

proba = best_xgb_model.predict_proba(test_features)
alpha = 0.5
uniform = np.full_like(proba, 1.0 / proba.shape[1])
test_predictions = (1 - alpha) * proba + alpha * uniform
test_predictions = test_predictions / test_predictions.sum(axis=1, keepdims=True)

sub_df = test_df[["id"]].copy()
sub_df[CFG.class_names] = test_predictions
sub_df.to_csv("/kaggle/working/submission.csv", index=False)

print(sub_df.head())
print("Wrote /kaggle/working/submission.csv with shape:", sub_df.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1873976417.py in <cell line: 0>()
      3 
      4 # Change (logloss stability): smooth with uniform prior and renormalize to valid probabilities.
----> 5 proba = best_xgb_model.predict_proba(test_features)
      6 alpha = 0.5
      7 uniform = np.full_like(proba, 1.0 / proba.shape[1])

NameError: name 'best_xgb_model' is not defined
