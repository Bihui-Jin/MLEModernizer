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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7805410791891036

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43069) has done: 'I fix the runtime error by configuring the Ridge regression to use a solver that works with the current SciPy version (e.g., `solver='lsqr'`). This prevents the internal call to `scipy.sparse.linalg.cg` that fails due to a changed signature. No other logic is altered, so the model, feature engineering, and submission format remain unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.59003) has done: 'I lower the Ridge regularization strength (alpha = 0.1) and add a character‑level TF‑IDF alongside the word‑level TF‑IDF using a FeatureUnion, which usually captures extra similarity cues and improves Pearson correlation. I also drop the rounding step – keeping the raw predictions (only clipping to [0, 1]) preserves more variance, which is beneficial for the Pearson metric while still respecting the required score range. These minimal changes keep the overall pipeline structure intact.'
- What this solution (achieved 0.30112) has done: 'I slightly strengthen the text features (broader n‑gram ranges and a few more TF‑IDF terms) and reduce the Ridge regularisation (α = 0.01). I also insert a `StandardScaler(with_mean=False)` right after the combined TF‑IDF so the sparse features are scaled before regression, which often improves Pearson correlation without changing the overall pipeline logic. These minimal tweaks are expected to move the validation Pearson closer to the target score.'
- What this solution (achieved 0.60471) has done: 'I raise the Ridge regularisation strength back to `alpha=0.1` (the setting that previously yielded a much higher Pearson) and remove the unnecessary `StandardScaler` on the sparse TF‑IDF matrix, which can hurt linear models. These tiny adjustments keep the original pipeline structure while expectedly moving the validation Pearson much closer to the target score.'
- What this solution (achieved 0.62809) has done: 'I slightly expand the TF‑IDF vocabularies (set `min_df=1` and increase `max_features`) to capture more useful n‑grams and lower the Ridge regularisation (`alpha=0.05`). These small changes keep the original pipeline structure while giving the linear model a richer representation, which should raise the Pearson correlation toward the target score. The rest of the code—including the train/validation split, clipping, and submission writing—remains unchanged.'
- What this solution (achieved 0.62305) has done: 'We slightly expand the TF‑IDF vocabularies and reduce the Ridge regularisation (alpha = 0.03). These minimal adjustments keep the original pipeline structure while giving the linear model a richer representation and a bit more flexibility, which is expected to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.61104) has done: 'I slightly enlarge the TF‑IDF vocabularies (more word and character n‑grams) and lower the Ridge regularisation (alpha = 0.01). These minimal hyper‑parameter adjustments keep the original pipeline unchanged while giving the linear model richer features and a bit more flexibility, which is expected to raise the validation Pearson toward the target score.'
- What this solution (achieved 0.59361) has done: 'I slightly enlarge the TF‑IDF vocabularies (more word and character n‑grams) and reduce the Ridge regularisation (alpha = 0.005). These minimal adjustments keep the original pipeline unchanged while giving the linear model richer representations and a little more flexibility, which is expected to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.62105) has done: 'I increase the regularisation strength (alpha = 0.1) and enlarge the TF‑IDF vocabularies (more word n‑grams up to 5‑grams and more character n‑grams, with higher max_features). These modest adjustments keep the same Ridge + TF‑IDF pipeline but give the model a richer representation and a better bias‑variance trade‑off, which should raise the validation Pearson closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.48305) has done: 'I lower the Ridge regularisation (alpha = 0.02) and enlarge the TF‑IDF vocabularies, add a sparse StandardScaler after the combined vectoriser, and slightly broaden the character n‑gram range. These minimal tweaks keep the original Ridge + TF‑IDF pipeline while giving the model a richer, properly‑scaled feature space, which is expected to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.43274) has done: 'I keep the original pipeline but increase the Ridge regularisation to `alpha=0.1` (which has shown to improve Pearson in earlier attempts) and add a simple linear calibration step derived from the validation split. The calibration (slope & intercept) is fitted on validation predictions vs. true scores, applied to both validation and test predictions, and then the scores are clipped to [0, 1]. This tiny post‑processing usually raises the correlation without altering the core model or feature engineering.'
- What this solution (achieved 0.60465) has done: 'I reduce the Ridge regularisation to α=0.05 and remove the unnecessary StandardScaler from the pipeline (scaling sparse TF‑IDF often hurts linear models). This keeps the overall Ridge + TF‑IDF architecture unchanged while expectedly raising the validation Pearson correlation, moving the score closer to the target.'
- What this solution (achieved 0.60154) has done: 'I slightly adjust the Ridge regularisation (lower α to 0.02) and expand the TF‑IDF vocabularies (raise max_features for both word‑ and character‑level vectorisers). These tiny hyper‑parameter tweaks keep the original pipeline intact while giving the linear model a richer feature set and a bit more flexibility, which should raise the validation Pearson correlation toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
sep = " "
train_df["inputs"] = (
    train_df["context"] + sep + train_df["anchor"] + sep + train_df["target"]
)
test_df["inputs"] = (
    test_df["context"] + sep + test_df["anchor"] + sep + test_df["target"]
)



## === cell 3
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["score"],
)



## === cell 4
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 5),
    max_features=1_200_000,
    min_df=1,
    sublinear_tf=True,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 7),
    max_features=1_000_000,
    min_df=1,
    sublinear_tf=True,
)

anchor_vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=800_000,
    min_df=1,
    sublinear_tf=True,
)

target_vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    max_features=800_000,
    min_df=1,
    sublinear_tf=True,
)


def column_selector(col):
    return FunctionTransformer(lambda X: X[col].astype("U"), validate=False)


word_pipe = Pipeline([("col", column_selector("inputs")), ("tfidf", word_vectorizer)])
char_pipe = Pipeline([("col", column_selector("inputs")), ("tfidf", char_vectorizer)])
anchor_pipe = Pipeline(
    [("col", column_selector("anchor")), ("tfidf", anchor_vectorizer)]
)
target_pipe = Pipeline(
    [("col", column_selector("target")), ("tfidf", target_vectorizer)]
)

combined_vectorizer = FeatureUnion(
    [
        ("word", word_pipe),
        ("char", char_pipe),
        ("anchor", anchor_pipe),
        ("target", target_pipe),
    ]
)

model = Ridge(alpha=0.05, solver="lsqr", random_state=42)

pipeline = make_pipeline(combined_vectorizer, model)



## === cell 5
pipeline.fit(train_split["inputs"], train_split["score"])
val_pred_raw = pipeline.predict(val_split["inputs"])
slope, intercept = np.polyfit(val_pred_raw, val_split["score"], 1)
val_pred = slope * val_pred_raw + intercept
pearson = np.corrcoef(val_pred, val_split["score"])[0, 1]
print(f"Validation Pearson (calibrated): {pearson:.6f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/index_class_helper.pxi in pandas._libs.index.Int64Engine._check_type()

KeyError: 'inputs'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/512801855.py in <cell line: 0>()
----> 1 pipeline.fit(train_split["inputs"], train_split["score"])
      2 val_pred_raw = pipeline.predict(val_split["inputs"])
      3 slope, intercept = np.polyfit(val_pred_raw, val_split["score"], 1)
      4 val_pred = slope * val_pred_raw + intercept
      5 pearson = np.corrcoef(val_pred, val_split["score"])[0, 1]

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
   1190             sum of `n_components` (output dimension) over transformers.
   1191         """
-> 1192         results = self._parallel_func(X, y, fit_params, _fit_transform_one)
   1193         if not results:
   1194             # All transformers are None

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _parallel_func(self, X, y, fit_params, func)
   1212         transformers = list(self._iter())
   1213 
-> 1214         return Parallel(n_jobs=self.n_jobs)(
   1215             delayed(func)(
   1216                 transformer,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
    435         """
    436         fit_params_steps = self._check_fit_params(**fit_params)
--> 437         Xt = self._fit(X, y, **fit_params_steps)
    438 
    439         last_step = self._final_estimator

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in transform(self, X)
    236         """
    237         X = self._check_input(X, reset=False)
--> 238         return self._transform(X, func=self.func, kw_args=self.kw_args)
    239 
    240     def inverse_transform(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in _transform(self, X, func, kw_args)
    308             func = _identity
    309 
--> 310         return func(X, **(kw_args if kw_args else {}))
    311 
    312     def __sklearn_is_fitted__(self):

/tmp/ipykernel_11/3584648495.py in <lambda>(X)
     35 # Helper to extract a single column from the DataFrame
     36 def column_selector(col):
---> 37     return FunctionTransformer(lambda X: X[col].astype("U"), validate=False)
     38 
     39 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'inputs'

## === cell 6
pipeline.fit(train_df["inputs"], train_df["score"])



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3136265264.py in <cell line: 0>()
----> 1 pipeline.fit(train_df["inputs"], train_df["score"])
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
   1190             sum of `n_components` (output dimension) over transformers.
   1191         """
-> 1192         results = self._parallel_func(X, y, fit_params, _fit_transform_one)
   1193         if not results:
   1194             # All transformers are None

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _parallel_func(self, X, y, fit_params, func)
   1212         transformers = list(self._iter())
   1213 
-> 1214         return Parallel(n_jobs=self.n_jobs)(
   1215             delayed(func)(
   1216                 transformer,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit_transform(self, X, y, **fit_params)
    435         """
    436         fit_params_steps = self._check_fit_params(**fit_params)
--> 437         Xt = self._fit(X, y, **fit_params_steps)
    438 
    439         last_step = self._final_estimator

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in transform(self, X)
    236         """
    237         X = self._check_input(X, reset=False)
--> 238         return self._transform(X, func=self.func, kw_args=self.kw_args)
    239 
    240     def inverse_transform(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in _transform(self, X, func, kw_args)
    308             func = _identity
    309 
--> 310         return func(X, **(kw_args if kw_args else {}))
    311 
    312     def __sklearn_is_fitted__(self):

/tmp/ipykernel_11/3584648495.py in <lambda>(X)
     35 # Helper to extract a single column from the DataFrame
     36 def column_selector(col):
---> 37     return FunctionTransformer(lambda X: X[col].astype("U"), validate=False)
     38 
     39 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'inputs'

## === cell 7
test_pred_raw = pipeline.predict(test_df["inputs"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/603673168.py in <cell line: 0>()
----> 1 test_pred_raw = pipeline.predict(test_df["inputs"])
      2 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in transform(self, X)
   1240             sum of `n_components` (output dimension) over transformers.
   1241         """
-> 1242         Xs = Parallel(n_jobs=self.n_jobs)(
   1243             delayed(_transform_one)(trans, X, None, weight)
   1244             for name, trans, weight in self._iter()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _transform_one(transformer, X, y, weight, **fit_params)
    874 
    875 def _transform_one(transformer, X, y, weight, **fit_params):
--> 876     res = transformer.transform(X)
    877     # if we have a weight for this transformer, multiply output
    878     if weight is None:

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in transform(self, X)
    656         Xt = X
    657         for _, _, transform in self._iter():
--> 658             Xt = transform.transform(Xt)
    659         return Xt
    660 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in transform(self, X)
    236         """
    237         X = self._check_input(X, reset=False)
--> 238         return self._transform(X, func=self.func, kw_args=self.kw_args)
    239 
    240     def inverse_transform(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_function_transformer.py in _transform(self, X, func, kw_args)
    308             func = _identity
    309 
--> 310         return func(X, **(kw_args if kw_args else {}))
    311 
    312     def __sklearn_is_fitted__(self):

/tmp/ipykernel_11/3584648495.py in <lambda>(X)
     35 # Helper to extract a single column from the DataFrame
     36 def column_selector(col):
---> 37     return FunctionTransformer(lambda X: X[col].astype("U"), validate=False)
     38 
     39 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'inputs'

## === cell 8
test_pred = slope * test_pred_raw + intercept



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3296086263.py in <cell line: 0>()
----> 1 test_pred = slope * test_pred_raw + intercept
      2 

NameError: name 'slope' is not defined

## === cell 9
test_pred = np.clip(test_pred, 0, 1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/849374345.py in <cell line: 0>()
----> 1 test_pred = np.clip(test_pred, 0, 1)
      2 

NameError: name 'test_pred' is not defined

## === cell 10
submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3843517207.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

NameError: name 'test_pred' is not defined
