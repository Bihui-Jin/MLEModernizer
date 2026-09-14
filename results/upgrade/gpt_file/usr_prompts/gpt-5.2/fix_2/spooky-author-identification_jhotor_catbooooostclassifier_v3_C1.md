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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.37993

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from catboost import CatBoostClassifier



## === cell 1
TRAIN_PATH = "/kaggle/input/spooky-author-identification/train.csv"
TEST_PATH = "/kaggle/input/spooky-author-identification/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/spooky-author-identification/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, usecols=["id", "text", "author"])
test = pd.read_csv(TEST_PATH, usecols=["id", "text"])

print(train.shape, test.shape)
train.head()



## === cell 2
train.sample(10, random_state=1234)



## === cell 3
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop(columns="author"),
    train["author"],
    test_size=0.25,
    stratify=train["author"],
    random_state=1234,
)

X_train.head()




## === cell 4
def make_model():
    try:
        m = CatBoostClassifier(
            text_features=["text"],
            random_state=1234,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            task_type="GPU",
            devices="0",
            verbose=False,
        )
        return m
    except Exception:
        m = CatBoostClassifier(
            text_features=["text"],
            random_state=1234,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            verbose=False,
        )
        return m


model = make_model()
model



## === cell 5
model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=25,
    verbose=False,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
_catboost.pyx in _catboost.get_float_feature()

_catboost.pyx in _catboost._FloatOrNan()

_catboost.pyx in _catboost._FloatOrNanFromString()

TypeError: Cannot convert 'id11025' to float

During handling of the above exception, another exception occurred:

CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/581932734.py in <cell line: 0>()
      1 # Keep the same training approach with early stopping against a validation split.
----> 2 model.fit(
      3     X_train,
      4     y_train,
      5     eval_set=(X_valid, y_valid),

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2393                 raise CatBoostError("y may be None only when X is an instance of catboost.Pool or string")
   2394 
-> 2395             train_params = self._prepare_train_params(
   2396                 X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features,
   2397                 pairs=pairs, graph=graph, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _prepare_train_params(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks)
   2273         embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')
   2274 
-> 2275         train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
   2276                                        sample_weight, group_id, group_weight, subgroup_id, pairs_weight,
   2277                                        baseline, column_description)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
   1511         if y is None:
   1512             raise CatBoostError("y has not initialized in fit(): X is not catboost.Pool object, y must be not None in fit().")
-> 1513         train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
   1514                           group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline)
   1515     return train_pool

/usr/local/lib/python3.11/dist-packages/catboost/core.py in __init__(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, column_description, pairs, graph, delimiter, has_header, ignore_csv_quoting, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count, log_cout, log_cerr, data_can_be_none)
    853                         )
    854 
--> 855                     self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
    856                                group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
    857             elif not data_can_be_none:

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _init(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1489         if feature_tags is not None:
   1490             feature_tags = self._check_transform_tags(feature_tags, feature_names)
-> 1491         self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
   1492                         group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1493 

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_features_order_layout_pool()

_catboost.pyx in _catboost._set_features_order_data_pd_data_frame()

_catboost.pyx in _catboost.create_num_factor_data()

_catboost.pyx in _catboost.get_float_feature()

CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=0]="id11025": Cannot convert 'id11025' to float

## === cell 6
print(model.predict_proba(X_valid)[:5])
print(y_valid.iloc[:5].tolist())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/8313207.py in <cell line: 0>()
----> 1 print(model.predict_proba(X_valid)[:5])
      2 print(y_valid.iloc[:5].tolist())
      3 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict_proba(self, X, ntree_start, ntree_end, thread_count, verbose, task_type)
   5349                 with probability for every class for each object.
   5350         """
-> 5351         return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
   5352 
   5353     def predict_log_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict_proba(). Use fit() to train model. Then use this method.

## === cell 7
predictions = model.predict(X_valid)
predictions[:5]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/151977496.py in <cell line: 0>()
----> 1 predictions = model.predict(X_valid)
      2 predictions[:5]
      3 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, task_type)
   5305                   with log probability for every class for each object.
   5306         """
-> 5307         return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)
   5308 
   5309     def predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict(). Use fit() to train model. Then use this method.

## === cell 8
predictions = np.array(predictions).reshape(-1).tolist()
predictions[:10]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1293461670.py in <cell line: 0>()
      1 # CatBoost can return shape (n,1) labels; flatten robustly.
----> 2 predictions = np.array(predictions).reshape(-1).tolist()
      3 predictions[:10]
      4 

NameError: name 'predictions' is not defined

## === cell 9
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_valid, predictions))
print(confusion_matrix(y_valid, predictions))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3971092143.py in <cell line: 0>()
      1 from sklearn.metrics import classification_report, confusion_matrix
      2 
----> 3 print(classification_report(y_valid, predictions))
      4 print(confusion_matrix(y_valid, predictions))
      5 

NameError: name 'predictions' is not defined

## === cell 10
final_model = make_model()
final_model.fit(train.drop(columns="author"), train["author"], verbose=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
_catboost.pyx in _catboost.get_float_feature()

_catboost.pyx in _catboost._FloatOrNan()

_catboost.pyx in _catboost._FloatOrNanFromString()

TypeError: Cannot convert 'id06121' to float

During handling of the above exception, another exception occurred:

CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/1515375442.py in <cell line: 0>()
      1 # Train final model on full training data (keeps core logic; no early stopping without eval_set).
      2 final_model = make_model()
----> 3 final_model.fit(train.drop(columns="author"), train["author"], verbose=False)
      4 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2393                 raise CatBoostError("y may be None only when X is an instance of catboost.Pool or string")
   2394 
-> 2395             train_params = self._prepare_train_params(
   2396                 X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features,
   2397                 pairs=pairs, graph=graph, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _prepare_train_params(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks)
   2273         embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')
   2274 
-> 2275         train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph,
   2276                                        sample_weight, group_id, group_weight, subgroup_id, pairs_weight,
   2277                                        baseline, column_description)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
   1511         if y is None:
   1512             raise CatBoostError("y has not initialized in fit(): X is not catboost.Pool object, y must be not None in fit().")
-> 1513         train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, graph=graph, weight=sample_weight, group_id=group_id,
   1514                           group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline)
   1515     return train_pool

/usr/local/lib/python3.11/dist-packages/catboost/core.py in __init__(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, column_description, pairs, graph, delimiter, has_header, ignore_csv_quoting, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count, log_cout, log_cerr, data_can_be_none)
    853                         )
    854 
--> 855                     self._init(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
    856                                group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
    857             elif not data_can_be_none:

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _init(self, data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1489         if feature_tags is not None:
   1490             feature_tags = self._check_transform_tags(feature_tags, feature_names)
-> 1491         self._init_pool(data, label, cat_features, text_features, embedding_features, embedding_features_data, pairs, graph, weight,
   1492                         group_id, group_weight, subgroup_id, pairs_weight, baseline, timestamp, feature_names, feature_tags, thread_count)
   1493 

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_pool()

_catboost.pyx in _catboost._PoolBase._init_features_order_layout_pool()

_catboost.pyx in _catboost._set_features_order_data_pd_data_frame()

_catboost.pyx in _catboost.create_num_factor_data()

_catboost.pyx in _catboost.get_float_feature()

CatBoostError: Bad value for num_feature[non_default_doc_idx=0,feature_idx=0]="id06121": Cannot convert 'id06121' to float

## === cell 11
preds = final_model.predict_proba(test.drop(columns="id"))

classes = list(final_model.classes_)
preds_df = pd.DataFrame(preds, columns=classes)

required_cols = ["EAP", "HPL", "MWS"]
preds_df = preds_df.reindex(columns=required_cols, fill_value=1e-15)

preds_df = pd.concat(
    [test[["id"]].reset_index(drop=True), preds_df.reset_index(drop=True)], axis=1
)
preds_df.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/190639616.py in <cell line: 0>()
      1 # Predict probabilities for submission.
      2 # Ensure we use the same feature columns as training (id is not a feature).
----> 3 preds = final_model.predict_proba(test.drop(columns="id"))
      4 
      5 # CatBoost class order follows sorted class labels typically; map to required columns safely.

/usr/local/lib/python3.11/dist-packages/catboost/core.py in predict_proba(self, X, ntree_start, ntree_end, thread_count, verbose, task_type)
   5349                 with probability for every class for each object.
   5350         """
-> 5351         return self._predict(X, 'Probability', ntree_start, ntree_end, thread_count, verbose, 'predict_proba', task_type)
   5352 
   5353     def predict_log_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type="CPU"):

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _predict(self, data, prediction_type, ntree_start, ntree_end, thread_count, verbose, parent_method_name, task_type)
   2618         if verbose is None:
   2619             verbose = False
-> 2620         data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
   2621         self._validate_prediction_type(prediction_type)
   2622 

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _process_predict_input_data(self, data, parent_method_name, thread_count, label)
   2594     def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
   2595         if not self.is_fitted() or self.tree_count_ is None:
-> 2596             raise CatBoostError(("There is no trained model to use {}(). "
   2597                                  "Use fit() to train model. Then use this method.").format(parent_method_name))
   2598         is_single_object = _is_data_single_object(data)

CatBoostError: There is no trained model to use predict_proba(). Use fit() to train model. Then use this method.

## === cell 12
preds_df.sample(5, random_state=1234)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3162845186.py in <cell line: 0>()
----> 1 preds_df.sample(5, random_state=1234)
      2 

NameError: name 'preds_df' is not defined

## === cell 13
SUB_PATH = "submission.csv"
preds_df.to_csv(SUB_PATH, index=False)
print(
    f"Wrote {SUB_PATH} with shape {preds_df.shape} and columns {preds_df.columns.tolist()}"
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3517680898.py in <cell line: 0>()
      1 # Write valid Kaggle submission.
      2 SUB_PATH = "submission.csv"
----> 3 preds_df.to_csv(SUB_PATH, index=False)
      4 print(
      5     f"Wrote {SUB_PATH} with shape {preds_df.shape} and columns {preds_df.columns.tolist()}"

NameError: name 'preds_df' is not defined

## === cell 14
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print(sample_sub.head())
print("Sample submission columns:", sample_sub.columns.tolist())
print("Our submission columns:", preds_df.columns.tolist())
assert (
    preds_df.columns.tolist() == sample_sub.columns.tolist()
), "Submission columns do not match sample submission."

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1918385003.py in <cell line: 0>()
      3 print(sample_sub.head())
      4 print("Sample submission columns:", sample_sub.columns.tolist())
----> 5 print("Our submission columns:", preds_df.columns.tolist())
      6 assert (
      7     preds_df.columns.tolist() == sample_sub.columns.tolist()

NameError: name 'preds_df' is not defined
