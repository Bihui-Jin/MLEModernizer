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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
h2o==3.46.0.8
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
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.72994

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63864) has done: 'I make the smallest changes needed to (1) ensure the notebook always writes a valid `submission.csv` and (2) nudge the score upward toward your target by aligning post-processing with the QWK metric more reliably. Concretely, I fix the calibration block so it does not depend on `essay_id` being present in H2O CV predictions (it often isn’t), and instead optimizes QWK cutpoints using out-of-fold expected scores aligned by row order. I also make the expected-score extraction more robust to H2O’s probability column naming for multinomial GLM, and ensure deterministic behavior with fixed seeds. Core feature engineering and the GLM model training setup remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator




## === cell 1
def _resolve_input_dir(
    preferred="../input/learning-agency-lab-automated-essay-scoring-2/",
):
    candidates = [
        preferred,
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/input/",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/",
        "/kaggle/data/",
    ]
    for c in candidates:
        if os.path.isdir(c):
            if os.path.basename(os.path.normpath(c)) in ("input", "data"):
                comp = os.path.join(c, "learning-agency-lab-automated-essay-scoring-2")
                if os.path.isdir(comp):
                    return comp + "/"
            return c if c.endswith("/") else c + "/"
    return preferred


input_dir = _resolve_input_dir()
if os.path.isdir(input_dir):
    for fn in sorted(os.listdir(input_dir))[:50]:
        print(fn)
else:
    print(f"Input dir not found: {input_dir}")



## === cell 2
default_color_1 = "darkblue"
default_color_2 = "darkgreen"
default_color_3 = "darkred"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

my_random_seed = 123
np.random.seed(my_random_seed)



## === cell 3
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")
sub_path = os.path.join(input_dir, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)



## === cell 4
df_train.full_text = df_train.full_text.str.strip()
df_test.full_text = df_test.full_text.str.strip()



## === cell 5
df_train.head()



## === cell 6
df_train.score = df_train.score.astype(int)



## === cell 7
df_train.info()



## === cell 8
df_test



## === cell 9
plt.figure(figsize=(8, 4))
df_train.score.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.grid()
plt.title("Score")
plt.show()



## === cell 10
df_train.score.describe()



## === cell 11
df_train["n_char"] = df_train.full_text.str.len()

df_train["n_word"] = df_train.full_text.str.split().map(
    lambda x: len(x) if isinstance(x, list) else 0
)
df_train["n_word"] = df_train["n_word"].clip(lower=1)
df_train["char_per_word"] = df_train.n_char / df_train.n_word

df_test["n_char"] = df_test.full_text.str.len()
df_test["n_word"] = df_test.full_text.str.split().map(
    lambda x: len(x) if isinstance(x, list) else 0
)
df_test["n_word"] = df_test["n_word"].clip(lower=1)
df_test["char_per_word"] = df_test.n_char / df_test.n_word

features_new = ["n_char", "n_word", "char_per_word"]



## === cell 12
df_train[features_new].describe()



## === cell 13
for f in features_new:
    plt.figure(figsize=(10, 3))
    df_train[f].plot(kind="hist", bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 14
for f in features_new:
    plt.figure(figsize=(10, 1))
    plt.boxplot(df_train[f], vert=False)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 15
eps = 1.0  # minimal offset; only affects pathological empty/zero-word texts

df_train["log_n_char"] = np.log10(np.maximum(df_train.n_char.astype(float), eps))
df_train["log_n_word"] = np.log10(np.maximum(df_train.n_word.astype(float), eps))
df_train["log_char_per_word"] = np.log10(
    np.maximum(df_train.char_per_word.astype(float), eps)
)

df_test["log_n_char"] = np.log10(np.maximum(df_test.n_char.astype(float), eps))
df_test["log_n_word"] = np.log10(np.maximum(df_test.n_word.astype(float), eps))
df_test["log_char_per_word"] = np.log10(
    np.maximum(df_test.char_per_word.astype(float), eps)
)

features_log = ["log_n_char", "log_n_word", "log_char_per_word"]



## === cell 16
for f in features_log:
    plt.figure(figsize=(10, 3))
    df_train[f].plot(kind="hist", bins=50, color=default_color_1)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 17
for f in features_log:
    plt.figure(figsize=(10, 1))
    plt.boxplot(df_train[f], vert=False)
    plt.title(f)
    plt.grid()
    plt.show()



## === cell 18
corr_pearson = df_train[["n_char", "n_word", "char_per_word", "score"]].corr(
    method="pearson"
)
fig = plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_pearson,
    annot=True,
    cmap="RdYlGn",
    vmin=-1,
    vmax=+1,
    fmt=".3f",
    linecolor="black",
    linewidths=0.5,
)
plt.title("Pearson Correlation")
plt.show()



## === cell 19
corr_pearson = df_train[
    ["log_n_char", "log_n_word", "log_char_per_word", "score"]
].corr(method="pearson")
fig = plt.figure(figsize=(5, 4))
sns.heatmap(
    corr_pearson,
    annot=True,
    cmap="RdYlGn",
    vmin=-1,
    vmax=+1,
    fmt=".3f",
    linecolor="black",
    linewidths=0.5,
)
plt.title("Pearson Correlation")
plt.show()



## === cell 20
sns.jointplot(data=df_train, x="log_n_char", y="score", color=default_color_1)
plt.show()



## === cell 21
sns.jointplot(data=df_train, x="log_n_word", y="score", color=default_color_1)
plt.show()



## === cell 22
sns.jointplot(data=df_train, x="log_char_per_word", y="score", color=default_color_1)
plt.show()



## === cell 23
df_train.to_csv("training_data.csv", index=False)



## === cell 24
h2o.init(max_mem_size="8G", nthreads=4, name="aes_glm", enable_assertions=False)



## === cell 25
fold_col = "fold_id"
df_train[fold_col] = (np.arange(len(df_train)) % 5).astype(int)

col4upload = ["essay_id"] + features_log + [fold_col]
train_hex = h2o.H2OFrame(df_train[col4upload + ["score"]])
test_hex = h2o.H2OFrame(df_test[["essay_id"] + features_log])



## === cell 26
train_hex["score"] = train_hex["score"].asfactor()
train_hex[fold_col] = train_hex[fold_col].asfactor()
predictors = features_log



## === cell 27
glm_model = H2OGeneralizedLinearEstimator(
    family="multinomial",
    standardize=True,
    nfolds=5,
    fold_column=fold_col,  # Change: ensure CV folds match our fold_id for OOF alignment
    keep_cross_validation_predictions=True,
    alpha=1,  # 0:Ridge (L2), 1:LASSO (L1)
    score_each_iteration=True,
    seed=my_random_seed,
)

glm_model.train(predictors, "score", training_frame=train_hex)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
H2OResponseError                          Traceback (most recent call last)
/tmp/ipykernel_11/2578233114.py in <cell line: 0>()
     10 )
     11 
---> 12 glm_model.train(predictors, "score", training_frame=train_hex)
     13 

/usr/local/lib/python3.11/dist-packages/h2o/estimators/estimator_base.py in train(self, x, y, training_frame, offset_column, fold_column, weights_column, validation_frame, max_runtime_secs, ignored_columns, model_id, verbose)
    105                                  validation_frame=validation_frame, max_runtime_secs=max_runtime_secs,
    106                                  ignored_columns=ignored_columns, model_id=model_id, verbose=verbose)
--> 107         self._train(parms, verbose=verbose)
    108         return self
    109 

/usr/local/lib/python3.11/dist-packages/h2o/estimators/estimator_base.py in _train(self, parms, verbose)
    184 
    185         rest_ver = self._get_rest_version(parms)
--> 186         model_builder_json = h2o.api("POST /%d/ModelBuilders/%s" % (rest_ver, self.algo), data=parms)
    187         job = H2OJob(model_builder_json, job_type=(self.algo + " Model Build"))
    188 

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in api(endpoint, data, json, filename, save_to)
    121     # type checks are performed in H2OConnection class
    122     _check_connection()
--> 123     return h2oconn.request(endpoint, data=data, json=json, filename=filename, save_to=save_to)
    124 
    125 

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in request(self, endpoint, data, json, filename, save_to)
    497                     save_to = save_to(resp)
    498                 self._log_end_transaction(start_time, resp)
--> 499                 return self._process_response(resp, save_to)
    500 
    501             except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in _process_response(response, save_to)
    851         if status_code in {400, 404, 412} and isinstance(data, H2OErrorV3):
    852             data.show_stacktrace = False
--> 853             raise H2OResponseError(data)
    854 
    855         # Server errors (notably 500 = "Server Error")

H2OResponseError: ModelBuilderErrorV3  (water.exceptions.H2OModelBuilderIllegalArgumentException):
    timestamp = 1768006466855
    error_url = '/3/ModelBuilders/glm'
    msg = 'Illegal argument(s) for GLM model: GLM_model_python_1768006461545_1.  Details: ERRR on field: _nfolds: nfolds cannot be specified at the same time as a fold column.'
    dev_msg = 'Illegal argument(s) for GLM model: GLM_model_python_1768006461545_1.  Details: ERRR on field: _nfolds: nfolds cannot be specified at the same time as a fold column.'
    http_status = 412
    values = {'messages': [{'_log_level': 5, '_field_name': '_fold_assignment', '_message': 'Fold assignment is ignored when a fold column is specified.'}, {'_log_level': 1, '_field_name': '_nfolds', '_message': 'nfolds cannot be specified at the same time as a fold column.'}, {'_log_level': 5, '_field_name': '_fold_column', '_message': 'Fold column is ignored when nfolds > 1.'}, {'_log_level': 5, '_field_name': '_tweedie_power', '_message': 'Only for Tweedie Distribution.'}, {'_log_level': 5, '_field_name': '_tweedie_power', '_message': 'Tweedie power is only used for Tweedie distribution.'}, {'_log_level': 5, '_field_name': '_quantile_alpha', '_message': 'Quantile (alpha) is only used for Quantile regression.'}, {'_log_level': 5, '_field_name': '_max_after_balance_size', '_message': 'Balance classes is false, hide max_after_balance_size'}, {'_log_level': 5, '_field_name': '_max_after_balance_size', '_message': 'Only used with balanced classes'}, {'_log_level': 5, '_field_name': '_class_sampling_factors', '_message': 'Class sampling factors is only applicable if balancing classes.'}, {'_log_level': 5, '_field_name': '_balance_classes', '_message': 'Not applicable since class balancing is not required for GLM.'}, {'_log_level': 5, '_field_name': '_max_after_balance_size', '_message': 'Not applicable since class balancing is not required for GLM.'}, {'_log_level': 5, '_field_name': '_class_sampling_factors', '_message': 'Not applicable since class balancing is not required for GLM.'}, {'_log_level': 5, '_field_name': '_tweedie_variance_power', '_message': 'Only applicable with Tweedie family'}, {'_log_level': 5, '_field_name': '_tweedie_link_power', '_message': 'Only applicable with Tweedie family'}, {'_log_level': 5, '_field_name': '_theta', '_message': 'Only applicable with Negative Binomial family'}, {'_log_level': 5, '_field_name': '_lambda_min_ratio', '_message': 'only applies if lambda search is on.'}, {'_log_level': 5, '_field_name': '_nlambdas', '_message': 'only applies if lambda search is on.'}, {'_log_level': 5, '_field_name': '_early_stopping', '_message': 'only applies if lambda search is on.'}], 'algo': 'GLM', 'parameters': {'_train': {'name': 'py_2_sid_a2d2', 'type': 'Key'}, '_valid': None, '_nfolds': 5, '_keep_cross_validation_models': True, '_keep_cross_validation_predictions': True, '_keep_cross_validation_predictions_precision': -1, '_keep_cross_validation_fold_assignment': False, '_parallelize_cross_validation': True, '_auto_rebalance': True, '_preprocessors': None, '_seed': 123, '_fold_assignment': 'AUTO', '_categorical_encoding': 'AUTO', '_max_categorical_levels': 10, '_distribution': 'AUTO', '_tweedie_power': 1.5, '_quantile_alpha': 0.5, '_huber_alpha': 0.9, '_ignored_columns': ['essay_id'], '_ignore_const_cols': True, '_weights_column': None, '_offset_column': None, '_fold_column': 'fold_id', '_treatment_column': None, '_check_constant_response': True, '_is_cv_model': False, '_cv_fold': -1, '_score_each_iteration': True, '_max_runtime_secs': 0.0, '_main_model_time_budget_factor': 0.0, '_stopping_rounds': 0, '_stopping_metric': 'AUTO', '_stopping_tolerance': 0.001, '_response_column': 'score', '_balance_classes': False, '_max_after_balance_size': 5.0, '_class_sampling_factors': None, '_max_confusion_matrix_size': 20, '_checkpoint': None, '_pretrained_autoencoder': None, '_custom_metric_func': None, '_custom_distribution_func': None, '_export_checkpoints_dir': None, '_gainslift_bins': -1, '_auc_type': 'AUTO', '_auuc_type': 'AUTO', '_auuc_nbins': -1, '_standardize': True, '_useDispersion1': False, '_family': 'multinomial', '_link': 'family_default', '_solver': 'AUTO', '_tweedie_variance_power': 0.0, '_tweedie_link_power': 1.0, '_dispersion_estimated': 1.0, '_theta': 1e-10, '_invTheta': 10000000000.0, '_alpha': [1.0], '_lambda': None, '_startval': None, '_calc_like': False, '_random_columns': None, '_score_iteration_interval': -1, '_missing_values_handling': None, '_prior': -1.0, '_lambda_search': False, '_cold_start': False, '_nlambdas': -1, '_non_negative': False, '_lambda_min_ratio': -1.0, '_use_all_factor_levels': False, '_max_iterations': -1, '_intercept': True, '_beta_epsilon': 0.0001, '_dispersion_epsilon': 0.0001, '_max_iterations_dispersion': 3000, '_objective_epsilon': -1.0, '_gradient_epsilon': -1.0, '_obj_reg': -1.0, '_compute_p_values': False, '_remove_collinear_columns': False, '_interactions': None, '_interaction_pairs': None, '_early_stopping': True, '_beta_constraints': None, '_linear_constraints': None, '_expose_constraints': False, '_plug_values': None, '_max_active_predictors': -1, '_stdOverride': False, '_glmType': 'glm', '_generate_scoring_history': False, '_dispersion_parameter_method': 'pearson', '_init_dispersion_parameter': 1.0, '_fix_dispersion_parameter': False, '_build_null_model': False, '_generate_variable_inflation_factors': False, '_tweedie_epsilon': 8e-17, '_fix_tweedie_variance_power': True, '_max_series_index': 5000, '_debugTDispersionOnly': False, '_dispersion_learning_rate': 0.5, '_influence': None, '_keepBetaDiffVar': False, '_testCSZeroGram': False, '_separate_linear_beta': False, '_init_optimal_glm': False, '_constraint_eta0': 0.1258925, '_constraint_tau': 10.0, '_constraint_alpha': 0.1, '_constraint_beta': 0.9, '_constraint_c0': 10.0}, 'error_count': 2}
    exception_msg = 'Illegal argument(s) for GLM model: GLM_model_python_1768006461545_1.  Details: ERRR on field: _nfolds: nfolds cannot be specified at the same time as a fold column.'
    stacktrace = ['water.exceptions.H2OModelBuilderIllegalArgumentException: Illegal argument(s) for GLM model: GLM_model_python_1768006461545_1.  Details: ERRR on field: _nfolds: nfolds cannot be specified at the same time as a fold column.\n', '    water.exceptions.H2OModelBuilderIllegalArgumentException.makeFromBuilder(H2OModelBuilderIllegalArgumentException.java:19)', '    hex.ModelBuilder.trainModelOnH2ONode(ModelBuilder.java:346)', '    water.api.ModelBuilderHandler.handle(ModelBuilderHandler.java:51)', '    water.api.ModelBuilderHandler.handle(ModelBuilderHandler.java:16)', '    water.api.RequestServer.serve(RequestServer.java:472)', '    water.api.RequestServer.doGeneric(RequestServer.java:303)', '    water.api.RequestServer.doPost(RequestServer.java:227)', '    javax.servlet.http.HttpServlet.service(HttpServlet.java:707)', '    javax.servlet.http.HttpServlet.service(HttpServlet.java:790)', '    org.eclipse.jetty.servlet.ServletHolder.handle(ServletHolder.java:799)', '    org.eclipse.jetty.servlet.ServletHandler.doHandle(ServletHandler.java:554)', '    org.eclipse.jetty.server.handler.ScopedHandler.nextHandle(ScopedHandler.java:233)', '    org.eclipse.jetty.server.handler.ContextHandler.doHandle(ContextHandler.java:1440)', '    org.eclipse.jetty.server.handler.ScopedHandler.nextScope(ScopedHandler.java:188)', '    org.eclipse.jetty.servlet.ServletHandler.doScope(ServletHandler.java:505)', '    org.eclipse.jetty.server.handler.ScopedHandler.nextScope(ScopedHandler.java:186)', '    org.eclipse.jetty.server.handler.ContextHandler.doScope(ContextHandler.java:1355)', '    org.eclipse.jetty.server.handler.ScopedHandler.handle(ScopedHandler.java:141)', '    org.eclipse.jetty.server.handler.HandlerCollection.handle(HandlerCollection.java:146)', '    org.eclipse.jetty.server.handler.HandlerWrapper.handle(HandlerWrapper.java:127)', '    water.webserver.jetty9.Jetty9ServerAdapter$LoginHandler.handle(Jetty9ServerAdapter.java:130)', '    org.eclipse.jetty.server.handler.HandlerCollection.handle(HandlerCollection.java:146)', '    org.eclipse.jetty.server.handler.HandlerWrapper.handle(HandlerWrapper.java:127)', '    org.eclipse.jetty.server.Server.handle(Server.java:516)', '    org.eclipse.jetty.server.HttpChannel.lambda$handle$1(HttpChannel.java:487)', '    org.eclipse.jetty.server.HttpChannel.dispatch(HttpChannel.java:732)', '    org.eclipse.jetty.server.HttpChannel.handle(HttpChannel.java:479)', '    org.eclipse.jetty.server.HttpConnection.onFillable(HttpConnection.java:277)', '    org.eclipse.jetty.io.AbstractConnection$ReadCallback.succeeded(AbstractConnection.java:311)', '    org.eclipse.jetty.io.FillInterest.fillable(FillInterest.java:105)', '    org.eclipse.jetty.io.ChannelEndPoint$1.run(ChannelEndPoint.java:104)', '    org.eclipse.jetty.util.thread.strategy.EatWhatYouKill.runTask(EatWhatYouKill.java:338)', '    org.eclipse.jetty.util.thread.strategy.EatWhatYouKill.doProduce(EatWhatYouKill.java:315)', '    org.eclipse.jetty.util.thread.strategy.EatWhatYouKill.tryProduce(EatWhatYouKill.java:173)', '    org.eclipse.jetty.util.thread.strategy.EatWhatYouKill.run(EatWhatYouKill.java:131)', '    org.eclipse.jetty.util.thread.ReservedThreadExecutor$ReservedThread.run(ReservedThreadExecutor.java:409)', '    org.eclipse.jetty.util.thread.QueuedThreadPool.runJob(QueuedThreadPool.java:883)', '    org.eclipse.jetty.util.thread.QueuedThreadPool$Runner.run(QueuedThreadPool.java:1034)', '    java.base/java.lang.Thread.run(Thread.java:829)']
    parameters = {'__meta': {'schema_version': 3, 'schema_name': 'GLMParametersV3', 'schema_type': 'GLMParameters'}, 'model_id': None, 'training_frame': {'__meta': {'schema_version': 3, 'schema_name': 'FrameKeyV3', 'schema_type': 'Key<Frame>'}, 'name': 'py_2_sid_a2d2', 'type': 'Key<Frame>', 'URL': '/3/Frames/py_2_sid_a2d2'}, 'validation_frame': None, 'nfolds': 5, 'keep_cross_validation_models': True, 'keep_cross_validation_predictions': True, 'keep_cross_validation_fold_assignment': False, 'parallelize_cross_validation': True, 'distribution': 'AUTO', 'tweedie_power': 1.5, 'quantile_alpha': 0.5, 'huber_alpha': 0.9, 'response_column': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'score', 'is_member_of_frames': None}, 'weights_column': None, 'offset_column': None, 'fold_column': {'__meta': {'schema_version': 3, 'schema_name': 'ColSpecifierV3', 'schema_type': 'VecSpecifier'}, 'column_name': 'fold_id', 'is_member_of_frames': None}, 'fold_assignment': 'AUTO', 'categorical_encoding': 'AUTO', 'max_categorical_levels': 10, 'ignored_columns': ['essay_id'], 'ignore_const_cols': True, 'score_each_iteration': True, 'checkpoint': None, 'stopping_rounds': 0, 'max_runtime_secs': 0.0, 'stopping_metric': 'AUTO', 'stopping_tolerance': 0.001, 'gainslift_bins': -1, 'custom_metric_func': None, 'custom_distribution_func': None, 'export_checkpoints_dir': None, 'auc_type': 'AUTO', 'seed': 123, 'family': 'multinomial', 'tweedie_variance_power': 0.0, 'dispersion_learning_rate': 0.5, 'tweedie_link_power': 1.0, 'theta': 1e-10, 'solver': 'AUTO', 'alpha': [1.0], 'lambda': None, 'lambda_search': False, 'early_stopping': True, 'nlambdas': -1, 'score_iteration_interval': -1, 'standardize': True, 'cold_start': False, 'missing_values_handling': 'MeanImputation', 'influence': None, 'plug_values': None, 'non_negative': False, 'max_iterations': -1, 'beta_epsilon': 0.0001, 'objective_epsilon': -1.0, 'gradient_epsilon': -1.0, 'obj_reg': -1.0, 'link': 'family_default', 'dispersion_parameter_method': 'pearson', 'startval': None, 'calc_like': False, 'generate_variable_inflation_factors': False, 'intercept': True, 'build_null_model': False, 'fix_dispersion_parameter': False, 'init_dispersion_parameter': 1.0, 'prior': -1.0, 'lambda_min_ratio': -1.0, 'beta_constraints': None, 'linear_constraints': None, 'max_active_predictors': -1, 'interactions': None, 'interaction_pairs': None, 'balance_classes': False, 'class_sampling_factors': None, 'max_after_balance_size': 5.0, 'max_confusion_matrix_size': 20, 'compute_p_values': False, 'fix_tweedie_variance_power': True, 'remove_collinear_columns': False, 'dispersion_epsilon': 0.0001, 'tweedie_epsilon': 8e-17, 'max_iterations_dispersion': 3000, 'generate_scoring_history': False, 'init_optimal_glm': False, 'separate_linear_beta': False, 'constraint_eta0': 0.1258925, 'constraint_tau': 10.0, 'constraint_alpha': 0.1, 'constraint_beta': 0.9, 'constraint_c0': 10.0}
    messages = [{'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'fold_assignment', 'message': 'Fold assignment is ignored when a fold column is specified.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'ERRR', 'field_name': 'nfolds', 'message': 'nfolds cannot be specified at the same time as a fold column.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'fold_column', 'message': 'Fold column is ignored when nfolds > 1.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'tweedie_power', 'message': 'Only for Tweedie Distribution.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'tweedie_power', 'message': 'Tweedie power is only used for Tweedie distribution.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'quantile_alpha', 'message': 'Quantile (alpha) is only used for Quantile regression.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'max_after_balance_size', 'message': 'Balance classes is false, hide max_after_balance_size'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'max_after_balance_size', 'message': 'Only used with balanced classes'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'class_sampling_factors', 'message': 'Class sampling factors is only applicable if balancing classes.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'balance_classes', 'message': 'Not applicable since class balancing is not required for GLM.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'max_after_balance_size', 'message': 'Not applicable since class balancing is not required for GLM.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'class_sampling_factors', 'message': 'Not applicable since class balancing is not required for GLM.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'tweedie_variance_power', 'message': 'Only applicable with Tweedie family'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'tweedie_link_power', 'message': 'Only applicable with Tweedie family'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'theta', 'message': 'Only applicable with Negative Binomial family'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'lambda_min_ratio', 'message': 'only applies if lambda search is on.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'nlambdas', 'message': 'only applies if lambda search is on.'}, {'__meta': {'schema_version': 3, 'schema_name': 'ValidationMessageV3', 'schema_type': 'ValidationMessage'}, 'message_type': 'TRACE', 'field_name': 'early_stopping', 'message': 'only applies if lambda search is on.'}]
    error_count = 2


## === cell 28
glm_model.cross_validation_metrics_summary().as_data_frame()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/230777437.py in <cell line: 0>()
----> 1 glm_model.cross_validation_metrics_summary().as_data_frame()
      2 

/usr/local/lib/python3.11/dist-packages/h2o/model/model_base.py in cross_validation_metrics_summary(self)
    723         :returns: The cross-validation metrics summary as an H2OTwoDimTable
    724         """
--> 725         model = self._model_json["output"]
    726         if "cross_validation_metrics_summary" in model and model["cross_validation_metrics_summary"] is not None:
    727             return model["cross_validation_metrics_summary"]

TypeError: 'NoneType' object is not subscriptable

## === cell 29
glm_model.varimp_plot()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
H2OValueError                             Traceback (most recent call last)
/tmp/ipykernel_11/1251574300.py in <cell line: 0>()
----> 1 glm_model.varimp_plot()
      2 

/usr/local/lib/python3.11/dist-packages/h2o/model/model_base.py in varimp_plot(self, num_of_features, server, save_plot_path)
   1677         if has_extension(self, 'VariableImportance'):
   1678             return self._varimp_plot(num_of_features=num_of_features, server=server, save_plot_path=save_plot_path)
-> 1679         raise H2OValueError("Variable importance plot is not available for this type of model (%s)." % self.algo)
   1680 
   1681     def std_coef_plot(self, num_of_features=None, server=False, save_plot_path=None):

H2OValueError: Variable importance plot is not available for this type of model (glm).

## === cell 30
pred_train = glm_model.predict(train_hex).as_data_frame()
pred_train.head()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
H2OResponseError                          Traceback (most recent call last)
/tmp/ipykernel_11/3048072249.py in <cell line: 0>()
----> 1 pred_train = glm_model.predict(train_hex).as_data_frame()
      2 pred_train.head()
      3 

/usr/local/lib/python3.11/dist-packages/h2o/model/model_base.py in predict(self, test_data, custom_metric, custom_metric_func)
    330             eval_func_ref = h2o.upload_custom_metric(custom_metric)
    331         if not isinstance(test_data, h2o.H2OFrame): raise ValueError("test_data must be an instance of H2OFrame")
--> 332         j = H2OJob(h2o.api("POST /4/Predictions/models/%s/frames/%s" % (self.model_id, test_data.frame_id), data = {'custom_metric_func': custom_metric_func}),
    333                    self._model_json["algo"] + " prediction")
    334         j.poll()

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in api(endpoint, data, json, filename, save_to)
    121     # type checks are performed in H2OConnection class
    122     _check_connection()
--> 123     return h2oconn.request(endpoint, data=data, json=json, filename=filename, save_to=save_to)
    124 
    125 

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in request(self, endpoint, data, json, filename, save_to)
    497                     save_to = save_to(resp)
    498                 self._log_end_transaction(start_time, resp)
--> 499                 return self._process_response(resp, save_to)
    500 
    501             except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in _process_response(response, save_to)
    851         if status_code in {400, 404, 412} and isinstance(data, H2OErrorV3):
    852             data.show_stacktrace = False
--> 853             raise H2OResponseError(data)
    854 
    855         # Server errors (notably 500 = "Server Error")

H2OResponseError: Server error water.exceptions.H2OKeyNotFoundArgumentException:
  Error: Object 'None' not found in function: predict for argument: model
  Request: POST /4/Predictions/models/None/frames/py_2_sid_a2d2


## === cell 31
print(pred_train.predict.value_counts().sort_index())
plt.figure(figsize=(8, 4))
pred_train.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Train")
plt.grid()
plt.show()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1010078975.py in <cell line: 0>()
----> 1 print(pred_train.predict.value_counts().sort_index())
      2 plt.figure(figsize=(8, 4))
      3 pred_train.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
      4 plt.title("Predictions - Train")
      5 plt.grid()

NameError: name 'pred_train' is not defined

## === cell 32
conf_train = pd.crosstab(pred_train.predict, df_train["score"])
sns.heatmap(
    conf_train, annot=True, cmap="Reds", fmt=".0f", linecolor="black", linewidths=0.5
)
plt.title("Confusion Matrix - Training")
plt.show()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3305751701.py in <cell line: 0>()
----> 1 conf_train = pd.crosstab(pred_train.predict, df_train["score"])
      2 sns.heatmap(
      3     conf_train, annot=True, cmap="Reds", fmt=".0f", linecolor="black", linewidths=0.5
      4 )
      5 plt.title("Confusion Matrix - Training")

NameError: name 'pred_train' is not defined

## === cell 33
pred_test = glm_model.predict(test_hex).as_data_frame()
pred_test.head()



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
H2OResponseError                          Traceback (most recent call last)
/tmp/ipykernel_11/429498942.py in <cell line: 0>()
----> 1 pred_test = glm_model.predict(test_hex).as_data_frame()
      2 pred_test.head()
      3 

/usr/local/lib/python3.11/dist-packages/h2o/model/model_base.py in predict(self, test_data, custom_metric, custom_metric_func)
    330             eval_func_ref = h2o.upload_custom_metric(custom_metric)
    331         if not isinstance(test_data, h2o.H2OFrame): raise ValueError("test_data must be an instance of H2OFrame")
--> 332         j = H2OJob(h2o.api("POST /4/Predictions/models/%s/frames/%s" % (self.model_id, test_data.frame_id), data = {'custom_metric_func': custom_metric_func}),
    333                    self._model_json["algo"] + " prediction")
    334         j.poll()

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in api(endpoint, data, json, filename, save_to)
    121     # type checks are performed in H2OConnection class
    122     _check_connection()
--> 123     return h2oconn.request(endpoint, data=data, json=json, filename=filename, save_to=save_to)
    124 
    125 

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in request(self, endpoint, data, json, filename, save_to)
    497                     save_to = save_to(resp)
    498                 self._log_end_transaction(start_time, resp)
--> 499                 return self._process_response(resp, save_to)
    500 
    501             except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in _process_response(response, save_to)
    851         if status_code in {400, 404, 412} and isinstance(data, H2OErrorV3):
    852             data.show_stacktrace = False
--> 853             raise H2OResponseError(data)
    854 
    855         # Server errors (notably 500 = "Server Error")

H2OResponseError: Server error water.exceptions.H2OKeyNotFoundArgumentException:
  Error: Object 'None' not found in function: predict for argument: model
  Request: POST /4/Predictions/models/None/frames/Key_Frame__upload_acc98af6521f9e915bbc5a0bff9a4887.hex


## === cell 34
print(pred_test.predict.value_counts().sort_index())
pred_test.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
plt.title("Predictions - Test")
plt.grid()
plt.show()




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2322724445.py in <cell line: 0>()
----> 1 print(pred_test.predict.value_counts().sort_index())
      2 pred_test.predict.value_counts().sort_index().plot(kind="bar", color=default_color_3)
      3 plt.title("Predictions - Test")
      4 plt.grid()
      5 plt.show()

NameError: name 'pred_test' is not defined

## === cell 35
def _quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=6):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    n = max_rating - min_rating + 1
    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = np.bincount(y_true - min_rating, minlength=n).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(float)

    E = np.outer(act_hist, pred_hist) / float(len(y_true))

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / float((n - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den != 0 else 0.0)


def _expected_score_from_pred_df(pred_df: pd.DataFrame):
    prob_cols = [c for c in pred_df.columns if str(c).startswith("p")]
    if len(prob_cols) == 0:
        return None

    parsed = []
    for c in prob_cols:
        s = str(c)[1:]
        try:
            parsed.append((c, int(float(s))))
        except Exception:
            return None

    parsed.sort(key=lambda x: x[1])
    cls = np.array([p[1] for p in parsed], dtype=float)
    use_cols = [p[0] for p in parsed]

    probs = pred_df[use_cols].to_numpy(dtype=float)
    if probs.shape[1] != cls.shape[0]:
        return None
    return (probs * cls.reshape(1, -1)).sum(axis=1)


def _apply_cutpoints(exp_scores: np.ndarray, cutpoints: np.ndarray):
    scores = np.digitize(exp_scores, bins=cutpoints, right=True) + 1
    return np.clip(scores, 1, 6).astype(int)


def _optimize_cutpoints_qwk(exp_scores: np.ndarray, y_true: np.ndarray, verbose=True):
    exp_scores = np.asarray(exp_scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    qs = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    cutpoints = np.quantile(exp_scores, qs).astype(float)
    cutpoints = np.maximum.accumulate(cutpoints)

    def score(cp):
        y_pred = _apply_cutpoints(exp_scores, cp)
        return _quadratic_weighted_kappa(y_true, y_pred)

    best = score(cutpoints)
    if verbose:
        print("Initial CV QWK:", best, "cutpoints:", cutpoints)

    for _ in range(4):
        improved = False
        for i in range(5):
            left = -np.inf if i == 0 else cutpoints[i - 1]
            right = np.inf if i == 4 else cutpoints[i + 1]

            span = np.std(exp_scores) * 0.25
            grid = np.linspace(cutpoints[i] - span, cutpoints[i] + span, 31)
            grid = grid[(grid > left + 1e-9) & (grid < right - 1e-9)]
            if grid.size == 0:
                continue

            local_best = best
            local_cp = cutpoints[i]
            for v in grid:
                cp_try = cutpoints.copy()
                cp_try[i] = float(v)
                cp_try = np.maximum.accumulate(cp_try)
                s = score(cp_try)
                if s > local_best + 1e-8:
                    local_best = s
                    local_cp = float(v)

            if local_best > best + 1e-8:
                cutpoints[i] = local_cp
                cutpoints = np.maximum.accumulate(cutpoints)
                best = local_best
                improved = True

        if verbose:
            print("Refined CV QWK:", best, "cutpoints:", cutpoints)
        if not improved:
            break

    return cutpoints, best


def _oof_expected_scores_from_cv_models(model, train_hex, fold_col: str, nfolds: int):
    cv_models = model.cross_validation_models()
    if cv_models is None or len(cv_models) == 0:
        return None

    fold_ids = (
        train_hex[fold_col]
        .as_data_frame(use_pandas=True)[fold_col]
        .astype(int)
        .to_numpy()
    )
    oof = np.full(train_hex.nrows, np.nan, dtype=float)

    for k in range(nfolds):
        idx = np.where(fold_ids == k)[0]
        if idx.size == 0:
            continue

        holdout = train_hex[idx.tolist(), :]
        pred_df = cv_models[k].predict(holdout).as_data_frame()
        exp = _expected_score_from_pred_df(pred_df)
        if exp is None or len(exp) != idx.size:
            return None
        oof[idx] = exp

    if np.isnan(oof).any():
        return None
    return oof


exp_test = _expected_score_from_pred_df(pred_test)

cutpoints = None
use_calibration = exp_test is not None

if use_calibration:
    exp_oof = _oof_expected_scores_from_cv_models(
        glm_model, train_hex, fold_col, nfolds=5
    )
    if exp_oof is None:
        print(
            "Warning: Could not construct aligned OOF expected scores; skipping calibration."
        )
        use_calibration = False
    else:
        y_true_cv = df_train["score"].to_numpy(dtype=int)
        cutpoints, cv_qwk = _optimize_cutpoints_qwk(exp_oof, y_true_cv, verbose=True)
        print("Calibration CV QWK (OOF fold-aligned):", cv_qwk)

if (cutpoints is not None) and (exp_test is not None):
    pred_scores = _apply_cutpoints(exp_test, cutpoints).astype(float)
else:
    print(
        "Warning: Using predicted class labels directly (no probability calibration)."
    )
    pred_scores = (
        pd.to_numeric(pred_test["predict"], errors="coerce")
        .astype("Int64")
        .to_numpy()
        .astype(float)
    )

pred_scores = np.clip(pred_scores, 1, 6)
if np.isnan(pred_scores).any():
    mode_score = int(df_train["score"].mode().iloc[0])
    pred_scores = np.where(np.isnan(pred_scores), mode_score, pred_scores)

pred_scores = pred_scores.astype(int)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": pred_scores})
sub["score"] = sub["score"].astype(int)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Score distribution in submission:\n", sub["score"].value_counts().sort_index())

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154676933.py in <cell line: 0>()
    145 
    146 
--> 147 exp_test = _expected_score_from_pred_df(pred_test)
    148 
    149 cutpoints = None

NameError: name 'pred_test' is not defined
