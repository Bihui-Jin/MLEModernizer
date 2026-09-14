# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
prettytable==3.16.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os, time, re

import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import pickle

from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from prettytable import PrettyTable

import warnings

warnings.filterwarnings("ignore")



## === cell 2
d = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/"
df = pd.read_csv(d + "train.csv")
print("Shape of dataset is: ", df.shape)

identity_columns = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

cols = ["id", "comment_text"] + identity_columns + ["target"]
df = df[cols]
print("Now shape of the data: ", df.shape)
df.head(2)




## === cell 3
def convert_to_bool(data, cols):
    for col in cols:
        data[col] = np.where(data[col] >= 0.5, True, False)
    return data


print(df.dtypes)

df = convert_to_bool(df, identity_columns)

df["target"] = df["target"].apply(lambda x: 1 if x >= 0.5 else 0)
print("\n\n")
df.head()



## === cell 4
comments = df["comment_text"].values
print(comments[0])
print("=" * 100)
print(comments[50])
print("=" * 100)
print(comments[100])
print("=" * 100)
print(comments[1000])
print("=" * 100)



## === cell 5
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stop_words = stop_words - {"not"}




## === cell 6
def text_process(row):
    try:
        text = row["comment_text"]
        text = str(text).lower()

        text = (
            text.replace("won't", "will not")
            .replace("cannot", "can not")
            .replace("can't", "can not")
            .replace("n't", " not")
            .replace("what's", "what is")
            .replace("it's", "itis")
            .replace("'ve", " have")
            .replace("i'm", "i am")
            .replace("'re", " are")
            .replace("he's", "he is")
            .replace("she's", "she is")
            .replace("'s", " own")
            .replace("%", " percent ")
            .replace("₹", " rupee ")
            .replace("$", " dollar")
            .replace("€", " euro ")
            .replace("'ll", " will")
        )

        text = re.sub(r"<.*?>", "", text)  # removes the htmltags

        text = re.sub("[^a-zA-Z0-9\n]", " ", text)
        text = re.sub("\s+", " ", text)

        text_to_words = []
        for word in text.split():
            if word not in stop_words:
                text_to_words.append(word)
            else:
                continue
        text = " ".join(text_to_words)

        return text
    except Exception:
        print("There is no value in comment_text, so returnin 'nan'")
        return np.nan




## === cell 7
tic = time.time()
print("processing train data...")
df.loc[:, "comment_text"] = df.apply(text_process, axis=1)
print("Time take to process the text data: {:.2f} seconds".format(time.time() - tic))



## === cell 8
X = df[[col for col in df.columns]]
Y = df[["target"]]

X_train, X_test, y_train, Y_test = model_selection.train_test_split(
    X, Y, train_size=0.8, stratify=Y, random_state=42
)
X_test, X_cv, y_test, y_cv = model_selection.train_test_split(
    X_test, Y_test, train_size=0.5, stratify=Y_test, random_state=42
)

print(
    "Number of datapoints in train data: {:,}\n\
Number of datapoints in CV data: {:,}\n\
Number of datapoints in test data: {:,}".format(
        X_train.shape[0], X_cv.shape[0], X_test.shape[0]
    )
)



## === cell 9
a = y_train["target"].value_counts()
cl_wt = {0: a[1], 1: a[0]}
print(cl_wt)

subgroups = identity_columns
actual_label = "target"
pred_label = "pred_target"



## === cell 10
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def cal_auc(y_true, y_pred):
    "returns the auc value"
    return roc_auc_score(y_true, y_pred)


def cal_subgroup_auc(data, subgroup, actual_label, pred_label):
    subgroup_examples = data[data[subgroup]]
    return cal_auc(subgroup_examples[actual_label], subgroup_examples[pred_label])


def cal_bpsn_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BPSN auc"""
    subgroup_negative_examples = data[data[subgroup] & ~data[actual_label]]
    background_positive_examples = data[~data[subgroup] & data[actual_label]]
    bpsn_examples = pd.concat(
        [subgroup_negative_examples, background_positive_examples], axis=0
    )
    return cal_auc(bpsn_examples[actual_label], bpsn_examples[pred_label])


def cal_bnsp_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BNSP auc"""
    subgroup_positive_examples = data[data[subgroup] & data[actual_label]]
    background_negative_examples = data[~data[subgroup] & ~data[actual_label]]
    bnsp_examples = pd.concat(
        [subgroup_positive_examples, background_negative_examples], axis=0
    )
    return cal_auc(bnsp_examples[actual_label], bnsp_examples[pred_label])


def cal_bias_metric(data, subgroups, actual_label, pred_label):
    """Computes per-subgroup metrics for all subgroups and one model
    and returns the dataframe which will have all three Bias metrices
    and number of exmaples for each subgroup"""
    records = []
    for subgroup in subgroups:
        record = {"subgroup": subgroup, "subgroup_size": len(data[data[subgroup]])}
        record["subgroup_auc"] = cal_subgroup_auc(
            data, subgroup, actual_label, pred_label
        )
        record["bpsn_auc"] = cal_bpsn_auc(data, subgroup, actual_label, pred_label)
        record["bnsp_auc"] = cal_bnsp_auc(data, subgroup, actual_label, pred_label)
        records.append(record)
    submetric_df = pd.DataFrame(records).sort_values("subgroup_auc", ascending=True)
    return submetric_df


def cal_overall_auc(data, actual_label, pred_label):
    return roc_auc_score(data[actual_label], data[pred_label])


def power_mean(series, p):
    total_sum = np.sum(np.power(series, p))
    return np.power(total_sum / len(series), 1 / p)


def final_metric(submetric_df, overall_auc, p=-5, w=0.25):
    generalized_subgroup_auc = power_mean(submetric_df["subgroup_auc"], p)
    generalized_bpsn_auc = power_mean(submetric_df["bpsn_auc"], p)
    generalized_bnsp_auc = power_mean(submetric_df["bnsp_auc"], p)
    overall_metric = w * overall_auc + w * (
        generalized_subgroup_auc + generalized_bpsn_auc + generalized_bnsp_auc
    )
    return overall_metric


def return_final_metric(data, subgroups, actual_label, pred_label, verbose=False):
    """Data is dataframe which include whole data and it also has the predicted target column"""
    submetric_df = cal_bias_metric(data, subgroups, actual_label, pred_label)
    if verbose:
        print("printing the submetric table for each identity or subgroup")
        print(submetric_df)
    overall_auc = cal_overall_auc(data, actual_label, pred_label)
    overall_metric = final_metric(submetric_df, overall_auc, p=-5, w=0.25)
    return overall_metric, submetric_df


def plot_confusion_matrix(train, cv, test):
    tr_pred = np.where(train["pred_target"] >= 0.5, 1, 0)
    cv_pred = np.where(cv["pred_target"] >= 0.5, 1, 0)
    te_pred = np.where(test["pred_target"] >= 0.5, 1, 0)

    tr_con_mat = confusion_matrix(train["target"], tr_pred)
    cv_con_mat = confusion_matrix(cv["target"], cv_pred)
    te_con_mat = confusion_matrix(test["target"], te_pred)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 4))
    sns.heatmap(tr_con_mat, annot=True, fmt="d", annot_kws={"size": 15}, ax=ax1)
    ax1.set_title("For Train data", fontsize=15)
    ax1.set_xlabel("Pridicted target", fontsize=12)
    ax1.set_ylabel("Actual target", fontsize=12)

    sns.heatmap(cv_con_mat, annot=True, fmt="d", annot_kws={"size": 15}, ax=ax2)
    ax2.set_title("For CV data", fontsize=15)
    ax2.set_xlabel("Pridicted target", fontsize=12)
    ax2.set_ylabel("Actual target", fontsize=12)

    sns.heatmap(te_con_mat, annot=True, fmt="d", annot_kws={"size": 15}, ax=ax3)
    ax3.set_title("For Test data", fontsize=15)
    ax3.set_xlabel("Pridicted target", fontsize=12)
    ax3.set_ylabel("Actual target", fontsize=12)

    plt.show()


def plot_confusion_for_each_identity(train, cv, test, subgroups):
    for subgroup in subgroups:
        print(
            "{}{} for '{}' identity {}".format(" " * 40, "*" * 15, subgroup, "*" * 15)
        )
        TR, CV, TE = train[train[subgroup]], cv[cv[subgroup]], test[test[subgroup]]
        plot_confusion_matrix(TR, CV, TE)
        print("\n\n")


def plot_auc(params, train_auc, cv_auc, hyp_name):
    plt.figure(figsize=(12, 8))
    plt.plot(params, train_auc, "bo-", label="Train")
    plt.plot(params, cv_auc, "ro-", label="CV")
    plt.title("Final Metric (AUC) Plot", fontsize=18)
    plt.xlabel("Hyperparameter '{}'".format(hyp_name), fontsize=14)
    plt.ylabel("Modified AUC (Final Metric)", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(1)
    plt.show()


def tune_GBDT(train_x, train_y):
    params = {
        "n_estimators": [50, 100, 200, 400, 500, 800, 1000],
        "max_depth": [2, 5, 8, 10, 15, 25, 50],
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    }

    gbdt_clf = LGBMClassifier(
        boosting_type="gbdt", n_jobs=-1, class_weight=cl_wt, random_state=42
    )
    gbdt_model = RandomizedSearchCV(
        gbdt_clf, params, cv=3, scoring="roc_auc", n_jobs=-1
    )
    start = time.time()
    print("Tunning the parameters...")
    gbdt_model.fit(train_x, train_y)
    print(
        "Done!\nTime take to tune the hyper-parameters: {:.4f} seconds.\n".format(
            time.time() - start
        )
    )

    print("Best parameters of tunned model is:\n", gbdt_model.best_params_)
    print("\nBest score of of tunned model is: {:.4f}\n".format(gbdt_model.best_score_))

    best_model = gbdt_model.best_estimator_
    print("Best model is:\n", best_model)
    return best_model




## === cell 11
def return_best_model(
    tr_df,
    cv_df,
    train_x,
    train_y,
    CV_x,
    subgroups,
    actual_label,
    pred_label,
    model_name,
    path,
    gram,
):
    """retruns trained model and save it given path"""
    if model_name == "log_reg":
        best_model = tune_log_reg(
            tr_df, cv_df, train_x, train_y, CV_x, subgroups, actual_label, pred_label
        )
    elif model_name == "DT":
        best_model = tune_DT(
            tr_df, cv_df, train_x, train_y, CV_x, subgroups, actual_label, pred_label
        )
    elif model_name == "RF":
        best_model = tune_RF(train_x, train_y)
    elif model_name == "GBDT":
        train_x = train_x.astype("float64")
        best_model = tune_GBDT(train_x, train_y)

    print("\n\nTraining the best model...")
    best_model.fit(train_x, train_y)

    file = path + model_name + str(gram) + ".pkl"
    print("Saving the model in path...")
    with open(file, "wb") as f:
        pickle.dump(best_model, f)

    return best_model


def report_model(
    model,
    tr_df,
    cv_df,
    te_df,
    train_x,
    cv_x,
    test_x,
    subgroups,
    actual_label,
    pred_label,
    CV_bool=True,
):
    tr_df["pred_target"] = model.predict_proba(train_x)[:, 1]
    cv_df["pred_target"] = model.predict_proba(cv_x)[:, 1]
    te_df["pred_target"] = model.predict_proba(test_x)[:, 1]

    final_train_auc, _ = return_final_metric(
        tr_df, subgroups, actual_label, pred_label, verbose=False
    )
    final_cv_auc, _ = return_final_metric(
        cv_df, subgroups, actual_label, pred_label, verbose=False
    )
    final_test_auc, _ = return_final_metric(
        te_df, subgroups, actual_label, pred_label, verbose=False
    )

    print(
        "Final metric for:\nTrain: {:.5f}\nCV: {:.5f}\nTest: {:.5f}".format(
            final_train_auc, final_cv_auc, final_test_auc
        )
    )
    print("\n\nPloting Confusion matrix for whole data....\n")
    plot_confusion_matrix(tr_df, cv_df, te_df)

    print("\n\nPlotting confusion matrix indentity-wise ...\n")
    plot_confusion_for_each_identity(tr_df, cv_df, te_df, subgroups)

    return final_train_auc, final_cv_auc, final_test_auc




## === cell 12
uni_bow_vectorizer = CountVectorizer(min_df=1, max_features=10000)
uni_bow_train = uni_bow_vectorizer.fit_transform(X_train["comment_text"].values)
uni_bow_cv = uni_bow_vectorizer.transform(X_cv["comment_text"].values)
uni_bow_test = uni_bow_vectorizer.transform(X_test["comment_text"].values)



## === cell 13
uni_bow_train = uni_bow_train.astype("float64")
uni_bow_cv = uni_bow_cv.astype("float64")
uni_bow_test = uni_bow_test.astype("float64")

model_path_in = "/kaggle/input/my-model/GBDT1.pkl"
model_path_out_dir = "/kaggle/working/"
model_name = "GBDT"
gram = 1

if os.path.isfile(model_path_in):
    with open(model_path_in, "rb") as f:
        best_model = pickle.load(f)
    print("Loaded existing model from:", model_path_in)
else:
    best_model = return_best_model(
        X_train,
        X_cv,
        uni_bow_train,
        y_train["target"].values,
        uni_bow_cv,
        subgroups,
        actual_label,
        pred_label,
        model_name=model_name,
        path=model_path_out_dir,
        gram=gram,
    )

tr, cv, te = report_model(
    best_model,
    X_train.copy(),
    X_cv.copy(),
    X_test.copy(),
    uni_bow_train,
    uni_bow_cv,
    uni_bow_test,
    subgroups,
    actual_label,
    pred_label,
)



## === cell 14
submission = pd.read_csv(
    "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"
)

tic = time.time()
print("processing test data...")
submission.loc[:, "comment_text"] = submission.apply(text_process, axis=1)
print("Time take to process the text data: {:.2f} seconds".format(time.time() - tic))
submission.head(2)



## === cell 15
test_x_to_submit = uni_bow_vectorizer.transform(submission["comment_text"].values)
print(test_x_to_submit.shape)



## === cell 16
test_x_to_submit = test_x_to_submit.astype("float64")
y_test_pred = best_model.predict_proba(test_x_to_submit)[:, 1].astype(np.float32)

submission_out = submission[["id"]].copy()
submission_out["prediction"] = y_test_pred

assert list(submission_out.columns) == ["id", "prediction"]
assert len(submission_out) == 205780

submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
