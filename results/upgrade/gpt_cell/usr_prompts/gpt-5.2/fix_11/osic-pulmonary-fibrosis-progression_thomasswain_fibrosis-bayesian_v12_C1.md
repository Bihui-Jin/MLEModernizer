# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import random
import numpy as np  # linear algebra

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("PYTENSOR_FLAGS", os.environ.get("PYTENSOR_FLAGS", ""))
os.environ.setdefault("THEANO_FLAGS", os.environ.get("THEANO_FLAGS", ""))

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

pm = None

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        break



## === cell 1
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_raw = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

train.drop(train[train.Patient == "ID00197637202246865691526"].index, inplace=True)

train = pd.concat([train, test], axis=0, ignore_index=True).drop_duplicates()

le_id = LabelEncoder()
train["PatientID"] = le_id.fit_transform(train["Patient"])

train.head()




## === cell 2
def patient_class_df(df: pd.DataFrame) -> np.ndarray:
    mapping = {
        ("Male", "Currently smokes"): 0,
        ("Male", "Ex-smoker"): 1,
        ("Male", "Never smoked"): 2,
        ("Female", "Currently smokes"): 3,
        ("Female", "Ex-smoker"): 4,
        ("Female", "Never smoked"): 5,
    }
    keys = list(zip(df["Sex"].values, df["SmokingStatus"].values))
    return np.fromiter((mapping.get(k, np.nan) for k in keys), dtype=float).astype(int)


train["Class"] = patient_class_df(train)
test["Class"] = patient_class_df(test)

test.head()
train.loc[train["Patient"] == "ID00007637202177411956430"]["Class"].max()




## === cell 3
def model_fit(data, examine=True):
    n_patients = data["Patient"].nunique()
    FVC_obs = data["FVC"].values
    Weeks = data["Weeks"].values
    PatientID = data["PatientID"].values
    patient_class = data["Class"].values

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data("Weeks_shared", Weeks)
        PatientID_shared = pm.Data("PatientID_shared", PatientID)
        patient_class_shared = pm.Data("patient_class_shared", patient_class)

        mu_a = pm.Normal("mu_a", mu=1700.0, sigma=400)
        sigma_a = pm.HalfNormal("sigma_a", 1000.0)
        mu_b = pm.Normal("mu_b", mu=-4.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal("sigma", 150.0, shape=6)

        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal(
            "FVC_like",
            mu=FVC_est,
            sigma=sigma[patient_class_shared],
            observed=FVC_obs_shared,
        )

        trace = pm.sample(
            2000, tune=2000, target_accept=0.9, init="adapt_diag", random_seed=SEED
        )

    if examine:
        with model:
            pm.traceplot(trace)

    return model, trace




## === cell 4
def generate_template(data):
    patients = data["Patient"].unique()
    weeks = np.arange(-12, 134, dtype=np.int16)
    n_p = len(patients)
    n_w = len(weeks)

    pat_rep = np.repeat(patients, n_w)
    week_tile = np.tile(weeks, n_p)

    class_by_patient = data.groupby("Patient", sort=False)["Class"].max()
    cls_rep = np.repeat(class_by_patient.reindex(patients).values.astype(np.int16), n_w)

    pred_template = pd.DataFrame(
        {
            "Patient": pat_rep,
            "Weeks": week_tile,
            "Class": cls_rep,
        }
    )
    pred_template["PatientID"] = le_id.transform(pred_template["Patient"]).astype(
        np.int32
    )
    return pred_template[["PatientID", "Weeks", "Patient", "Class"]]




## === cell 5
template_train_test = generate_template(test)
template_train_test.head()




## === cell 6
def model_predict(model, trace, template):
    with model:
        pm.set_data(
            {
                "PatientID_shared": template["PatientID"].values.astype(int),
                "Weeks_shared": template["Weeks"].values.astype(int),
                "FVC_obs_shared": np.zeros(len(template), dtype=int),
                "patient_class_shared": template["Class"].values.astype(int),
            }
        )
        post_pred = pm.sample_posterior_predictive(trace, random_seed=SEED)
    fvc_like = post_pred["FVC_like"]  # shape: (draws, n_obs)

    fvc_mean = fvc_like.mean(axis=0)
    fvc_std = fvc_like.std(axis=0)

    df = pd.DataFrame(
        {
            "Patient": le_id.inverse_transform(template["PatientID"].values),
            "Weeks": template["Weeks"].values,
            "FVC_pred": fvc_mean,
            "sigma": fvc_std,
        }
    )
    df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
    df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
    df = pd.merge(
        df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
    )
    df = df.rename(columns={"FVC": "FVC_true"})
    return df




## === cell 7
def examine_predictions(data):
    n = (data["Patient"].nunique()) + 1
    f, axes = plt.subplots((n // 3) + 1, 3, figsize=(15, 5 * ((n // 3) + 1)))
    for i, patient in enumerate(data["Patient"].unique()):
        ax = axes[i // 3, i % 3]
        df = data[data["Patient"] == patient]
        x = df["Weeks"]
        ax.set_title(patient)
        ax.plot(x, df["FVC_true"], "o")
        ax.plot(x, df["FVC_pred"])
        ax = sns.regplot(x, df["FVC_true"], ax=ax, ci=None, line_kws={"color": "red"})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"], alpha=0.5, color="#ffcd3c")
        ax.set_ylabel("FVC")
    axes[n // 3, n % 3].plot()




## === cell 8
def evaluate_predictions(df, use_only_last_3_measures=True, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby("Patient").tail(3)
    else:
        y = df.dropna()

    rmse = ((y["FVC_pred"] - y["FVC_true"]) ** 2).mean() ** (1 / 2)
    mae = (y["FVC_pred"] - y["FVC_true"]).abs()
    mae_mean = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    mae_sd = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    mae_max = (np.sqrt((y["FVC_pred"] - y["FVC_true"]) ** 2)).mean()
    sigma_c = y["sigma"].values
    sigma_c[sigma_c < 70] = 70
    delta = (y["FVC_pred"] - y["FVC_true"]).abs()
    delta[delta > 1000] = 1000
    lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y["sigma_c"] = y["sigma"]
    y["sigma_c"].values[y["sigma_c"].values < 70] = 70
    y["delta_c"] = (y["FVC_pred"] - y["FVC_true"]).abs()
    y["delta_c"].values[y["delta_c"].values > 1000] = 1000
    y["main_loss"] = y["delta_c"] / y["sigma_c"]
    if examine:
        plt.hist(y["main_loss"], bins=100)

    return lll.mean()




## === cell 9
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model, trace = model_fit(train, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train)
    template_train.head()
    pred_train = model_predict(model, trace, template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid)
        pred_valid = model_predict(model, trace, template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid




## === cell 10
for fold in range(0):
    examine_lll = True

    all_patients = train["Patient"].unique()
    validation_patients = np.random.choice(all_patients, size=20, replace=False)
    df_valid = train[train["Patient"].isin(validation_patients)]
    df_train = train[~train["Patient"].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby("Patient").head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f"Fold: {fold}")
    pred_train, pred_valid, lll_train, lll_valid = evaluation_cycle(
        df_train, df_valid, examine_trace=True, examine_preds=True
    )

    print(f"Laplace Log Likelihoods for fold: {fold}")
    print(f"Training:     {lll_train:.4f}")
    print(f"Validation:   {lll_valid:.4f}")
    print("")

    evaluate_predictions(pred_train, use_only_last_3_measures=True, examine=examine_lll)
    evaluate_predictions(pred_valid, use_only_last_3_measures=True, examine=examine_lll)



## === cell 11
if pm is None:
    try:
        import pymc3 as pm  # type: ignore  # noqa: F401
    except Exception:
        import pymc as pm  # type: ignore  # noqa: F401

print("Fit model ...")
model, trace = model_fit(train, examine=False)
print("")

print("Make predictions for test data ...")
template_test = generate_template(test)
template_test.head()
pred_test = model_predict(model, trace, template_test)

final = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"])
final["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
final["FVC"] = pred_test["FVC_pred"]
final["Confidence"] = pred_test["sigma"]
final.head()
final.to_csv("submission.csv", index=False)
print(final.shape)
final.head()

## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/lazylinker_c.py[0m in [0;36m<module>[0;34m[0m
[1;32m     65[0m         [0;32mif[0m [0mversion[0m [0;34m!=[0m [0mactual_version[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 66[0;31m             raise ImportError(
[0m[1;32m     67[0m                 [0;34m"Version check of the existing lazylinker compiled file."[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: Version check of the existing lazylinker compiled file. Looking for version 0.31, but found 0.211. Extra debug information: force_compile=False, _need_reload=True

During handling of the above exception, another exception occurred:

[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/lazylinker_c.py[0m in [0;36m<module>[0;34m[0m
[1;32m     86[0m             [0;32mif[0m [0mversion[0m [0;34m!=[0m [0mactual_version[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 87[0;31m                 raise ImportError(
[0m[1;32m     88[0m                     [0;34m"Version check of the existing lazylinker compiled file."[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: Version check of the existing lazylinker compiled file. Looking for version 0.31, but found 0.211. Extra debug information: force_compile=False, _need_reload=True

During handling of the above exception, another exception occurred:

[0;31mAssertionError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/447499719.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0mprint[0m[0;34m([0m[0;34m"Fit model ..."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0mmodel[0m[0;34m,[0m [0mtrace[0m [0;34m=[0m [0mmodel_fit[0m[0;34m([0m[0mtrain[0m[0;34m,[0m [0mexamine[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0mprint[0m[0;34m([0m[0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/950538634.py[0m in [0;36mmodel_fit[0;34m(data, examine)[0m
[1;32m     32[0m [0;34m[0m[0m
[1;32m     33[0m         [0;31m# Keep exact sampling parameters; add random_seed for determinism.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m         trace = pm.sample(
[0m[1;32m     35[0m             [0;36m2000[0m[0;34m,[0m [0mtune[0m[0;34m=[0m[0;36m2000[0m[0;34m,[0m [0mtarget_accept[0m[0;34m=[0m[0;36m0.9[0m[0;34m,[0m [0minit[0m[0;34m=[0m[0;34m"adapt_diag"[0m[0;34m,[0m [0mrandom_seed[0m[0;34m=[0m[0mSEED[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/sampling/mcmc.py[0m in [0;36msample[0;34m(draws, tune, chains, cores, random_seed, progressbar, progressbar_theme, step, var_names, nuts_sampler, initvals, init, jitter_max_retries, n_init, trace, discard_tuned_samples, compute_convergence_checks, keep_warning_stat, return_inferencedata, idata_kwargs, nuts_sampler_kwargs, callback, mp_ctx, blas_cores, model, compile_kwargs, **kwargs)[0m
[1;32m    823[0m             [0;34m[[0m[0mkwargs[0m[0;34m.[0m[0msetdefault[0m[0;34m([0m[0mk[0m[0;34m,[0m [0mv[0m[0;34m)[0m [0;32mfor[0m [0mk[0m[0;34m,[0m [0mv[0m [0;32min[0m [0mnuts_kwargs[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    824[0m         [0;32mwith[0m [0mjoined_blas_limiter[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 825[0;31m             initial_points, step = init_nuts(
[0m[1;32m    826[0m                 [0minit[0m[0;34m=[0m[0minit[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    827[0m                 [0mchains[0m[0;34m=[0m[0mchains[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/sampling/mcmc.py[0m in [0;36minit_nuts[0;34m(init, chains, n_init, model, random_seed, progressbar, jitter_max_retries, tune, initvals, compile_kwargs, **kwargs)[0m
[1;32m   1589[0m         ]
[1;32m   1590[0m [0;34m[0m[0m
[0;32m-> 1591[0;31m     [0mlogp_dlogp_func[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mlogp_dlogp_function[0m[0;34m([0m[0mravel_inputs[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0;34m**[0m[0mcompile_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1592[0m     [0mlogp_dlogp_func[0m[0;34m.[0m[0mtrust_input[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1593[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/model/core.py[0m in [0;36mlogp_dlogp_function[0;34m(self, grad_vars, tempered, initial_point, ravel_inputs, **kwargs)[0m
[1;32m    563[0m         [0minput_vars[0m [0;34m=[0m [0;34m{[0m[0mi[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mgraph_inputs[0m[0;34m([0m[0mcosts[0m[0;34m)[0m [0;32mif[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mi[0m[0;34m,[0m [0mConstant[0m[0;34m)[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    564[0m         [0;32mif[0m [0minitial_point[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 565[0;31m             [0minitial_point[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0minitial_point[0m[0;34m([0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    566[0m         extra_vars_and_values = {
[1;32m    567[0m             [0mvar[0m[0;34m:[0m [0minitial_point[0m[0;34m[[0m[0mvar[0m[0;34m.[0m[0mname[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/model/core.py[0m in [0;36minitial_point[0;34m(self, random_seed)[0m
[1;32m   1031[0m             [0mMaps[0m [0mnames[0m [0mof[0m [0mtransformed[0m [0mvariables[0m [0mto[0m [0mnumeric[0m [0minitial[0m [0mvalues[0m [0;32min[0m [0mthe[0m [0mtransformed[0m [0mspace[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1032[0m         """
[0;32m-> 1033[0;31m         [0mfn[0m [0;34m=[0m [0mmake_initial_point_fn[0m[0;34m([0m[0mmodel[0m[0;34m=[0m[0mself[0m[0;34m,[0m [0mreturn_transformed[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1034[0m         [0;32mreturn[0m [0mPoint[0m[0;34m([0m[0mfn[0m[0;34m([0m[0mrandom_seed[0m[0;34m)[0m[0;34m,[0m [0mmodel[0m[0;34m=[0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1035[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/initial_point.py[0m in [0;36mmake_initial_point_fn[0;34m(model, overrides, jitter_rvs, default_strategy, return_transformed)[0m
[1;32m    173[0m     [0;31m# when calling the final seeded function[0m[0;34m[0m[0;34m[0m[0m
[1;32m    174[0m     [0minitial_values[0m [0;34m=[0m [0mreplace_rng_nodes[0m[0;34m([0m[0minitial_values[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 175[0;31m     [0mfunc[0m [0;34m=[0m [0mcompile[0m[0;34m([0m[0minputs[0m[0;34m=[0m[0;34m[[0m[0;34m][0m[0;34m,[0m [0moutputs[0m[0;34m=[0m[0minitial_values[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0mpytensor[0m[0;34m.[0m[0mcompile[0m[0;34m.[0m[0mmode[0m[0;34m.[0m[0mFAST_COMPILE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    176[0m [0;34m[0m[0m
[1;32m    177[0m     [0mvarnames[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pymc/pytensorf.py[0m in [0;36mcompile[0;34m(inputs, outputs, random_seed, mode, **kwargs)[0m
[1;32m    940[0m     [0mopt_qry[0m [0;34m=[0m [0mmode[0m[0;34m.[0m[0mprovided_optimizer[0m[0;34m.[0m[0mincluding[0m[0;34m([0m[0;34m"random_make_inplace"[0m[0;34m,[0m [0mcheck_parameter_opt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    941[0m     [0mmode[0m [0;34m=[0m [0mMode[0m[0;34m([0m[0mlinker[0m[0;34m=[0m[0mmode[0m[0;34m.[0m[0mlinker[0m[0;34m,[0m [0moptimizer[0m[0;34m=[0m[0mopt_qry[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 942[0;31m     pytensor_function = pytensor.function(
[0m[1;32m    943[0m         [0minputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    944[0m         [0moutputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/compile/function/__init__.py[0m in [0;36mfunction[0;34m(inputs, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input, trust_input)[0m
[1;32m    330[0m         [0;31m# note: pfunc will also call orig_function -- orig_function is[0m[0;34m[0m[0;34m[0m[0m
[1;32m    331[0m         [0;31m#      a choke point that all compilation must pass through[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 332[0;31m         fn = pfunc(
[0m[1;32m    333[0m             [0mparams[0m[0;34m=[0m[0minputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    334[0m             [0moutputs[0m[0;34m=[0m[0moutputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/compile/function/pfunc.py[0m in [0;36mpfunc[0;34m(params, outputs, mode, updates, givens, no_default_updates, accept_inplace, name, rebuild_strict, allow_input_downcast, profile, on_unused_input, output_keys, fgraph, trust_input)[0m
[1;32m    464[0m     )
[1;32m    465[0m [0;34m[0m[0m
[0;32m--> 466[0;31m     return orig_function(
[0m[1;32m    467[0m         [0minputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    468[0m         [0mcloned_outputs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/compile/function/types.py[0m in [0;36morig_function[0;34m(inputs, outputs, mode, accept_inplace, name, profile, on_unused_input, output_keys, fgraph, trust_input)[0m
[1;32m   1833[0m         )
[1;32m   1834[0m         [0;32mwith[0m [0mconfig[0m[0;34m.[0m[0mchange_flags[0m[0;34m([0m[0mcompute_test_value[0m[0;34m=[0m[0;34m"off"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1835[0;31m             [0mfn[0m [0;34m=[0m [0mm[0m[0;34m.[0m[0mcreate[0m[0;34m([0m[0mdefaults[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1836[0m     [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1837[0m         [0;32mif[0m [0mprofile[0m [0;32mand[0m [0mfn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/compile/function/types.py[0m in [0;36mcreate[0;34m(self, input_storage, storage_map)[0m
[1;32m   1717[0m [0;34m[0m[0m
[1;32m   1718[0m         [0;32mwith[0m [0mconfig[0m[0;34m.[0m[0mchange_flags[0m[0;34m([0m[0mtraceback__limit[0m[0;34m=[0m[0mconfig[0m[0;34m.[0m[0mtraceback__compile_limit[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1719[0;31m             _fn, _i, _o = self.linker.make_thunk(
[0m[1;32m   1720[0m                 [0minput_storage[0m[0;34m=[0m[0minput_storage_lists[0m[0;34m,[0m [0mstorage_map[0m[0;34m=[0m[0mstorage_map[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1721[0m             )

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/basic.py[0m in [0;36mmake_thunk[0;34m(self, input_storage, output_storage, storage_map, **kwargs)[0m
[1;32m    243[0m         [0;34m**[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    244[0m     ) -> tuple["BasicThunkType", "InputStorageType", "OutputStorageType"]:
[0;32m--> 245[0;31m         return self.make_all(
[0m[1;32m    246[0m             [0minput_storage[0m[0;34m=[0m[0minput_storage[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    247[0m             [0moutput_storage[0m[0;34m=[0m[0moutput_storage[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/vm.py[0m in [0;36mmake_all[0;34m(self, profiler, input_storage, output_storage, storage_map)[0m
[1;32m   1283[0m             [0mpost_thunk_clear[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1284[0m [0;34m[0m[0m
[0;32m-> 1285[0;31m         vm = self.make_vm(
[0m[1;32m   1286[0m             [0morder[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1287[0m             [0mthunks[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/vm.py[0m in [0;36mmake_vm[0;34m(self, nodes, thunks, input_storage, output_storage, storage_map, post_thunk_clear, computed, compute_map, updated_vars)[0m
[1;32m   1011[0m [0;34m[0m[0m
[1;32m   1012[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1013[0;31m             [0;32mfrom[0m [0mpytensor[0m[0;34m.[0m[0mlink[0m[0;34m.[0m[0mc[0m[0;34m.[0m[0mcvm[0m [0;32mimport[0m [0mCVM[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1014[0m         [0;32mexcept[0m [0;34m([0m[0mMissingGXX[0m[0;34m,[0m [0mImportError[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1015[0m             [0mCVM[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/cvm.py[0m in [0;36m<module>[0;34m[0m
[1;32m     11[0m             [0;34m"lazylinker will not be imported if pytensor.config.cxx is not set."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m         )
[0;32m---> 13[0;31m     [0;32mfrom[0m [0mpytensor[0m[0;34m.[0m[0mlink[0m[0;34m.[0m[0mc[0m[0;34m.[0m[0mlazylinker_c[0m [0;32mimport[0m [0mCLazyLinker[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m     [0;32mclass[0m [0mCVM[0m[0;34m([0m[0mCLazyLinker[0m[0;34m,[0m [0mVM[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/lazylinker_c.py[0m in [0;36m<module>[0;34m[0m
[1;32m    120[0m [0;34m[0m[0m
[1;32m    121[0m             [0margs[0m [0;34m=[0m [0mGCC_compiler[0m[0;34m.[0m[0mcompile_args[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0mGCC_compiler[0m[0;34m.[0m[0mcompile_str[0m[0;34m([0m[0mdirname[0m[0;34m,[0m [0mcode[0m[0;34m,[0m [0mlocation[0m[0;34m=[0m[0mloc[0m[0;34m,[0m [0mpreargs[0m[0;34m=[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m             [0;31m# Save version into the __init__.py file.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0minit_py[0m [0;34m=[0m [0mloc[0m [0;34m/[0m [0;34m"__init__.py"[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/cmodule.py[0m in [0;36mcompile_str[0;34m(module_name, src_code, location, include_dirs, lib_dirs, libs, preargs, py_module, hide_symbols)[0m
[1;32m   2688[0m                 [0;32mpass[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2689[0m             [0;32massert[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0misfile[0m[0;34m([0m[0mlib_filename[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2690[0;31m             [0;32mreturn[0m [0mdlimport[0m[0;34m([0m[0mlib_filename[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2691[0m [0;34m[0m[0m
[1;32m   2692[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytensor/link/c/cmodule.py[0m in [0;36mdlimport[0;34m(fullpath, suffix)[0m
[1;32m    353[0m             [0;32mdel[0m [0msys[0m[0;34m.[0m[0mpath[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    354[0m [0;34m[0m[0m
[0;32m--> 355[0;31m     [0;32massert[0m [0mfullpath[0m[0;34m.[0m[0mstartswith[0m[0;34m([0m[0mrval[0m[0;34m.[0m[0m__file__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    356[0m     [0;32mreturn[0m [0mrval[0m[0;34m[0m[0;34m[0m[0m
[1;32m    357[0m [0;34m[0m[0m

[0;31mAssertionError[0m:
