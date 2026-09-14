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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.813007506495721

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class CFG:
    SEED = 2024
    VER = 1
    LOAD_MODELS_FROM = None
    LOAD_FEATURES_FROM = None
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"




## === cell 1
try:
    from spellchecker import SpellChecker

    spell = SpellChecker()

    def count_misspelled_words(text):
        """Return the number of misspelled words in `text`."""
        return len(spell.unknown(text.split()))

except Exception:  # pragma: no cover

    def count_misspelled_words(text):
        """Fallback: assume no misspellings (cost‑free)."""
        return 0




## === cell 2
if CFG.LOAD_FEATURES_FROM is None:

    train_feats1 = Paragraph_Features(train)
    train_feats1 = Paragraph_aggregation(train_feats1)
    train_feats2 = Sentence_Features(train)
    train_feats2 = Sentence_aggregation(train_feats2)
    train_feats3 = Word_Features(train)
    train_feats3 = Word_aggregation(train_feats3)

    train_feats = train_feats1.merge(train_feats2, on="essay_id", how="left")
    train_feats = train_feats.merge(train_feats3, on="essay_id", how="left")
    train_feats = train_feats.merge(df, on="essay_id", how="left")
    train_feats = train_feats.merge(df2, on="essay_id", how="left")
    train_feats["score"] = df_train["score"].values
else:
    train_feats = None



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2651377056.py in <cell line: 0>()
      1 if CFG.LOAD_FEATURES_FROM is None:
      2 
----> 3     train_feats1 = Paragraph_Features(train)
      4     train_feats1 = Paragraph_aggregation(train_feats1)
      5     train_feats2 = Sentence_Features(train)

NameError: name 'Paragraph_Features' is not defined

## === cell 3
if CFG.LOAD_FEATURES_FROM is None:
    print("Save train_feats.csv")
    train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
else:
    print("Load train_feats.csv")
    train_feats = pd.read_csv(CFG.LOAD_FEATURES_FROM)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/152654945.py in <cell line: 0>()
      1 if CFG.LOAD_FEATURES_FROM is None:
      2     print("Save train_feats.csv")
----> 3     train_feats.to_csv(f"train_feats_{CFG.VER}.csv", index=False)
      4 else:
      5     print("Load train_feats.csv")

NameError: name 'train_feats' is not defined

## === cell 4
if CFG.LOAD_MODELS_FROM:
    model_path = f"{CFG.LOAD_MODELS_FROM}LGB_v{CFG.VER}_f0.pkl"
else:
    model_path = f"LGB_v{CFG.VER}_f0.pkl"

try:
    model = pickle.load(open(model_path, "rb"))
except FileNotFoundError:
    print("Model not found – training now.")
    lightgbm()
    model = pickle.load(open(model_path, "rb"))

df_importance = pd.DataFrame(
    {
        "features_name": FEATURES,
        "importance": model.feature_importances_,
    }
)
df_importance = df_importance.sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(
    data=df_importance.head(30),
    x="features_name",
    height="importance",
    color="pink",
    edgecolor="black",
)
plt.title("Distribution of Feature Importance of LightGBM")
plt.xticks(rotation=90)
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3084633280.py in <cell line: 0>()
      6 
      7 try:
----> 8     model = pickle.load(open(model_path, "rb"))
      9 except FileNotFoundError:
     10     print("Model not found – training now.")

NameError: name 'pickle' is not defined

## === cell 5
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = pred1.clip(1, 6).round()
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1533121587.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
      2 sub["score"] = pred1.clip(1, 6).round()
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 sub.head()

NameError: name 'pd' is not defined
