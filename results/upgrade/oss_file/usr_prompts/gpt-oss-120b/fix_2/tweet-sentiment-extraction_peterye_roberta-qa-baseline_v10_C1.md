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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.4975326359272003

# 6. Current score

0.59357

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.59357) has done: 'I fixed the runtime errors by removing the failing TensorFlow GPU check, correcting the transformer imports, and skipping the unavailable pretrained model. Instead of using a missing QA model, I added a simple heuristic that selects the sentiment word from each tweet when present (or the whole tweet otherwise). This ensures the pipeline runs end‑to‑end, produces a correctly sized `submission.csv`, and gives a reasonable baseline score while keeping the original workflow structure.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
try:
    import tensorflow as tf

    device_name = tf.test.gpu_device_name()
    if device_name == "/device:GPU:0":
        print("Found GPU at: {}".format(device_name))
    else:
        print("GPU not found, proceeding with CPU.")
except Exception as e:
    print(f"GPU check skipped due to error: {e}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")




## === cell 3
try:
    from transformers import RobertaTokenizerFast
except Exception as e:
    print(f"RobertaTokenizerFast could not be imported: {e}")
    RobertaTokenizerFast = None  # fallback to simple python split




## === cell 4
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
print("Number of test sentences: {:,}".format(df_test.shape[0]))


def simple_prediction(row):
    sentiment = str(row["sentiment"]).strip().lower()
    text = str(row["text"])
    if sentiment in text.lower():
        lower_text = text.lower()
        start_idx = lower_text.find(sentiment)
        end_idx = start_idx + len(sentiment)
        while start_idx > 0 and lower_text[start_idx - 1] != " ":
            start_idx -= 1
        while end_idx < len(lower_text) and lower_text[end_idx] != " ":
            end_idx += 1
        selected = text[start_idx:end_idx].strip()
        return selected if selected else text
    else:
        return text


predictions = df_test.apply(simple_prediction, axis=1).tolist()
print(f"Generated {len(predictions)} predictions.")




## === cell 5
def id_to_word(answer_start, answer_end, input_ids):
    return ""


def get_start_end(start_score, end_score):
    return np.zeros(len(start_score)), np.zeros(len(start_score))




## === cell 6
pass




## === cell 7
df_sub = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
if len(df_sub) != len(predictions):
    raise ValueError(
        f"Length mismatch: submission template has {len(df_sub)} rows, "
        f"but predictions have {len(predictions)}."
    )
df_sub["selected_text"] = predictions

df_sub["selected_text"] = df_sub["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
df_sub["selected_text"] = df_sub["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
df_sub["selected_text"] = df_sub["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

output_path = "/kaggle/working/submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 8
df_sub.head()
