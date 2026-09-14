# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.14

# 3. Installed packages

geopandas==0.14.4
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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
textblob==0.19.0
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

1.0907981714516313

# 6. Current score

1.91375

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.89192) has done: 'I wrap all imports that may fail (including transformers and sentence‑transformers) in a single try/except so the notebook continues even if protobuf incompatibilities arise. I also guard the similarity‑model loading code against missing classes. Finally, instead of a pure uniform prediction I compute simple TextBlob sentiment scores for each response, turn the polarity difference into a sigmoid‑based probability for models A and B, give a tiny constant to the tie class, and renormalise. This small, deterministic heuristic should improve the log‑loss slightly toward the target while keeping the core logic unchanged and ensuring a valid CSV with the required columns is written.'
- What this solution (achieved 2.07884) has done: 'I make the import of the heavy transformers library optional by wrapping it in a broad try/except so that any protobuf‑related crash is suppressed and the notebook continues. Then I slightly enhance the heuristic prediction: combine TextBlob polarity + subjectivity differences and a scaled length difference before applying the sigmoid, and use a smaller tie constant. These changes prevent the runtime error and give a modest improvement in log‑loss while keeping the original logic intact.'
- What this solution (achieved 1.91375) has done: 'Implemented a safe import handling to avoid protobuf‑related crashes by bypassing heavy `transformers` imports entirely. Enhanced the heuristic prediction: now uses exponentiated differences for A/B scores and a distance‑based tie component, providing a more calibrated probability distribution while keeping the original lightweight approach. The script now runs end‑to‑end and writes a proper Kaggle submission CSV.'

# 9. Code solution

## === cell 0
import logging
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow/CUDA if used
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Attempting to register.*factory")
warnings.filterwarnings("ignore", message="The 'repr' attribute.*")



## === cell 1
AutoTokenizer = None
AutoModelForCausalLM = None
pipeline = None
AutoModelForSequenceClassification = None
AutoModel = None
SentenceTransformer = None

import pandas as pd
import numpy as np
import torch
import json
import ast
import re
import time
from tqdm import tqdm
from textblob import TextBlob

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print("⚠️ cosine_similarity import failed – will not be used.", e)
    cosine_similarity = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_FILE = "/kaggle/input/lmsys-chatbot-arena/test.csv"
CHECKPOINT_DIR = "checkpoints_llm_judge"
FEATURE_OUTPUT = "lmsys_test_features_final.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



## === cell 2
SIM_MODEL_PATH = "/kaggle/input/all-minilm-l6-v2-offline"
print("\nContents of similarity model directory (may be empty in this environment):")
if os.path.isdir(SIM_MODEL_PATH):
    for root, dirs, files in os.walk(SIM_MODEL_PATH):
        level = root.replace(SIM_MODEL_PATH, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:5]:
            print(f"{indent}  {f}")
else:
    print("⚠️ Similarity model directory not found – placeholder features will be used.")



## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(SIM_MODEL_PATH):
    print(f"Loading similarity model from: {SIM_MODEL_PATH}")
    if AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
    else:
        tokenizer = None
    if AutoModel:
        model = AutoModel.from_pretrained(SIM_MODEL_PATH, local_files_only=True)
    else:
        model = None
    if model:
        model.to(device)
        model.eval()
        print("✅ Similarity model loaded.")
else:
    print("⚠️ Similarity model path not found – similarity scores will be set to 0.5.")
    tokenizer = None
    model = None



## === cell 4
test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
print("Test shape:", test_df.shape)
test_df.head()



## === cell 5
logger.info("Creating placeholder feature matrix for test data.")
desired_order = [
    "id",
    "sentiment_negative_A",
    "sentiment_neutral_A",
    "sentiment_positive_A",
    "sentiment_negative_B",
    "sentiment_neutral_B",
    "sentiment_positive_B",
    "semantic_similarity_A",
    "semantic_similarity_B",
    "textblob_polarity_A",
    "textblob_subjectivity_A",
    "textblob_polarity_B",
    "textblob_subjectivity_B",
    "flesch_kincaid_grade_A",
    "gunning_fog_A",
    "flesch_kincaid_grade_B",
    "gunning_fog_B",
    "sentiment_positive_diff_A_minus_B",
    "sentiment_negative_diff_A_minus_B",
    "sentiment_neutral_diff_A_minus_B",
    "textblob_polarity_diff_A_minus_B",
    "textblob_subjectivity_diff_A_minus_B",
    "flesch_kincaid_grade_diff_A_minus_B",
    "gunning_fog_diff_A_minus_B",
    "semantic_similarity_diff_A_minus_B",
    "len_prompt",
    "len_response_a",
    "len_response_b",
    "len_diff_A_minus_B",
    "p_codefences",
    "p_bullets",
    "p_numlist",
    "p_list_lines",
    "p_is_question",
    "p_asks_steps",
    "p_asks_code",
    "p_asks_math",
    "p_asks_advice",
    "p_compare",
    "p_summarize",
    "p_rewrite",
    "p_translate",
    "p_classify",
    "a_chars",
    "a_words",
    "a_sents",
    "a_paragraphs",
    "a_codefences",
    "a_headings",
    "a_bullets",
    "a_numlist",
    "a_list_lines",
    "a_qmarks",
    "a_exclaims",
    "a_qmarks_per100w",
    "a_exclaims_per100w",
    "a_list_lines_per100w",
    "a_codefences_per100w",
    "a_headings_per100w",
    "b_chars",
    "b_words",
    "b_sents",
    "b_paragraphs",
    "b_codefences",
    "b_headings",
    "b_bullets",
    "b_numlist",
    "b_list_lines",
    "b_qmarks",
    "b_exclaims",
    "b_qmarks_per100w",
    "b_exclaims_per100w",
    "b_list_lines_per100w",
    "b_codefences_per100w",
    "b_headings_per100w",
    "log_ratio_chars",
    "ratio_chars",
    "log_ratio_words",
    "ratio_words",
    "log_ratio_sents",
    "ratio_sents",
    "log_ratio_paragraphs",
    "ratio_paragraphs",
    "log_ratio_codefences",
    "ratio_codefences",
    "log_ratio_headings",
    "ratio_headings",
    "log_ratio_list_lines",
    "ratio_list_lines",
    "diff_qmarks",
    "ratio_qmarks",
    "diff_exclaims",
    "ratio_exclaims",
    "diff_qmarks_per100w",
    "ratio_qmarks_per100w",
    "diff_exclaims_per100w",
    "ratio_exclaims_per100w",
    "diff_list_lines_per100w",
    "ratio_list_lines_per100w",
    "diff_codefences_per100w",
    "ratio_codefences_per100w",
    "diff_headings_per100w",
    "ratio_headings_per100w",
    "a_to_prompt_word_ratio",
    "b_to_prompt_word_ratio",
    "a_longer_word",
    "a_longer_char",
    "helpfulness_proxy_A",
    "helpfulness_proxy_B",
    "helpfulness_proxy_diff_A_minus_B",
]

df_features = pd.DataFrame({"id": test_df["id"]})
for col in desired_order:
    if col == "id":
        continue
    df_features[col] = 0.0

df_features.to_csv(FEATURE_OUTPUT, index=False)
logger.info(
    f"Placeholder features written to {FEATURE_OUTPUT} (shape: {df_features.shape})"
)




## === cell 6
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


logger.info(
    "Generating enhanced heuristic predictions based on TextBlob sentiment, subjectivity, and length."
)
test_data = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
ids = test_data["id"].values

polarity_a = (
    test_data["response_a"].apply(lambda x: TextBlob(str(x)).sentiment.polarity).values
)
polarity_b = (
    test_data["response_b"].apply(lambda x: TextBlob(str(x)).sentiment.polarity).values
)

subjectivity_a = (
    test_data["response_a"]
    .apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    .values
)
subjectivity_b = (
    test_data["response_b"]
    .apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    .values
)

len_a = test_data["response_a"].astype(str).str.len().values
len_b = test_data["response_b"].astype(str).str.len().values
len_diff_scaled = (len_a - len_b) / 100.0

diff = (
    polarity_a
    - polarity_b
    + 0.3 * (subjectivity_a - subjectivity_b)
    + 0.1 * len_diff_scaled
)

score_a = np.exp(diff)
score_b = np.exp(-diff)

tie_raw = np.exp(-np.abs(diff) * 2) * 0.1  # scale factor gives modest tie mass

total = score_a + score_b + tie_raw

prob_a = score_a / total
prob_b = score_b / total
prob_tie = tie_raw / total

submission = pd.DataFrame(
    {
        "id": ids,
        "winner_model_a": prob_a,
        "winner_model_b": prob_b,
        "winner_tie": prob_tie,
    }
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(submission.head())
print(f"✅ Submission saved to {submission_path} with shape: {submission.shape}")
