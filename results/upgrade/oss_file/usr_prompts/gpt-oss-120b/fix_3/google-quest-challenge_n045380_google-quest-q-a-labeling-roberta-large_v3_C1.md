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

3.14

# 3. Installed packages

geopandas==0.14.4
huggingface-hub==0.36.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.3649142456650344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

LOCAL_MODEL_PATH = (
    "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3"
)

if os.path.isdir(LOCAL_MODEL_PATH) and os.path.exists(
    os.path.join(LOCAL_MODEL_PATH, "config.json")
):
    MODEL_NAME = LOCAL_MODEL_PATH
else:
    MODEL_NAME = "microsoft/deberta-v3-large"

MAX_LENGTH = 512
BATCH_SIZE = 8  # Adjust based on your GPU memory
SEED = 42

CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta_large_fold2_val_spearman0.4418.ckpt"

QUESTION_TARGET_COLS = [
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
]

ANSWER_TARGET_COLS = [
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

TARGET_COLS = QUESTION_TARGET_COLS + ANSWER_TARGET_COLS
NUM_QUESTION_TARGETS = len(QUESTION_TARGET_COLS)
NUM_ANSWER_TARGETS = len(ANSWER_TARGET_COLS)
NUM_TARGETS = len(TARGET_COLS)

DROPOUT_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]

print(f"Number of targets: {NUM_TARGETS}")
print(f"Question targets: {NUM_QUESTION_TARGETS}")
print(f"Answer targets: {NUM_ANSWER_TARGETS}")
print(f"Model identifier: {MODEL_NAME}")
print(f"Checkpoint: {CHECKPOINT_PATH}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2376109102.py in <cell line: 0>()
      1 DATA_DIR = "/kaggle/input/google-quest-challenge"
----> 2 TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
      3 TEST_FILE = os.path.join(DATA_DIR, "test.csv")
      4 
      5 # Path that may contain a local pretrained model – fall back to the HuggingFace repo if not usable

NameError: name 'os' is not defined

## === cell 1
def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, device: torch.device
) -> nn.Module:
    """
    Load model from a PyTorch Lightning checkpoint.
    If the checkpoint file is missing or cannot be loaded, fall back to a
    freshly instantiated pretrained model.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    model = SiameseQuestModel(model_name, pretrained=True)

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            state_dict = checkpoint.get("state_dict", {})
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Failed to load checkpoint ({e}). Using pretrained weights.")
    else:
        print("Checkpoint not found – initializing model with pretrained weights.")

    model = model.to(device)
    model.eval()
    return model




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3494246203.py in <cell line: 0>()
      1 def load_model_from_checkpoint(
----> 2     checkpoint_path: str, model_name: str, device: torch.device
      3 ) -> nn.Module:
      4     """
      5     Load model from a PyTorch Lightning checkpoint.

NameError: name 'torch' is not defined

## === cell 2
print(f"Loading tokenizer: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(
        f"Tokenizer loading failed ({e}). Falling back to 'microsoft/deberta-v3-large'."
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
print(f"Vocab size: {tokenizer.vocab_size}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/266150440.py in <cell line: 0>()
----> 1 print(f"Loading tokenizer: {MODEL_NAME}")
      2 try:
      3     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
      4 except Exception as e:
      5     print(

NameError: name 'MODEL_NAME' is not defined

## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)

all_predictions = []
all_qa_ids = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Predicting"):
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)

        logits = model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_attention_mask,
            a_input_ids=a_input_ids,
            a_attention_mask=a_attention_mask,
        )

        preds = torch.sigmoid(logits)

        all_predictions.append(preds.cpu().numpy())
        all_qa_ids.extend(batch["qa_id"])

predictions = np.concatenate(all_predictions, axis=0)
print(f"Predictions shape: {predictions.shape}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2527335737.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print(f"Using device: {device}")
      3 
      4 model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)
      5 

NameError: name 'torch' is not defined

## === cell 4
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", [int(x) for x in all_qa_ids])

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

print(f"Submission shape: {submission.shape}")
submission.to_csv("submission.csv", index=False)
print("submission.csv created!")
submission.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3111608873.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(predictions, columns=TARGET_COLS)
      2 submission.insert(0, "qa_id", [int(x) for x in all_qa_ids])
      3 
      4 for col in TARGET_COLS:
      5     submission[col] = submission[col].clip(0, 1)

NameError: name 'pd' is not defined
