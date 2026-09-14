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

0.358533902324924

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.03189) has done: 'I replace the local model path with the HuggingFace identifier `"microsoft/deberta-v3-large"` so the tokenizer can be loaded, add a safe fallback that creates the model from pretrained weights when the checkpoint file is missing, and adjust the inference cell to use this fallback model. These fixes resolve the import errors, allow the dataset and dataloader to be built, and guarantee that a valid `submission.csv` file is written.'
- What this solution (achieved 0.03189) has done: 'I added a safe loading routine for the checkpoint: the code now attempts to load the file with `torch.load(..., weights_only=True)` inside a try‑except block. If loading fails (e.g., due to protobuf incompatibility) it falls back to initializing a fresh pretrained model, keeping the original inference pipeline unchanged. This resolves the `MessageFactory` AttributeError and allows the model (including the fine‑tuned checkpoint when it loads successfully) to generate predictions and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "microsoft/deberta-v3-large"

MAX_LENGTH = 512
BATCH_SIZE = 8  # Adjust based on your GPU memory
SEED = 42

CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/2/deberta_v3_large_fold3_val_spearman0.4392.ckpt"

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
print(f"Checkpoint: {CHECKPOINT_PATH}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2103064582.py in <cell line: 0>()
      1 DATA_DIR = "/kaggle/input/google-quest-challenge"
----> 2 TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
      3 TEST_FILE = os.path.join(DATA_DIR, "test.csv")
      4 
      5 MODEL_NAME = "microsoft/deberta-v3-large"

NameError: name 'os' is not defined

## === cell 1
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Spearman correlation."""
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return np.mean(scores) if scores else 0.0


set_seed(SEED)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2078970057.py in <cell line: 0>()
     22 
     23 
---> 24 def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
     25     """Compute mean Spearman correlation."""
     26     scores = []

NameError: name 'np' is not defined

## === cell 2
class QuestDataset(Dataset):
    """
    Dataset for Google QUEST Q&A Labeling.
    Creates two inputs for Siamese architecture:
    - Question input: [CLS] question_title [SEP] question_body [SEP]
    - Answer input: [CLS] question_title + body [SEP] answer [SEP]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        self.question_titles = [clean_text(t) for t in self.df["question_title"].values]
        self.question_bodies = [clean_text(t) for t in self.df["question_body"].values]
        self.answers = [clean_text(t) for t in self.df["answer"].values]

        if not is_test:
            self.targets = self.df[TARGET_COLS].values.astype(np.float32)

        self.qa_ids = self.df["qa_id"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = self.question_titles[idx]
        body = self.question_bodies[idx]
        answer = self.answers[idx]

        q_encoding = self.tokenizer(
            title,
            body,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        question_text = f"{title} {body}"
        a_encoding = self.tokenizer(
            question_text,
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        item = {
            "q_input_ids": q_encoding["input_ids"].squeeze(0),
            "q_attention_mask": q_encoding["attention_mask"].squeeze(0),
            "a_input_ids": a_encoding["input_ids"].squeeze(0),
            "a_attention_mask": a_encoding["attention_mask"].squeeze(0),
            "qa_id": self.qa_ids[idx],
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/951744261.py in <cell line: 0>()
----> 1 class QuestDataset(Dataset):
      2     """
      3     Dataset for Google QUEST Q&A Labeling.
      4     Creates two inputs for Siamese architecture:
      5     - Question input: [CLS] question_title [SEP] question_body [SEP]

NameError: name 'Dataset' is not defined

## === cell 3
class AttentionPooling(nn.Module):
    """Attention-weighted pooling over sequence dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = F.softmax(weights, dim=1)
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for better generalization."""

    def __init__(self, dropout_rates: List[float] = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        outputs = torch.stack([linear(dropout(x)) for dropout in self.dropouts])
        return outputs.mean(dim=0)


class SiameseQuestModel(nn.Module):
    """
    Siamese Dual-Transformer for Google QUEST Q&A Labeling.

    Architecture:
    - Shared transformer encoder (DeBERTa-v3-large)
    - Two branches: Question and Answer
    - Weighted layer aggregation
    - Attention pooling
    - Multi-sample dropout
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            config.output_hidden_states = True
            self.transformer = AutoModel.from_config(config)

        hidden_size = self.transformer.config.hidden_size
        num_layers = self.transformer.config.num_hidden_layers

        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))

        self.q_attention = AttentionPooling(hidden_size)
        self.a_attention = AttentionPooling(hidden_size)

        self.multi_dropout = MultiSampleDropout()

        self.question_head = nn.Linear(hidden_size, NUM_QUESTION_TARGETS)
        self.answer_head = nn.Linear(hidden_size, NUM_ANSWER_TARGETS)
        self.combined_head = nn.Linear(hidden_size * 2, NUM_TARGETS)

    def weighted_layer_pooling(
        self, hidden_states: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)  # (layers, batch, seq, hidden)
        weights = F.softmax(self.layer_weights, dim=0)  # (layers)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def encode_branch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pooling: AttentionPooling,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.weighted_layer_pooling(outputs.hidden_states)
        pooled = attention_pooling(hidden, attention_mask)
        return pooled

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_pooled = self.encode_branch(q_input_ids, q_attention_mask, self.q_attention)
        a_pooled = self.encode_branch(a_input_ids, a_attention_mask, self.a_attention)

        q_preds = self.multi_dropout(q_pooled, self.question_head)
        a_preds = self.multi_dropout(a_pooled, self.answer_head)

        combined = torch.cat([q_pooled, a_pooled], dim=-1)
        combined_preds = self.multi_dropout(combined, self.combined_head)

        specialized_preds = torch.cat([q_preds, a_preds], dim=-1)
        final_logits = specialized_preds * 0.5 + combined_preds * 0.5

        return final_logits




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1522702867.py in <cell line: 0>()
----> 1 class AttentionPooling(nn.Module):
      2     """Attention-weighted pooling over sequence dimension."""
      3 
      4     def __init__(self, hidden_size: int):
      5         super().__init__()

NameError: name 'nn' is not defined

## === cell 4
def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, device: torch.device
) -> nn.Module:
    """
    Load model from a checkpoint if it exists; otherwise create a fresh pretrained model.
    Uses a robust loading strategy to avoid protobuf compatibility issues.
    """
    if os.path.exists(checkpoint_path):
        try:
            print(f"Attempting to load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            model = SiameseQuestModel(model_name, pretrained=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(
                f"Failed to load checkpoint ({e}); falling back to pretrained weights."
            )
            model = SiameseQuestModel(model_name, pretrained=True)
    else:
        print("Checkpoint not found – initializing model with pretrained weights only.")
        model = SiameseQuestModel(model_name, pretrained=True)

    model.to(device)
    model.eval()
    return model




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3342174848.py in <cell line: 0>()
      1 def load_model_from_checkpoint(
----> 2     checkpoint_path: str, model_name: str, device: torch.device
      3 ) -> nn.Module:
      4     """
      5     Load model from a checkpoint if it exists; otherwise create a fresh pretrained model.

NameError: name 'torch' is not defined

## === cell 5
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2983688992.py in <cell line: 0>()
      1 print("Loading data...")
----> 2 train_df = pd.read_csv(TRAIN_FILE)
      3 test_df = pd.read_csv(TEST_FILE)
      4 
      5 print(f"Train shape: {train_df.shape}")

NameError: name 'pd' is not defined

## === cell 6
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocab size: {tokenizer.vocab_size}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1208204489.py in <cell line: 0>()
----> 1 print(f"Loading tokenizer: {MODEL_NAME}")
      2 tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
      3 print(f"Vocab size: {tokenizer.vocab_size}")
      4 

NameError: name 'MODEL_NAME' is not defined

## === cell 7
test_dataset = QuestDataset(
    df=test_df,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    is_test=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1001318372.py in <cell line: 0>()
----> 1 test_dataset = QuestDataset(
      2     df=test_df,
      3     tokenizer=tokenizer,
      4     max_length=MAX_LENGTH,
      5     is_test=True,

NameError: name 'QuestDataset' is not defined

## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)

try:
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
    print(f"Predictions shape (model): {predictions.shape}")

except Exception as e:
    print(f"Inference failed ({e}); using mean baseline.")
    mean_vals = train_df[TARGET_COLS].mean().values.astype(np.float32)
    predictions = np.tile(mean_vals, (len(test_df), 1))
    all_qa_ids = test_df["qa_id"].tolist()
    print(f"Predictions shape (baseline): {predictions.shape}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/697055038.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 print(f"Using device: {device}")
      3 
      4 model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)
      5 

NameError: name 'torch' is not defined

## === cell 9
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", [int(x) for x in all_qa_ids])

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

print(f"Submission shape: {submission.shape}")
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path} created!")
submission.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1403909533.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(predictions, columns=TARGET_COLS)
      2 submission.insert(0, "qa_id", [int(x) for x in all_qa_ids])
      3 
      4 for col in TARGET_COLS:
      5     submission[col] = submission[col].clip(0, 1)

NameError: name 'pd' is not defined
