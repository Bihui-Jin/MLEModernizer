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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
transformers==4.53.3

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

0.6518295764981217

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I wrap the model and tokenizer loading in a try‑except block so that if the custom local paths are unavailable (which caused the protobuf `MessageFactory` error) the code falls back to the standard HuggingFace `bert-base-uncased` model. I also clamp the rounded predictions to the valid score range [1, 6] and ensure the submission CSV is written with the correct column order. This fixes the runtime errors, guarantees a valid `.csv` file, and makes only minimal, score‑neutral changes.'
- What this solution (achieved 0.0) has done: 'I wrapped the transformer imports in a safe try‑except block and mapped any failure to the generic `AutoTokenizer`/`AutoModel` classes, ensuring the code can load a tokenizer and model even when the protobuf version causes the original `BertTokenizer`/`BertModel` import to fail. This eliminates the “MessageFactory” error, allowing the script to run end‑to‑end and produce a valid `submission.csv` while keeping the original BERT‑based regression logic intact.'
- What this solution (achieved 0.0) has done: 'Implemented an environment fix to avoid the protobuf import error and added a safe fallback prediction using the training set’s average score. The script now sets the protobuf implementation early, loads the training data to compute a mean‑based constant prediction when a fine‑tuned model isn’t available, and writes a correctly formatted CSV submission.'
- What this solution (achieved 0.54286) has done: 'The changes move the BERT model and tensors onto GPU when available, increase batch size, enable pinned memory, and pre‑tokenize all texts once in `EssayDataset` to avoid repeated Python‑level tokenization. These adjustments keep every mathematical operation identical, only accelerating the data‑loading and forward‑pass steps, so the predictions remain unchanged while fitting comfortably inside the 600 s limit.'

# 9. Code solution

## === cell 0
class EssayDataset(Dataset):
    """Custom Dataset for essay texts with pre‑tokenized tensors."""

    def __init__(self, tokenizer, texts, max_length):
        encodings = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class BertRegressor(nn.Module):
    """BERT based regressor (used only for loading fine‑tuned weights)."""

    def __init__(self, pretrained_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.out(pooled)


LOCAL_MODEL_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_model"
MODEL_WEIGHTS_PATH = (
    "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor/bert_regressor.pth"
)
TOKENIZER_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_tokenizer"

BATCH_SIZE = 32  # larger batch size for better GPU utilization
MAX_LEN = 256

try:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    logging.info("Loaded tokenizer from local path.")
except Exception as e:
    logging.warning(f"Local tokenizer load failed ({e}); using 'bert-base-uncased'.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_available = True
fine_tuned_available = False

try:
    model = BertRegressor(LOCAL_MODEL_PATH)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    logging.info("Loaded fine‑tuned model weights.")
    fine_tuned_available = True
except Exception as e:
    logging.warning(f"BERT model load failed ({e}); will use TF‑IDF fallback.")
    bert_available = False
    model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if bert_available and model is not None:
    model.to(device)
    model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1205039817.py in <cell line: 0>()
----> 1 class EssayDataset(Dataset):
      2     """Custom Dataset for essay texts with pre‑tokenized tensors."""
      3 
      4     def __init__(self, tokenizer, texts, max_length):
      5         encodings = tokenizer.batch_encode_plus(

NameError: name 'Dataset' is not defined

## === cell 1
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)

baseline_score = int(round(train_df["score"].mean()))
baseline_score = np.clip(baseline_score, 1, 6)


def get_embeddings(loader):
    """Compute BERT pooled embeddings using the chosen device."""
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.pooler_output  # (batch, hidden)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


if bert_available:
    test_dataset = EssayDataset(tokenizer, test_df["full_text"].tolist(), MAX_LEN)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    if not fine_tuned_available:
        SAMPLE_SIZE = 20000
        if len(train_df) > SAMPLE_SIZE:
            train_subset = train_df.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            train_subset = train_df

        train_dataset = EssayDataset(
            tokenizer, train_subset["full_text"].tolist(), MAX_LEN
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
        )

        logging.info("Computing BERT embeddings for the training subset...")
        train_embeddings = get_embeddings(train_loader)
        train_scores = train_subset["score"].values.astype(np.float32)

        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(train_embeddings, train_scores)

        logging.info("Computing BERT embeddings for the test set...")
        test_embeddings = get_embeddings(test_loader)
        preds = ridge.predict(test_embeddings)
    else:
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch_device)  # shape: (batch, 1)
                preds.extend(out.squeeze().tolist())
        preds = np.array(preds)

    preds = np.round(preds).astype(int)
    preds = np.clip(preds, 1, 6)

else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    logging.info("Using TF‑IDF + Ridge regression as BERT fallback.")

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf.fit(train_df["full_text"].tolist())

    train_tfidf = tfidf.transform(train_df["full_text"])
    ridge = Ridge(alpha=1.0, random_state=42, solver="lsqr")
    ridge.fit(train_tfidf, train_df["score"])

    test_tfidf = tfidf.transform(test_df["full_text"])
    preds = ridge.predict(test_tfidf)
    preds = np.round(preds).astype(int)
    preds = np.clip(preds, 1, 6)


test_df["score"] = preds
submission = test_df[["essay_id", "score"]]

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2359965236.py in <cell line: 0>()
      2 train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
      3 
----> 4 test_df = pd.read_csv(test_path)
      5 train_df = pd.read_csv(train_path)
      6 

NameError: name 'pd' is not defined
