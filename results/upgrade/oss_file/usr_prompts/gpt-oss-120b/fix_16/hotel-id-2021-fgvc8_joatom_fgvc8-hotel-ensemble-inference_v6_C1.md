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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.6125957733434361

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I adjust the script to safely locate any available model files, load only those that exist, and fall back to a simple frequency‑based baseline when no models can be loaded. This prevents the FileNotFoundError, ensures `probs` is defined, and creates a correctly‑formatted `submission.csv` so the notebook runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 0.00209) has done: 'Implemented fixes to resolve the import error, correctly locate test image paths (including their chain directories), and use a chain‑aware frequency baseline when no pretrained models are available. This greatly improves prediction relevance and moves the MAP@5 score toward the target while ensuring a properly formatted `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'The script now speeds up the fallback K‑NN embedding pipeline by replacing the manual Python loops with a batched `torch.utils.data.DataLoader` that loads images in parallel workers, uses `torch.inference_mode()` for faster torch inference, and computes test‑train similarities in manageable batches to avoid building a huge similarity matrix. These changes keep the exact same model architecture, preprocessing, and prediction logic, so the results are unchanged while the runtime stays well under the 600 s limit.'
- What this solution (achieved 0.0) has done: 'I make the script robust to different directory layouts by dynamically locating the data folder and, if no model predictions or image embeddings can be computed, fall back to a simple global‑frequency top‑5 baseline for every test image. This ensures a non‑empty, correctly formatted submission and moves the MAP@5 score away from 0.0 toward the target without altering the core model logic.'
- What this solution (achieved 0.0) has done: 'I add a simple chain‑aware frequency baseline: for each hotel chain we compute its top‑5 most common hotel IDs from the training data and use that list when a test image belongs to a known chain. If the chain is unknown we fall back to the global top‑5 list. This change keeps the overall pipeline intact while providing more relevant predictions, moving the MAP@5 score toward the target.'
- What this solution (achieved 0.0) has done: 'Implemented key performance enhancements while keeping the original algorithm unchanged:

- **GPU‑resident embeddings**: Train and test embeddings are now kept on the GPU instead of being transferred to CPU after each batch. This removes the expensive CPU‑side matrix‑multiply and speeds up the similarity search dramatically.
- **Adjusted similarity computation**: All similarity calculations (`torch.mm` and `topk`) are performed on the GPU tensors, then only the small index tensors are moved to CPU for final list conversion.
- **Minor clean‑up**: Removed unnecessary `.cpu()` calls and added comments clarifying the changes.

These optimizations preserve exact model predictions and ranking logic, reducing runtime well below the 600‑second limit.'
- What this solution (achieved 0.0) has done: 'We add a globally‑computed fallback `baseline_pred` (the global top‑5 hotel IDs) and make sure every row gets a non‑empty prediction by replacing empty strings with this baseline after any model or embedding based logic. This guarantees a valid submission CSV and moves the MAP@5 score away from 0 toward the target without altering the core model pipeline.'
- What this solution (achieved 0.00209) has done: 'I replace the heavy embedding fallback with a simple chain‑aware baseline: for each test image, if its chain is known we output the top‑5 most common hotel IDs for that chain, otherwise we fall back to the global top‑5 list. This keeps the core logic unchanged while providing more relevant predictions than a uniform baseline, moving the MAP@5 score toward the target.'
- What this solution (achieved 0.00209) has done: 'I add a lightweight visual‑similarity fallback that runs when no pretrained fastai models are found.  
It extracts ResNet‑18 features for a random subset of ≈ 20 k training images, then for each test image finds the top‑5 nearest training embeddings and uses their hotel IDs as the prediction list.  
If anything fails it still falls back to the existing chain‑aware/global‑frequency baseline, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.00209) has done: 'The update keeps the same overall workflow (use pretrained fastai models if any are found, otherwise fall back to a visual‑similarity method) but eliminates the costly repeated loading of many large model files and speeds up the fallback embedding computation by using larger batch sizes, more workers, and cuDNN benchmarking. These changes are purely performance‑oriented and do not alter the fundamental prediction logic, ensuring identical behavior when a model is loaded and preserving the same fallback algorithm otherwise.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision.all import *
import dill
from pathlib import Path
from torchvision import models as tv_models
from torchvision import transforms
from PIL import Image



## === cell 1
print(f"fastai {fastai.__version__}", f"torch {torch.__version__}")



## === cell 2
torch.backends.cudnn.benchmark = True  # faster GPU kernels when input size is constant

possible_roots = [
    Path("../input/hotel-id-2021-fgvc8"),
    Path("/kaggle/input/hotel-id-2021-fgvc8"),
    Path("input/hotel-id-2021-fgvc8"),
]
data_root = next((p for p in possible_roots if p.exists()), None)
if data_root is None:
    raise FileNotFoundError("Cannot find the data root for hotel-id-2021-fgvc8")

sample_sub_path = data_root / "sample_submission.csv"
train_csv_path = data_root / "train.csv"
test_images_root = data_root / "test_images"
train_images_root = data_root / "train_images"

submission = pd.read_csv(sample_sub_path)
test = submission.copy()

image_path_map = {p.name: p for p in test_images_root.rglob("*.jpg")}
test["image_path"] = test["image"].map(image_path_map.get)

test["chain"] = test["image_path"].apply(
    lambda p: (
        int(p.parent.name) if (p is not None and p.parent.name.isdigit()) else np.nan
    )
)

if not train_csv_path.exists():
    raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")
train_df = pd.read_csv(train_csv_path)

global_top5 = train_df["hotel_id"].value_counts().index.tolist()[:5]
baseline_pred = " ".join(map(str, global_top5))

chain_top5 = (
    train_df.groupby("chain")["hotel_id"]
    .apply(lambda x: x.value_counts().index[:5].tolist())
    .to_dict()
)
chain_top5 = {float(k): v for k, v in chain_top5.items()}

probs = None
learn = None  # will hold the last successfully loaded learner (if any)

possible_model_paths = [
    "../input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "../input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "../input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
    "../input/fgvc8hotel/export_res101_Fall_5it_4.pkl",
    "../input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",
    "/kaggle/input/fgvc8hotel/export_dn161_Fa_CE_bs32.pkl",
    "/kaggle/input/fgvc8hotel/export_dn161_Fa_FL_bs32.pkl",
    "/kaggle/input/fgvc8hotel/export_res101_Fall_HQAdam.pkl",
    "/kaggle/input/fgvc8hotel/export_res101_Fall_5it_4.pkl",
    "/kaggle/input/hotel-train-fastai-densnet161/export_dn161_kaggle_notebook.pkl",
]
model_files = [Path(p) for p in possible_model_paths if Path(p).exists()]
print(f"Found {len(model_files)} model file(s).")

test_dl = None
for model_path in model_files:
    try:
        learn = load_learner(fname=model_path, cpu=False, pickle_module=dill)
        if test_dl is None:
            test_dl = learn.dls.test_dl(test)
        probs_temp, _ = learn.tta(dl=test_dl, n=5)
        probs = probs_temp if probs is None else probs + probs_temp
        print(f"Loaded predictions from {model_path.name}")
        break
    except Exception as e:
        print(f"Skipping {model_path.name}: {e}")

if probs is None:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = tv_models.resnet50(pretrained=True)
        backbone.fc = torch.nn.Identity()
        backbone = backbone.to(device).eval()

        tfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def get_train_img_path(row):
            try:
                chain_val = int(row["chain"])
            except Exception:
                chain_val = 0
            img_path = train_images_root / str(chain_val) / row["image"]
            return img_path if img_path.exists() else None

        train_df["img_path"] = train_df.apply(get_train_img_path, axis=1)
        train_df_valid = train_df[train_df["img_path"].notnull()].reset_index(drop=True)

        sample_n = min(50000, len(train_df_valid))
        train_sample = train_df_valid.sample(n=sample_n, random_state=42).reset_index(
            drop=True
        )

        class ImagePathDataset(torch.utils.data.Dataset):
            def __init__(self, df, transform):
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img = Image.open(row["img_path"]).convert("RGB")
                return self.transform(img), int(row["hotel_id"])

        train_dataset = ImagePathDataset(train_sample, tfm)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,  # larger batch for speed
            shuffle=False,
            num_workers=4,  # more workers for I/O
            pin_memory=True,
        )

        train_embs = []
        train_ids = []
        with torch.no_grad():
            for imgs, ids in train_loader:
                imgs = imgs.to(device)
                emb = backbone(imgs)
                train_embs.append(emb.cpu())
                train_ids.extend(ids.tolist())
        train_embs = torch.cat(train_embs, dim=0)  # (N, D)

        class TestPathDataset(torch.utils.data.Dataset):
            def __init__(self, df, transform):
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                path = self.df.iloc[idx]["image_path"]
                if path is None:
                    img = Image.new("RGB", (224, 224))
                else:
                    img = Image.open(path).convert("RGB")
                return self.transform(img)

        test_dataset = TestPathDataset(test, tfm)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=128,  # match train batch size
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        preds = []
        train_embs_norm = torch.nn.functional.normalize(train_embs, dim=1)
        with torch.no_grad():
            for batch_imgs in test_loader:
                batch_imgs = batch_imgs.to(device)
                batch_emb = backbone(batch_imgs)
                batch_emb = torch.nn.functional.normalize(batch_emb, dim=1)
                sim = batch_emb @ train_embs_norm.t()  # (B, N)
                topk = sim.topk(5, dim=1).indices  # (B,5)
                for idxs in topk:
                    chosen_ids = [str(train_ids[i]) for i in idxs.tolist()]
                    preds.append(" ".join(chosen_ids))
        if len(preds) != len(test):
            raise RuntimeError("Mismatch in prediction count")
    except Exception as e:
        print(f"Visual‑similarity fallback failed ({e}), using frequency baselines.")
        preds = []
        for _, row in test.iterrows():
            chain_val = row["chain"]
            if not pd.isna(chain_val) and float(chain_val) in chain_top5:
                pred = " ".join(map(str, chain_top5[float(chain_val)]))
            else:
                pred = baseline_pred
            preds.append(pred)
else:
    if isinstance(probs, torch.Tensor) and probs.ndim > 1:
        topk_indices = probs.topk(5, dim=1).indices  # (N,5)
        vocab = learn.dls.vocab if learn is not None else []
        preds = [
            " ".join(map(str, [vocab[i] for i in idx.tolist()])) for idx in topk_indices
        ]
    else:
        preds = [""] * len(submission)

preds = [p if p.strip() else baseline_pred for p in preds]



## === cell 3
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
submission.head()
