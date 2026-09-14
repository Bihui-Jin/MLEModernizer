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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.38103

# 6. Current score

0.6874

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67075) has done: 'I update the notebook from deprecated fastai v0.7 imports/APIs to the installed fastai v2 equivalents so it actually runs in this environment. I keep the same core approach (ResNet101 transfer learning, image resizing/augmentation, fit, then test-time augmentation and averaged probabilities) while fixing dataset paths, CUDA handling, and submission-column alignment to `sample_submission.csv`. I also remove the unsafe symlink/cleanup logic that’s causing filesystem errors and instead read images directly from the provided Kaggle input folders. Finally, I guarantee that a valid `submission.csv` is written with the exact required header/order.'
- What this solution (achieved 0.70857) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end and reliably writes a valid `submission.csv`. The main issues are (1) `StratifiedSplitter` isn’t available in fastai v2, so I replace it with an equivalent stratified split built via `sklearn.model_selection.StratifiedShuffleSplit`, and (2) your `sub` variable is being shadowed by fastai’s `sub()` function, causing `sub.to_csv` to fail—so I rename the DataFrame to `sub_df`. These are correctness/stability fixes and preserve your core model/training/tta approach. I also keep the submission column order aligned to `sample_submission.csv` and ensure probability normalization/clipping for log-loss safety.'
- What this solution (achieved 0.64242) has done: 'Your score is much worse than the target (0.70857 vs 0.38103, lower is better), so the smallest safe way to improve toward the target is to reduce overfitting and better match the competition’s log-loss objective without changing the overall approach. I keep the same ResNet101 + fastai `fine_tune` + TTA pipeline, but (1) switch the reported metric from accuracy to `error_rate` + `RocAuc` is not appropriate for multiclass logloss; instead we add `cross_entropy`-aligned monitoring via `loss` and enable `label_smoothing` in the loss (minimal change, same semantics: multiclass classification). I also (2) use fastai’s built-in `RandomSplitter` replacement with stratification already done, but keep your stratified split; the key improvement is (3) to increase input resolution slightly (224→299) which typically improves dog-breed performance with minimal code changes, while keeping the same model and training loop. Finally, I keep submission alignment exactly to `sample_submission.csv` and maintain probability normalization/clipping for log-loss safety.'
- What this solution (achieved 0.64152) has done: 'Your current logloss is far above the target (0.64242 vs 0.38103, lower is better), so the smallest legitimate way to move toward the target is to improve generalization without changing the model/training loop structure. I keep the same ResNet101 + fastai `fine_tune` + TTA pipeline, but fix two score-relevant details: (1) use ImageNet normalization (`Normalize.from_stats(*imagenet_stats)`) which materially improves calibration/logloss for pretrained ResNets, and (2) switch `Resize` to a standard transfer-learning setup (`Resize(460)` then `RandomResizedCrop(sz)` in `batch_tfms`) to reduce train/valid distribution mismatch at inference resolution while preserving the same augmentation approach. Everything else (split logic, architecture, epochs, TTA, submission alignment/clipping) stays the same and it still write a valid `submission.csv`.'
- What this solution (achieved 0.65789) has done: 'Your current logloss (0.64152, lower is better) is still far from the target (0.38103), so the smallest safe way to move closer without changing the core ResNet101 + fine_tune + TTA approach is to improve probability calibration and reduce overconfident errors. I keep the exact same data pipeline and training loop, but (1) use fastai’s `learn.tta(..., use_max=False)` to average predictions across augmentations (more stable for logloss than max), and (2) apply a very light “uniform prior” probability blending after TTA to reduce extreme probabilities that hurt logloss. These are minimal post-processing/calibration changes that preserve evaluation semantics (still a valid probability distribution per row) and typically lower logloss. Submission alignment to `sample_submission.csv` and CSV writing are kept unchanged.'
- What this solution (achieved 0.80921) has done: 'You’re still far above the target logloss (0.65789 vs 0.38103, lower is better), so we should make the smallest changes that legitimately improve calibration/generalization without changing the ResNet101 + fine_tune + TTA core pipeline. I keep your architecture, training call, and TTA exactly the same, but (1) switch the training metric to `accuracy` (metrics don’t affect optimization, but `error_rate` can confuse monitoring) and (2) replace the fixed uniform blending with a slightly stronger, class-prior blending computed from the training labels, which is usually better for logloss than a uniform prior. I also add a single lightweight temperature-scaling step on probabilities (post-processing only) to reduce overconfidence; it preserves valid probability semantics and often reduces logloss. Submission formatting, column alignment to `sample_submission.csv`, and CSV writing stay unchanged.'
- What this solution (achieved 0.62538) has done: 'Your current logloss (0.80921, lower is better) is far above the target (0.38103), so we should make a small, score-relevant calibration fix rather than changing the model/training pipeline. The biggest risk in your current post-processing is that the extra “prior blending” plus “temperature scaling” can be mis-tuned and *increase* logloss; we keep the same TTA predictions but (1) tune the blend strength down and (2) choose temperature automatically from the validation set (using the existing stratified split) to reduce overconfidence in a metric-aligned way. This preserves core logic (ResNet101 + fine_tune + TTA) and only adjusts probability calibration/post-processing to better match multiclass logloss. Submission formatting and column alignment to `sample_submission.csv` remain identical.'
- What this solution (achieved 0.62206) has done: 'I keep your ResNet101 + fine_tune + TTA pipeline intact and only adjust the score-relevant post-processing that can materially reduce multiclass logloss. Specifically, I (1) compute the class-prior and the temperature on the validation set using the *same* probability pipeline you apply at test time, and (2) expand the (alpha, temperature) search a bit on the validation set so calibration is less likely to be mis-tuned and hurt logloss. This is a minimal change (no architecture/training/augmentation changes), but it directly targets logloss by reducing overconfident wrong predictions. The submission formatting, column alignment to `sample_submission.csv`, and CSV writing remain identical.'
- What this solution (achieved 0.62366) has done: 'We’re far above the target logloss (0.62206 vs 0.38103, lower is better), so we should make the smallest changes that legitimately improve probability quality without changing your core ResNet101 + fine_tune + TTA pipeline. The highest-impact minimal fix here is to calibrate using the *same validation-set object you already have* but without applying random test-time augmentation on validation during calibration, which adds noise and can mis-tune (alpha, T) for logloss; we instead use deterministic predictions (`learn.get_preds` on the valid_dl) for stable calibration. Then we apply the chosen (alpha, T) to the TTA test probabilities exactly as before. This preserves architecture, training loop, and inference approach, but makes the calibration step better aligned and more reliable for logloss.'
- What this solution (achieved 0.62366) has done: 'I keep your ResNet101 + fine_tune + TTA pipeline unchanged and only adjust the post-processing calibration step that directly affects multiclass log-loss. Right now, your temperature scaling is applied by re-softmaxing `log(p)/T`, which is not the standard/logit-consistent operation and can worsen calibration; I switch it to a probability-power transform `p^(1/T)` which is the correct mapping when only probabilities are available. I also ensure the validation calibration uses the same transform, and I slightly widen the (alpha, T) grid in a controlled way to better find a nearby optimum without changing training. Everything else (data, split, augmentation, training epochs, TTA settings, submission alignment/format) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.69813) has done: 'Your current gap to the target logloss is large (0.62366 vs 0.38103; lower is better), so we should make a minimal, metric-aligned improvement without changing the ResNet101 + fine_tune + TTA core pipeline. The biggest likely score drag in your current setup is that you’re applying label smoothing during training (which changes the optimized objective) and then separately trying to calibrate probabilities; for Kaggle logloss, it’s usually safer to train with plain cross-entropy and keep calibration as the only “softening” step. I therefore remove label smoothing (use standard `CrossEntropyLossFlat`) while keeping the same model, data, training call, TTA, and calibration grid search logic intact. Additionally, I ensure the vocabulary/order used for predictions exactly matches the `sample_submission.csv` breed columns (same set, same order), which can reduce any subtle class-order mismatch risk.'
- What this solution (achieved 0.6874) has done: 'Your current logloss is far worse than the target (0.69813 vs 0.38103; lower is better), so we make the smallest changes that legitimately improve probability quality without changing the core ResNet101 + fine_tune + TTA pipeline. The biggest score drag in your current code is that you’re applying stochastic test-time augmentation to the test set but calibrating (alpha, T) on *non-TTA* validation probabilities, which mismatches the distribution and can mis-tune calibration for logloss. I calibrate (alpha, T) on validation predictions produced via the same `learn.tta(..., use_max=False)` pipeline (with a small `n` for runtime), then apply the chosen parameters to test TTA probabilities exactly as before. Everything else (data, split, model, training call, submission formatting/column order, probability clipping/normalization) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.6874) has done: 'We’re far from the target (0.6874 vs 0.38103, lower is better), so the smallest score-relevant change is to remove the calibration steps that are likely hurting logloss (prior blending + temperature scaling), while keeping your core ResNet101 + fine_tune + TTA pipeline unchanged. We still compute validation logloss for visibility, but we default to “no calibration” (alpha=0, T=1) unless the validation grid search clearly improves over raw TTA by a small margin, to avoid overfitting calibration to the split. This preserves identical model/training/inference logic (still TTA probabilities), just makes post-processing more conservative and typically improves multiclass logloss. Submission formatting/column alignment remains exactly tied to `sample_submission.csv` and we still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
LABELS_CSV = os.path.join(BASE_INPUT, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Exists TRAIN_DIR:", os.path.exists(TRAIN_DIR))
print("Exists TEST_DIR :", os.path.exists(TEST_DIR))
print("Exists LABELS_CSV:", os.path.exists(LABELS_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print(
    "Train images:", len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else "NA"
)
print("Test images :", len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else "NA")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print(labels_df.head())
print(sample_sub.head())
print("Num columns (sample submission):", sample_sub.shape[1])



## === cell 1
from fastai.vision.all import *
import torch
from sklearn.model_selection import StratifiedShuffleSplit

set_seed(42, reproducible=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "| device:", device)



## === cell 2
sz = 299
bs = 128
arch = resnet101

assert labels_df["id"].is_unique, "labels.csv ids must be unique"

sample_cols = list(sample_sub.columns)
sample_breeds = sample_cols[1:]
breed_set = set(labels_df["breed"].unique().tolist())

extra_cols = [c for c in sample_breeds if c not in breed_set]
missing_cols = [c for c in breed_set if c not in set(sample_breeds)]

print("Extra non-breed columns in sample_submission:", extra_cols)
print("Breed columns missing from sample_submission:", missing_cols)
assert (
    len(missing_cols) == 0
), "sample_submission is missing some breed columns; cannot align safely."
assert (
    len(extra_cols) == 0
), "sample_submission has unexpected extra prediction columns; cannot align safely."

breed_vocab = list(sample_breeds)
print("Num breed classes (from sample_submission):", len(breed_vocab))



## === cell 3
train_df = labels_df.copy()
train_df["fname"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, f"{x}.jpg"))

missing = train_df.loc[~train_df["fname"].map(os.path.exists)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} train image files, e.g. {missing.iloc[0]['fname']}"
    )

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idxs = np.arange(len(train_df))
y = train_df["breed"].values
train_idx, valid_idx = next(sss.split(idxs, y))
splitter = IndexSplitter(valid_idx)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=breed_vocab)),
    get_x=ColReader("fname"),
    get_y=ColReader("breed"),
    splitter=splitter,
    item_tfms=Resize(460),
    batch_tfms=[
        *aug_transforms(size=sz, max_zoom=1.1),
        Normalize.from_stats(*imagenet_stats),
    ],
)

dls = dblock.dataloaders(train_df, bs=bs, shuffle=True)

print("dls vocab size:", len(dls.vocab))
print("Example vocab head:", dls.vocab[:5])



## === cell 4
learn = vision_learner(
    dls,
    arch,
    metrics=[accuracy],
    loss_func=CrossEntropyLossFlat(),
).to_fp32()

lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep), show_plot=False)
lr = float(lr_min) if lr_min is not None else 1e-2
print("Suggested lr (minimum):", lr)



## === cell 5
learn.fine_tune(5, base_lr=lr)




## === cell 6
def _apply_temperature_prob_power(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    invT = 1.0 / float(T)
    pT = np.power(p, invT)
    pT = np.clip(pT, 1e-12, 1.0)
    pT = pT / pT.sum(axis=1, keepdims=True)
    return pT


def _logloss_from_probs(p: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx])))


valid_breeds = train_df.iloc[valid_idx]["breed"].tolist()
y_valid_idx = np.array([dls.vocab.o2i[b] for b in valid_breeds], dtype=np.int64)

valid_probs_tta_t, _ = learn.tta(dl=learn.dls.valid, n=4, beta=0.0, use_max=False)
valid_probs = valid_probs_tta_t.cpu().numpy()
valid_probs = np.clip(valid_probs, 1e-12, 1.0)
valid_probs = valid_probs / valid_probs.sum(axis=1, keepdims=True)

assert valid_probs.shape[0] == len(
    y_valid_idx
), "Valid preds count mismatch; split/order issue."

raw_ll = _logloss_from_probs(valid_probs, y_valid_idx)
print("Raw valid logloss (TTA probs, no calibration):", raw_ll)

vc = train_df["breed"].value_counts()
prior_vec = np.array([vc.get(c, 0) for c in dls.vocab], dtype=np.float64)
prior_vec = prior_vec / prior_vec.sum()
prior_vec = np.clip(prior_vec, 1e-12, 1.0)
prior_vec = prior_vec / prior_vec.sum()
prior = np.broadcast_to(prior_vec, valid_probs.shape)

alphas = np.array([0.00, 0.005, 0.01, 0.02, 0.03, 0.05], dtype=np.float64)
Ts = np.array([0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.60], dtype=np.float64)

best = (0.0, 1.0, raw_ll)  # start from no-calibration baseline
grid_losses = {(0.0, 1.0): raw_ll}

for a in alphas:
    blended = (1.0 - float(a)) * valid_probs + float(a) * prior
    blended = np.clip(blended, 1e-12, 1.0)
    blended = blended / blended.sum(axis=1, keepdims=True)
    for T in Ts:
        pT = _apply_temperature_prob_power(blended, float(T))
        ll = _logloss_from_probs(pT, y_valid_idx)
        grid_losses[(float(a), float(T))] = float(ll)
        if ll < best[2]:
            best = (float(a), float(T), float(ll))

best_alpha, best_T, best_ll = best
print("Best (alpha, T) on valid:", (best_alpha, best_T), "| valid logloss:", best_ll)

min_improve = 0.002
if best_ll <= raw_ll - min_improve:
    chosen_alpha, chosen_T = best_alpha, best_T
    print("Using calibrated params (pass margin):", (chosen_alpha, chosen_T))
else:
    chosen_alpha, chosen_T = 0.0, 1.0
    print(
        "Calibration not used (did not beat raw by margin). Using:",
        (chosen_alpha, chosen_T),
    )

top5 = sorted(grid_losses.items(), key=lambda kv: kv[1])[:5]
print("Top-5 (alpha,T)->logloss:", {k: v for k, v in top5})



## === cell 7
test_ids = sample_sub["id"].tolist()
test_files = [os.path.join(TEST_DIR, f"{i}.jpg") for i in test_ids]

missing_test = [p for p in test_files if not os.path.exists(p)]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test image files, e.g. {missing_test[0]}"
    )

test_dl = learn.dls.test_dl(test_files, with_labels=False)



## === cell 8
tta_preds, _ = learn.tta(dl=test_dl, n=4, beta=0.0, use_max=False)
probs = tta_preds.cpu().numpy()
print("probs shape:", probs.shape)



## === cell 9
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

vc = train_df["breed"].value_counts()
prior = np.array([vc.get(c, 0) for c in dls.vocab], dtype=np.float64)
prior = prior / prior.sum()
prior = np.clip(prior, 1e-12, 1.0)
prior = prior / prior.sum()
prior = np.broadcast_to(prior, probs.shape)

probs = (1.0 - float(chosen_alpha)) * probs + float(chosen_alpha) * prior
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

probs = _apply_temperature_prob_power(probs, float(chosen_T))

sub_df = pd.DataFrame(probs, columns=dls.vocab)
sub_df.insert(0, "id", test_ids)

sub_df = sub_df[sample_cols]

p = sub_df.iloc[:, 1:].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-12, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sub_df.iloc[:, 1:] = p

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print(
    "Columns match sample submission:", list(sub_df.columns) == list(sample_sub.columns)
)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "| bytes:", os.path.getsize(out_path))
print("First row prob sum:", float(sub_df.iloc[0, 1:].sum()))
