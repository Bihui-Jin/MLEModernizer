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

No external packages required in the script and installed.

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

0.7201545002946846

# 6. Current score

0.0014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'I replace the shell‑only execution with a small Python pipeline that (a) installs the required wheels, (b) loads the training metadata to compute the five most frequent hotel IDs, (c) builds a valid `submission.csv` for every test image using those IDs as a simple baseline, and (d) writes the file to `/kaggle/working/submission.csv`. This guarantees a correctly‑formatted submission and moves the score from “no result” toward the target without altering the core model code.'
- What this solution (achieved 0.00209) has done: 'I replace the single‑global top‑5 baseline with a per‑chain top‑5 heuristic: for each hotel chain I compute its five most frequent hotel IDs from the training data and assign that list to every test image that belongs to the same chain (derived from the test image’s folder). If a chain is missing or unknown, the global top‑5 list is used as a fallback. This small change respects the original pipeline while providing a much more relevant prediction, moving the MAP@5 score significantly toward the target.'
- What this solution (achieved 0.00209) has done: 'I make the test‑image‑to‑chain mapping robust by gathering JPEGs from **all** “test_images” folders found under /kaggle/input, instead of only the first match. This ensures each test image gets the correct chain ID (when available) and therefore receives the appropriate per‑chain top‑5 list, moving the MAP@5 score noticeably closer to the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.0014) has done: 'I make the fallback prediction a bit more specific: when a test image’s chain is missing we first try the “chain 0” (unknown‑chain) frequent hotels (if that list exists) before falling back to the overall global top‑5. I also ensure the returned list always contains exactly five distinct IDs by filling any gaps with global IDs while preserving order. This small adjustment keeps the original heuristic unchanged yet should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.00171) has done: 'I adjust the top‑5 prediction logic so that the globally most frequent hotel ID is always placed first, then the chain‑specific frequent IDs (excluding duplicates), and finally fill any remaining slots with the next global frequent IDs. This small change preserves the original heuristic while giving higher weight to the best‑overall guess, which should raise the MAP@5 score toward the target.'
- What this solution (achieved 0.0014) has done: 'I keep the package‑installation block unchanged and rename the cells to start at 1. In the prediction function I switch the ordering so that, when a chain‑specific list exists, it is used first (up to five IDs) and only then falls back to the globally most frequent hotels. This respects the original heuristic while giving the chain information higher priority, which should raise MAP@5 toward the target. The rest of the pipeline (reading data, mapping images to chains, writing submission.csv) stays the same.'
- What this solution (achieved 0.0014) has done: 'I tighten the test‑image‑to‑chain mapping so that the chain ID is correctly extracted even when the folder hierarchy is deeper, and I enlarge the global pool of frequent hotels (top 100) to give a richer fallback list while still preferring chain‑specific choices. These minimal adjustments keep the original heuristic intact but increase the chance that the true hotel appears among the five predicted IDs, moving the MAP@5 score closer to the target.'

# 9. Code solution

## === cell 0
import subprocess, sys, os, glob

deps_path = "/kaggle/input/pekolib-deps"
if os.path.isdir(deps_path):
    for whl in glob.glob(os.path.join(deps_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

pk_path = "/kaggle/input/pekolib"
if os.path.isdir(pk_path):
    for whl in glob.glob(os.path.join(pk_path, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])

abn_src = "/kaggle/input/pekolib-deps/inplace_abn-1.0.12/inplace_abn"
if os.path.isdir(abn_src):
    tmp_dir = "/tmp/inplace_abn"
    subprocess.run(["cp", "-r", abn_src, tmp_dir])
    subprocess.run(
        ["bash", "-c", f"cd {tmp_dir} && {sys.executable} -m pip install -q ."]
    )

print("Package installation completed.")




## === cell 1
import pandas as pd
from pathlib import Path

train_path = next(Path("/kaggle/input").rglob("train.csv"))
sample_sub_path = next(Path("/kaggle/input").rglob("sample_submission.csv"))

train_df = pd.read_csv(train_path)

global_top5 = train_df["hotel_id"].value_counts().nlargest(5).index.astype(str).tolist()
global_top100 = (
    train_df["hotel_id"].value_counts().nlargest(100).index.astype(str).tolist()
)

per_chain_top5 = (
    train_df.groupby(train_df["chain"].astype(str))["hotel_id"]
    .apply(lambda s: s.value_counts().nlargest(5).index.astype(str).tolist())
    .to_dict()
)

unknown_chain_fallback = per_chain_top5.get("0", [])

img_to_chain = {}
test_root_candidates = list(Path("/kaggle/input").rglob("test_images"))
if not test_root_candidates:
    raise FileNotFoundError("Cannot locate any test_images folder in /kaggle/input")

for root in test_root_candidates:
    for img_path in root.rglob("*.jpg"):
        img_name = img_path.name
        rel_parts = img_path.relative_to(root).parts
        chain_id = rel_parts[0] if len(rel_parts) >= 2 else None
        img_to_chain[img_name] = chain_id

sample_sub = pd.read_csv(sample_sub_path)


def get_top5_for_image(img_name):
    """
    Return a space‑delimited string of 5 hotel IDs.
    Priority:
      1. Chain‑specific frequent IDs (if known)
      2. Global most‑frequent IDs (top‑5)
      3. Additional global frequent IDs (top‑100) to fill any gaps
    Duplicates are avoided.
    """
    prediction = []

    chain_id = img_to_chain.get(img_name)
    chain_tops = per_chain_top5.get(chain_id) if chain_id is not None else None
    if chain_tops is None:
        chain_tops = unknown_chain_fallback

    if chain_tops:
        for h in chain_tops:
            if h not in prediction:
                prediction.append(h)
            if len(prediction) == 5:
                break

    if len(prediction) < 5:
        for h in global_top5:
            if h not in prediction:
                prediction.append(h)
            if len(prediction) == 5:
                break

    if len(prediction) < 5:
        for h in global_top100:
            if h not in prediction:
                prediction.append(h)
            if len(prediction) == 5:
                break

    return " ".join(prediction)


submission = pd.DataFrame(
    {
        "image": sample_sub["image"],
        "hotel_id": sample_sub["image"].apply(get_top5_for_image),
    }
)

output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
print("Sample rows:")
print(submission.head())
