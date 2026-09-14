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
Identify contrails in satellite imagery.

## Metric
Global Dice coefficient. The Dice coefficient formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where X is the entire set of predicted contrail pixels for all observations in the test data and Y is the ground truth set of all contrail pixels in the test data.

## Submission Format
Use run-length encoding. For example, '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

Use a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

Empty predictions must be marked with '-' in the submission file.

The file should contain a header and have the following format:

```
record_id,encoded_pixels  
1000834164244036115,1 1 5 1  
1002653297254493116,-  
etc.
```

## Dataset 
Some key labeling guidance:
- Contrails must contain at least 10 pixels
- At some time in their life, Contrails must be at least 3x longer than they are wide
- Contrails must either appear suddenly or enter from the sides of the image
- Contrails should be visible in at least two image

A sequence of images at 10-minute intervals are provided. Each example (`record_id`) contains exactly one labeled frame.

- **train/** - the training set; each folder represents a `record_id` and contains the following data:
    - **band_{08-16}.npy**: array with size of `H x W x T`, where `T = n_times_before + n_times_after + 1`, representing the number of images in the sequence. There are `n_times_before` and `n_times_after` images before and after the labeled frame respectively. In our dataset all examples have `n_times_before=4` and `n_times_after=3`. Each band represents an infrared channel at different wavelengths and is converted to brightness temperatures based on the calibration parameters. The number in the filename corresponds to the GOES-16 ABI band number. Details of the ABI bands can be found [here](https://www.goes-r.gov/mission/ABI-bands-quick-info.html).
    - **human_individual_masks.npy**: array with size of `H x W x 1 x R`. Each example is labeled by `R` individual human labelers. `R` is not the same for all samples. The labeled masks have value either 0 or 1 and correspond to the `(n_times_before+1)`-th image in `band_{08-16}.npy`. They are available only in the training set.
    - **human_pixel_masks.npy**: array with size of `H x W x 1` containing the binary ground truth. A pixel is regarded as contrail pixel in evaluation if it is labeled as contrail by more than half of the labelers.
- **validation/** - the same as the training set, without the individual label annotations; it is permitted to use this as training data if desired
- **test/** - the test set; your objective is to identify contrails found in these records.
- **{train|validation}_metadata.json** - metadata information for each record; contains the timestamps and the projection parameters to reproduce the satellite images.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

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
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        input/
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        working/
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
```

-> data/google-research-identify-contrails-reduce-global-warming/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/google-research-identify-contrails-reduce-global-warming/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/google-research-identify-contrails-reduce-global-warming/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> (stopped after 10 files for performance)

# 5. Target score

0.00344

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the submission generation deterministic and correctly aligned to the competition’s required `record_id` order by using `sample_submission.csv` as the authoritative index instead of filesystem glob order (which can silently shuffle rows and hurt/void scoring). I also ensure `record_id` is written as a string (no unintended integer formatting) and that the output CSV contains exactly the required columns. Since your current approach predicts an all-ones mask for every test image (very likely to score extremely poorly on global Dice), I switch to the minimal “all-empty masks” baseline (`-`), which typically improves Dice substantially over predicting everything as contrail, while preserving the same overall submission-building logic (RLE utilities remain intact). The script still run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a submission-format issue (most commonly: wrong `record_id` dtype/order, extra columns, or an RLE that doesn’t match the required pixel-ordering/uniqueness rules). I keep your “all-empty mask” baseline (which should score > 0 in this competition) but make the submission generation stricter: use `sample_submission.csv` as the authoritative row order, write `record_id` as a string, and emit exactly the two required columns. I also normalize the encoded string to exactly `'-'` for empty masks and add a lightweight validation to catch accidental format violations before writing the CSV.'
- What this solution (achieved 0.0) has done: 'Your current script produces an all-empty submission, which should generally score above 0, so a 0.0 strongly suggests an evaluation mismatch caused by submission schema (extra columns in the sample, dtype/order issues) or a strict format requirement not being met. I keep the same “all-empty masks” core logic (no model/feature changes) but (1) read the sample submission robustly from either location, (2) write `record_id` as plain strings (not pandas `string` dtype) to avoid any Kaggle-side parsing quirks, (3) ensure the output has exactly the two required columns and exactly the same `record_id` order as the sample, and (4) add a hard validation that every `record_id` is unique/non-null and every `encoded_pixels` is either `'-'` or a valid RLE. These are minimal changes aimed at eliminating format-related 0.0 and moving the score toward your target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests your submission is being treated as invalid or mis-parsed, because an all-empty (`'-'`) submission in this competition should typically yield a small but non-zero Dice. I keep your all-empty baseline (so we don’t change core prediction logic) and make the output schema match Kaggle’s expected format more strictly: use the sample’s `record_id` order, output **exactly** the two required columns, and normalize `record_id` to plain Python strings. I also add a pre-write validation that the submission row count and `record_id` set exactly match the sample to prevent silent mismatches that can lead to 0.0. These are minimal, format-focused changes intended to move the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your current “all-empty masks” baseline should score non-zero, so the 0.0 is most likely coming from a strict submission-schema mismatch rather than model quality. I make the output match the competition’s required format exactly by (1) reading `record_id` as string consistently, (2) writing **only** `record_id,encoded_pixels` (dropping `height/width` even if present in the sample), (3) enforcing exact row-order alignment to the sample, and (4) adding a final, hard pre-write check that the saved CSV re-loads identically (catching any dtype/format surprises). These changes keep your core “all-empty prediction” logic intact while eliminating the common causes of silent-invalid submissions that yield 0.0, pushing the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your current all-empty submission should score slightly above 0, so a 0.0 strongly suggests the evaluator is not matching your rows to the correct `record_id`s or is failing to parse the IDs (most commonly due to scientific-notation / precision loss when `record_id` is treated as numeric). I make the `record_id` handling fully lossless by reading it strictly as string, normalizing it, and then writing the CSV with explicit CSV quoting to prevent any accidental numeric coercion. I also add a hard “exact match” check against the sample submission’s `record_id` list (same length, same order, same values) before writing, because any mismatch can silently yield a 0.0. Core prediction logic remains identical: still an all-empty `'-'` baseline, only strengthening schema correctness to move your score upward toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score with an all-`'-'` submission strongly points to a schema mismatch with what Kaggle’s evaluator expects, not a “model quality” issue. The most likely culprit here is that your `sample_submission.csv` in this environment contains extra columns (`height`, `width`), and Kaggle’s ingestion can be strict about matching the sample’s column set/order for that specific competition snapshot. I keep the same core “all-empty mask” logic, but write the submission using the sample’s exact columns (filling `height/width` from the sample) while still keeping `record_id` order identical and `encoded_pixels` normalized to `'-'`. This is the smallest change that plausibly converts a silently-invalid submission (scoring 0.0) into a valid one, moving the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 with an all-`'-'` submission strongly suggests Kaggle is treating the file as malformed/unsupported rather than “bad predictions.” The most common remaining cause here is a strict schema mismatch: your environment’s sample has 4 columns (`record_id, encoded_pixels, height, width`), but the competition’s evaluator may require **exactly two columns** (`record_id, encoded_pixels`). I keep your core “all-empty baseline” logic (so we don’t change modeling semantics), but write a submission with exactly those two columns, enforce `record_id` as lossless strings, and keep the row order identical to the sample. This is the smallest change likely to convert a 0.0 into a small non-zero Dice, moving toward 0.00344.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a submission ingestion mismatch (not with “all-empty” predictions), so the smallest score-improving change is to mirror the provided sample’s schema exactly instead of forcing 2 columns. In your environment the sample has 4 columns (`record_id, encoded_pixels, height, width`), so we keep your all-empty baseline but write a submission with the same columns and the same row order, copying `height/width` from the sample. This preserves your core prediction logic (all `'-'`) while removing the most likely remaining cause of a silently-invalid submission that yields 0.0, pushing the score upward toward the target. We also keep the strict pre-write and post-write validations, updated to validate the 4-column schema.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly not from “all-empty masks” (which should usually yield a small non-zero Dice), but from the evaluator failing to match/parse rows—most commonly because `record_id` gets written in a way Kaggle reads as numeric/scientific notation or loses precision. I make the `record_id` handling fully lossless by enforcing pure string I/O end-to-end (read as `str`, write with explicit quoting for the `record_id` field), while still keeping your core logic identical (all `'-'` predictions, same ordering as the sample). I also add a strict post-write check that reloads the CSV with `dtype=str` and verifies the reloaded `record_id` values match the sample exactly, catching the last common cause of silent 0.0. The submission still mirror the sample schema (including `height/width` if present) to avoid schema-strict ingestion issues.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 with an all-`'-'` baseline is most consistent with a subtle ingestion issue, not with “bad predictions”, so the smallest score-improving change is to make the output schema and CSV formatting as Kaggle-friendly as possible. I keep the core “all-empty” prediction logic identical, but (1) ensure we write **exactly** the two required columns (`record_id,encoded_pixels`) while still validating against the sample’s `record_id` order, and (2) remove `QUOTE_NONNUMERIC` so the header/fields are written in the most standard way (Kaggle often expects plain CSV without unusual quoting). I also add a strict check that `record_id` strings contain only digits (no hidden `.0`/scientific notation artifacts) before writing. These are minimal, format-focused changes aimed at converting the submission from “silently invalid => 0.0” to a small non-zero Dice closer to your 0.00344 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 with an all-`'-'` baseline is most consistent with Kaggle ingestion/schema mismatch rather than “bad predictions,” especially since your environment’s `sample_submission.csv` clearly has 4 columns (`record_id, encoded_pixels, height, width`). The smallest change to move the score up toward 0.00344 is to mirror the sample’s exact column set and order, while keeping the same all-empty prediction logic and the same `record_id` order. I keep your strict validations, but update them to validate and write the 4-column schema and re-load-check it identically. This should convert a silently-invalid submission into a valid one without changing the prediction semantics.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
from pathlib import Path
import csv




## === cell 1
def rle_encode(x: np.ndarray, fg_val: int = 1):
    """
    Run-length encoding following the competition's convention:
    pixels are counted top-to-bottom then left-to-right.
    Using x.T.flatten() achieves that ordering for a (H, W) array.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {x.shape}")
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """Converts list to a string representation. Empty list returns '-'."""
    if x:
        return " ".join(map(str, x))
    return "-"


def validate_rle_string(s: str):
    """
    Minimal sanity checks to avoid invalid submissions that can score 0.0:
    - Empty must be exactly '-'
    - Otherwise must be even count of positive integers, strictly increasing starts,
      and lengths positive.
    """
    if s == "-":
        return True
    parts = s.split()
    if len(parts) % 2 != 0:
        return False
    try:
        nums = list(map(int, parts))
    except Exception:
        return False
    if any(n <= 0 for n in nums):
        return False
    starts = nums[0::2]
    lens = nums[1::2]
    if any(l <= 0 for l in lens):
        return False
    if any(starts[i] <= starts[i - 1] for i in range(1, len(starts))):
        return False
    return True


def read_sample_submission():
    candidates = [
        Path(
            "/kaggle/input/google-research-identify-contrails-reduce-global-warming/sample_submission.csv"
        ),
        Path("/kaggle/input/sample_submission.csv"),
        Path(
            "/kaggle/data/google-research-identify-contrails-reduce-global-warming/sample_submission.csv"
        ),
        Path("/kaggle/data/sample_submission.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, dtype={"record_id": str}, keep_default_na=False)
            return df, p
    raise FileNotFoundError(f"Could not find sample_submission.csv in: {candidates}")


def build_submission_all_empty_match_sample_schema(
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core logic unchanged: predicts all-empty masks ('-') in sample order.

    Change (score-related): mirror the sample submission's *exact* schema (columns and order).
    In this environment, the sample has 4 columns (record_id, encoded_pixels, height, width).
    A schema mismatch can cause Kaggle ingestion/eval to fail or mis-parse, yielding 0.0.
    """
    if (
        "record_id" not in sample_df.columns
        or "encoded_pixels" not in sample_df.columns
    ):
        raise ValueError(
            f"sample_submission missing required columns. Columns: {list(sample_df.columns)}"
        )

    record_ids = [str(x).strip() for x in sample_df["record_id"].tolist()]
    bad = [rid for rid in record_ids if (rid == "") or (not rid.isdigit())]
    if bad:
        raise ValueError(
            f"Non-digit/empty record_id values found (showing up to 5): {bad[:5]}"
        )

    sub_df = sample_df.copy()

    sub_df["record_id"] = record_ids
    sub_df["encoded_pixels"] = ["-"] * len(record_ids)

    if len(sub_df) != len(sample_df):
        raise ValueError(
            f"Row count mismatch vs sample: {len(sub_df)} vs {len(sample_df)}"
        )
    if not sub_df["record_id"].is_unique:
        raise ValueError("record_id is not unique in submission.")
    if sub_df["record_id"].tolist() != record_ids:
        raise ValueError("record_id order mismatch vs sample (should be identical).")
    if not all(
        validate_rle_string(s)
        for s in pd.Series(sub_df["encoded_pixels"]).astype(str).unique().tolist()
    ):
        raise ValueError("Invalid RLE detected in submission.")

    sub_df = sub_df.loc[:, list(sample_df.columns)]
    return sub_df




## === cell 2
predicted_mask = np.zeros((256, 256), dtype=np.uint8)

submission_string = list_to_string(rle_encode(predicted_mask))
assert submission_string == "-", "Expected empty mask to encode as '-'"
assert validate_rle_string(
    submission_string
), f"Invalid RLE produced: {submission_string[:100]}"

submission_string



## === cell 3
sample_df, sample_path = read_sample_submission()

sub_df = build_submission_all_empty_match_sample_schema(sample_df)

sub_df["record_id"] = sub_df["record_id"].map(lambda x: str(x).strip())
sub_df["encoded_pixels"] = sub_df["encoded_pixels"].map(
    lambda x: "-" if str(x).strip() in ["", "-"] else str(x).strip()
)

sample_record_ids = [str(x).strip() for x in sample_df["record_id"].tolist()]
if sub_df["record_id"].tolist() != sample_record_ids:
    raise ValueError(
        "Submission record_id list does not exactly match sample_submission record_id list (order/content)."
    )

if list(sub_df.columns) != list(sample_df.columns):
    raise ValueError(
        f"Submission columns do not match sample columns.\n"
        f"Submission: {list(sub_df.columns)}\nSample: {list(sample_df.columns)}"
    )

assert sub_df["record_id"].is_unique
assert sub_df["record_id"].notna().all()
assert sub_df["encoded_pixels"].notna().all()
assert all(
    validate_rle_string(s)
    for s in pd.Series(sub_df["encoded_pixels"]).astype(str).unique().tolist()
)

sub_df.head()



## === cell 4
out_path = Path("submission.csv")

sub_df.to_csv(out_path, index=False)

reloaded = pd.read_csv(out_path, dtype=str, keep_default_na=False)

if list(reloaded.columns) != list(sample_df.columns):
    raise ValueError(
        f"Reloaded submission has wrong columns: {list(reloaded.columns)}; "
        f"expected {list(sample_df.columns)}"
    )
if len(reloaded) != len(sub_df):
    raise ValueError(
        f"Reloaded submission row count mismatch: {len(reloaded)} vs {len(sub_df)}"
    )

r_ids_reload = [str(x).strip() for x in reloaded["record_id"].tolist()]
r_ids_sample = [str(x).strip() for x in sample_df["record_id"].tolist()]
if r_ids_reload != r_ids_sample:
    raise ValueError(
        "Reloaded record_id does not exactly match sample record_id list (order/content)."
    )

enc_reload = [str(x).strip() for x in reloaded["encoded_pixels"].tolist()]
if any(x != "-" for x in enc_reload):
    raise ValueError(
        "Found non-'-' encoded_pixels after reload; expected all-empty baseline."
    )
if any(x == "" for x in enc_reload):
    raise ValueError("Found empty string encoded_pixels; should be '-'.")

print("Read sample from:", str(sample_path))
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print(sub_df.head())
print("Unique encoded strings:", pd.Series(sub_df["encoded_pixels"]).nunique())
