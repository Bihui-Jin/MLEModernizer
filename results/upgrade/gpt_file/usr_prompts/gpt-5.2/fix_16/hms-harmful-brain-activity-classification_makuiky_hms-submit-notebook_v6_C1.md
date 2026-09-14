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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.8836837324286071

# 6. Current score

1.39458

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.44263) has done: 'I fix the TensorFlow/protobuf crash by not forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides (those commonly trigger the `MessageFactory.GetPrototype` issue under newer protobuf). Then I fix the Keras loss string bug by switching to the built-in `tf.keras.losses.KLDivergence()` object, which preserves the same KL-divergence semantics but is compatible with TF/Keras 2.18. Finally, I fix the “Submission and answers must have the same length” error by writing predictions in the exact `sample_submission.csv` row order (9850 rows) rather than using a deduplicated test list, while still caching computed predictions per `eeg_id` for speed and ensuring each row sums to 1.'
- What this solution (achieved 1.45205) has done: 'I fix the protobuf/TensorFlow crash by ensuring we don’t force any protobuf implementation overrides and by importing TensorFlow after that environment cleanup (this is the direct cause of the `MessageFactory.GetPrototype` error). Then I keep your model/training/prediction logic the same but add a minimal, metric-aligned improvement: compute a class-prior from the training vote distributions and blend it slightly with model predictions (label-smoothing-style prior mix), which typically reduces KL-divergence for weak models and should move the score down toward your target. Finally, I keep the submission aligned to `sample_submission.csv` order and re-normalize rows to sum to 1 to avoid invalid submissions.'
- What this solution (achieved 1.40594) has done: 'I fix the protobuf/TensorFlow crash by forcing a compatible pure-Python protobuf runtime *before* importing TensorFlow (this directly addresses the `MessageFactory.GetPrototype` error in this Kaggle image). Then I keep your model, training loop, and prediction logic the same, but compute the class-prior from the full `train.csv` vote distribution (instead of only the first 512 rows) so the prior-mix calibration is less biased and should reduce KL toward your target with a minimal, metric-aligned change. Finally, I keep submission alignment identical to `sample_submission.csv`, enforce per-row normalization to sum to 1, and ensure `submission.csv` is written.'
- What this solution (achieved 1.41507) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` override that triggers `MessageFactory.GetPrototype` under this Kaggle image, and instead import TensorFlow in the default environment (this is the root runtime blocker). Then I keep your exact model/training/inference logic intact, only adding a safe fallback so that if TensorFlow still fails to import for any reason, the notebook still produces a valid, normalized submission using the same prior-mix calibration you already use. Finally, I ensure the submission is written as `submission.csv` in the exact `sample_submission.csv` order with probabilities clipped and renormalized to sum to 1.'
- What this solution (achieved 1.40902) has done: 'The runtime blocker is the TensorFlow/protobuf `MessageFactory.GetPrototype` crash during import, so I force a compatible pure-Python protobuf implementation *before* importing TensorFlow (this is the minimal change that reliably unblocks TF in this Kaggle image). Then I keep your exact model/training/inference logic the same, but ensure the TF-available/TF-unavailable branches both execute end-to-end and always write `submission.csv` with the correct row order and normalized probabilities. Finally, I keep your prior-mix calibration intact (already metric-aligned for KL) and only add small safety guards (clipping/renorm + deterministic seeds) that are score-neutral but prevent invalid submissions.'
- What this solution (achieved 1.42615) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides that trigger `MessageFactory.GetPrototype` in this Kaggle image, while keeping the same model/training/inference logic. To still always produce a valid submission even if TensorFlow fails to import for any other reason, I keep the prior-mix fallback branch and make sure it runs end-to-end and writes `submission.csv`. I also keep the existing prior-based calibration (already metric-aligned for KL) and only add small safety guards (stable normalization/clipping and deterministic seeds) that are score-neutral but prevent invalid rows. No architecture, loss, feature generation, or training-loop changes are introduced beyond the import/runtime fix.'
- What this solution (achieved 1.41481) has done: 'We fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python backend *before* importing TensorFlow, which is the most reliable minimal change in this Kaggle image. Then we keep your model/training/inference logic unchanged, but make sure the TF import failure is truly caught and the fallback branch still produces a valid submission. Finally, we ensure the submission is always written as `submission.csv` in exact `sample_submission.csv` order with clipped + renormalized probabilities so every row sums to 1.'
- What this solution (achieved 1.43861) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides that are triggering `MessageFactory.GetPrototype` in this Kaggle environment, while keeping your model/training/inference logic the same. I also keep a robust fallback path that always writes a valid `submission.csv` even if TensorFlow still can’t import for any reason. To nudge the score down (lower is better) toward your target with minimal risk, I keep your existing prior-mix calibration but make sure it consistently uses the full-train prior and is applied after proper normalization (no semantic change, just stability). Finally, I ensure the submission is written in the exact `sample_submission.csv` row order with per-row probability sums enforced to 1.'
- What this solution (achieved 1.42416) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *before* importing TensorFlow, which is the most reliable minimal change in this Kaggle image and unblocks end-to-end execution. Then I keep your exact model/training/inference logic and prior-mix calibration intact, only adding a safe fallback so that even if TF still fails for any reason, a valid `submission.csv` is always written. Finally, I ensure the submission stays aligned to `sample_submission.csv` row order and that probabilities are clipped and renormalized to sum to 1 (submission-validity, score-neutral).'
- What this solution (achieved 1.39458) has done: 'We fix the TensorFlow/protobuf import crash by not forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` (this is what triggers the `MessageFactory.GetPrototype` error in this environment) and instead simply avoid overriding protobuf behavior at all. To ensure the notebook always runs end-to-end, we keep your existing TF-optional fallback branch, but make the TF import failure truly non-fatal so a valid `submission.csv` is always written. The model, training loop, feature generation, and prior-mix calibration are kept identical; the only score-impacting behavior remains your existing prior blending. Finally, we keep submission row order exactly aligned to `sample_submission.csv` and enforce per-row normalization to guarantee a valid submission.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print("WARNING: TensorFlow import failed; will write prior-based submission only.")
    print("TF import error:", repr(e))

from PIL import Image



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EEG_TEST_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
SPEC_TEST_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)

if not os.path.exists("/kaggle/working/test_eegs_img/"):
    os.makedirs("/kaggle/working/test_eegs_img/")
EEG_IMG_TEST_PATH = "/kaggle/working/test_eegs_img/"

if not os.path.exists("/kaggle/working/test_spec_img/"):
    os.makedirs("/kaggle/working/test_spec_img/")
SPEC_IMG_TEST_PATH = "/kaggle/working/test_spec_img/"

META_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"



## === cell 2
if TF_AVAILABLE:
    tf.random.set_seed(42)
np.random.seed(42)
if TF_AVAILABLE:
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass



## === cell 3
eeg_zone = {
    "Cz-Pz": ["Cz", "Pz"],
    "Fz-Cz": ["Fz", "Cz"],
    "P4-O2": ["P4", "O2"],
    "C4-P4": ["C4", "P4"],
    "F4-C4": ["F4", "C4"],
    "Fp2-F4": ["Fp2", "F4"],
    "P3-O1": ["P3", "O1"],
    "C3-P3": ["C3", "P3"],
    "F3-C3": ["F3", "C3"],
    "Fp1-F3": ["Fp1", "F3"],
    "T6-O2": ["T6", "O2"],
    "T4-T6": ["T4", "T6"],
    "F8-T4": ["F8", "T4"],
    "Fp2-F8": ["Fp2", "F8"],
    "T5-O1": ["T5", "O1"],
    "T3-T5": ["T3", "T5"],
    "F7-T3": ["F7", "T3"],
    "Fp1-F7": ["Fp1", "F7"],
}



## === cell 4
plt.switch_backend("Agg")

_EEG_NLINES = len(eeg_zone)
_EEG_REL_POS = []
relpos = 0
for i in range(_EEG_NLINES):
    _EEG_REL_POS.append(relpos)
    if i in (1, 5, 9, 13):
        relpos += 200
    else:
        relpos += 40
_EEG_REL_POS = np.asarray(_EEG_REL_POS, dtype=np.float32)

_eeg_fig = None
_eeg_ax = None
_eeg_lines = None


def _ensure_eeg_figure(linewidth=0.2):
    global _eeg_fig, _eeg_ax, _eeg_lines
    if _eeg_fig is not None:
        return
    _eeg_fig, _eeg_ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    _eeg_ax.set_xticks([])
    _eeg_ax.set_yticks([])
    _eeg_ax.set_xlim(0, 50)
    _eeg_lines = []
    x = np.linspace(0, 50, 10000, dtype=np.float32)  # placeholder
    y0 = np.zeros_like(x)
    for _ in range(_EEG_NLINES):
        (ln,) = _eeg_ax.plot(x, y0, color="black", linewidth=linewidth)
        _eeg_lines.append(ln)


def _fig_to_rgb_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    buf.close()
    return arr


def generate_eeg(
    eegid,
    input_path=EEG_TEST_PATH,
    eeg_zone=eeg_zone,
    linewidth=0.2,
    eeg_out=EEG_IMG_TEST_PATH,  # kept for signature compatibility
):
    _ensure_eeg_figure(linewidth=linewidth)
    eeg = pd.read_parquet(f"{input_path}{eegid}.parquet")

    x = eeg.index.to_numpy(dtype=np.float32) / 200.0
    for i, (key, values) in enumerate(eeg_zone.items()):
        y = (
            eeg[values[0]].to_numpy(dtype=np.float32)
            - eeg[values[1]].to_numpy(dtype=np.float32)
        ) + _EEG_REL_POS[i]
        _eeg_lines[i].set_data(x, y)

    arr = _fig_to_rgb_array(_eeg_fig)
    return arr  # in-memory image




## === cell 5
spec_zones = ["LL", "RL", "LP", "RP"]

_spec_fig = None
_spec_axes = None
_spec_imgs = None


def _ensure_spec_figure():
    global _spec_fig, _spec_axes, _spec_imgs
    if _spec_fig is not None:
        return
    _spec_fig, _spec_axes = plt.subplots(
        nrows=len(spec_zones), figsize=(3, 3), sharex=True
    )
    _spec_imgs = []
    for row in range(len(spec_zones)):
        im = _spec_axes[row].imshow(
            np.zeros((10, 10), dtype=np.float32),
            cmap="turbo",
            aspect="auto",
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=0,
            vmax=1.0,
        )
        _spec_axes[row].set_xticks([])
        _spec_axes[row].set_yticks([])
        _spec_imgs.append(im)
    plt.subplots_adjust(hspace=0.01)


def generate_spectrogram(
    specid,
    input_path=SPEC_TEST_PATH,
    spec_out=SPEC_IMG_TEST_PATH,  # kept for signature compatibility
    spec_zones=spec_zones,
    output_filter=0.5,
):
    _ensure_spec_figure()

    spec = pd.read_parquet(f"{input_path}{specid}.parquet")
    spec = spec.fillna(0)

    if "time" in spec.columns:
        spec = spec.set_index("time").T
    else:
        spec = spec.T

    idx = pd.Index(spec.index.astype(str))
    parts = idx.to_series().str.split("_", n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = "0"

    brainreg = parts[0].to_numpy()
    freq = (
        pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    )

    spec = spec.reset_index(drop=True)
    spec["brainreg"] = brainreg
    spec["freq"] = freq
    spec = spec.set_index("freq")

    for row, zone in enumerate(spec_zones):
        data = spec[spec["brainreg"] == zone].drop(
            columns=["brainreg"], errors="ignore"
        )
        arr = data.to_numpy(dtype=np.float32)

        vmax = float(np.max(arr)) * float(output_filter) if arr.size else 0.0
        if vmax <= 0:
            vmax = 1.0

        _spec_imgs[row].set_data(
            arr if arr.size else np.zeros((10, 10), dtype=np.float32)
        )
        if data.shape[0] > 0 and data.shape[1] > 0:
            _spec_imgs[row].set_extent(
                [
                    float(pd.to_numeric(data.columns, errors="coerce").min()),
                    float(pd.to_numeric(data.columns, errors="coerce").max()),
                    float(data.index.min()),
                    float(data.index.max()),
                ]
            )
        _spec_imgs[row].set_clim(vmin=0.0, vmax=float(vmax))

    arr_rgb = _fig_to_rgb_array(_spec_fig)
    return arr_rgb  # in-memory image




## === cell 6
metadata = pd.read_csv(META_TEST)
train_metadata = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)




## === cell 7
def drop_images(paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass




## === cell 8
if TF_AVAILABLE:

    def _rgb_array_to_model_tensor(
        arr: np.ndarray, target_size=(224, 224)
    ) -> "tf.Tensor":
        img = Image.fromarray(arr).resize(target_size)
        out = np.asarray(img, dtype=np.float32) / 255.0
        return tf.convert_to_tensor(out)[None, ...]  # (1,H,W,3)

else:

    def _rgb_array_to_model_tensor(arr: np.ndarray, target_size=(224, 224)):
        raise RuntimeError("TensorFlow not available; cannot build model tensors.")


_eeg_cache = {}
_spec_cache = {}


def preprocess_data(metadata, row, train=False):
    if train:
        eegid = int(metadata.loc[row].eeg_id)
        specid = int(metadata.loc[row].spectrogram_id)
        eeg_arr = _eeg_cache.get(eegid)
        if eeg_arr is None:
            eeg_arr = generate_eeg(
                eegid=eegid,
                input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/",
            )
            _eeg_cache[eegid] = eeg_arr

        spec_arr = _spec_cache.get(specid)
        if spec_arr is None:
            spec_arr = generate_spectrogram(
                specid=specid,
                output_filter=0.7,
                input_path="/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/",
            )
            _spec_cache[specid] = spec_arr
    else:
        eegid = int(metadata.loc[row].eeg_id)
        specid = int(metadata.loc[row].spectrogram_id)

        eeg_arr = _eeg_cache.get(eegid)
        if eeg_arr is None:
            eeg_arr = generate_eeg(eegid=eegid)
            _eeg_cache[eegid] = eeg_arr

        spec_arr = _spec_cache.get(specid)
        if spec_arr is None:
            spec_arr = generate_spectrogram(specid=specid, output_filter=0.7)
            _spec_cache[specid] = spec_arr

    eeg_img_tr = _rgb_array_to_model_tensor(eeg_arr)
    spec_img_tr = _rgb_array_to_model_tensor(spec_arr)
    return eeg_img_tr, spec_img_tr, eegid




## === cell 9
iter_dict = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

all_votes = train_metadata[iter_dict].to_numpy(dtype=np.float64)
all_sum = all_votes.sum(axis=1, keepdims=True)
all_sum = np.where(all_sum <= 0, 1.0, all_sum)
all_probs = all_votes / all_sum
prior = all_probs.mean(axis=0).astype(np.float64)
prior = np.clip(prior, 1e-8, 1.0)
prior = prior / prior.sum()

print("Prior used (from FULL train.csv votes):", prior)



## === cell 10
if TF_AVAILABLE:

    def build_fallback_model(input_shape=(224, 224, 3), n_classes=6):
        eeg_in = keras.Input(shape=input_shape, name="eeg_img")
        spec_in = keras.Input(shape=input_shape, name="spec_img")

        def branch(x):
            x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
            x = keras.layers.MaxPool2D()(x)
            x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            x = keras.layers.MaxPool2D()(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            return x

        b1 = branch(eeg_in)
        b2 = branch(spec_in)
        x = keras.layers.Concatenate()([b1, b2])
        x = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)

        model = keras.Model(inputs=[eeg_in, spec_in], outputs=out)
        model.compile(optimizer="adam", loss=tf.keras.losses.KLDivergence())
        return model

    model = build_fallback_model()
else:
    model = None



## === cell 11
if TF_AVAILABLE:
    TRAIN_N = 512  # keep runtime reasonable (core training approach unchanged)
    train_subset = train_metadata.iloc[:TRAIN_N].copy()

    y_votes = train_subset[iter_dict].to_numpy(dtype=np.float32)
    y_sum = y_votes.sum(axis=1, keepdims=True)
    y_sum = np.where(y_sum <= 0, 1.0, y_sum).astype(np.float32)
    y = y_votes / y_sum

    eeg_tensors = []
    spec_tensors = []
    for i in range(len(train_subset)):
        eeg_img, spec_img, _ = preprocess_data(
            train_subset, train_subset.index[i], train=True
        )
        eeg_tensors.append(eeg_img)
        spec_tensors.append(spec_img)
        if (i + 1) % 64 == 0:
            gc.collect()

    X_eeg = tf.concat(eeg_tensors, axis=0)
    X_spec = tf.concat(spec_tensors, axis=0)

    model.fit([X_eeg, X_spec], y, epochs=1, batch_size=32, verbose=0)

    del X_eeg, X_spec, eeg_tensors, spec_tensors
    gc.collect()



## === cell 12
sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)

pdsubmit = sample_sub.copy()
pdsubmit["eeg_id"] = pdsubmit["eeg_id"].astype(np.int64)

PRIOR_ALPHA = 0.25
default = np.full((len(iter_dict),), 1.0 / len(iter_dict), dtype=np.float64)

if not TF_AVAILABLE:
    preds_aligned = np.tile(
        ((1.0 - PRIOR_ALPHA) * default + PRIOR_ALPHA * prior)[None, :],
        (len(pdsubmit), 1),
    )
    preds_aligned = np.clip(preds_aligned, 1e-8, 1.0)
    preds_aligned = preds_aligned / preds_aligned.sum(axis=1, keepdims=True)

    for k, col in enumerate(iter_dict):
        pdsubmit[col] = preds_aligned[:, k].astype(np.float32)

    pdsubmit = pdsubmit[["eeg_id"] + iter_dict]
    pdsubmit.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", pdsubmit.shape)
    print(pdsubmit.head())
    print(
        "Min/Max row-sum:",
        float(pdsubmit[iter_dict].sum(axis=1).min()),
        float(pdsubmit[iter_dict].sum(axis=1).max()),
    )
    print("Unique eeg_id in submission:", bool(pdsubmit["eeg_id"].is_unique))



## === cell 13
if TF_AVAILABLE:
    pred_map = {}

    BATCH = 64
    eeg_batch = []
    spec_batch = []
    id_batch = []

    def _flush_batch():
        if not id_batch:
            return
        eeg_tensor = tf.concat(eeg_batch, axis=0)
        spec_tensor = tf.concat(spec_batch, axis=0)
        preds = model.predict([eeg_tensor, spec_tensor], verbose=0).astype(np.float64)

        preds = np.clip(preds, 1e-8, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)

        preds = (1.0 - PRIOR_ALPHA) * preds + PRIOR_ALPHA * prior[None, :]
        preds = np.clip(preds, 1e-8, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)

        for j, eeg_id in enumerate(id_batch):
            pred_map[int(eeg_id)] = preds[j].tolist()

        eeg_batch.clear()
        spec_batch.clear()
        id_batch.clear()

    meta_map = metadata.copy()
    meta_map["eeg_id"] = meta_map["eeg_id"].astype(np.int64)
    meta_map["spectrogram_id"] = meta_map["spectrogram_id"].astype(np.int64)
    eeg_to_spec = dict(
        zip(meta_map["eeg_id"].to_numpy(), meta_map["spectrogram_id"].to_numpy())
    )

    for i, eeg_id in enumerate(pdsubmit["eeg_id"].to_numpy()):
        eeg_id = int(eeg_id)
        if eeg_id in pred_map:
            continue

        spec_id = eeg_to_spec.get(eeg_id, None)
        if spec_id is None:
            continue

        row_idx = meta_map.index[meta_map["eeg_id"] == eeg_id]
        if len(row_idx) == 0:
            continue
        row_idx = row_idx[0]

        eeg_img, spec_img, _ = preprocess_data(meta_map, row_idx, train=False)
        eeg_batch.append(eeg_img)
        spec_batch.append(spec_img)
        id_batch.append(eeg_id)

        if len(id_batch) >= BATCH:
            _flush_batch()
        if (i + 1) % 200 == 0:
            gc.collect()

    _flush_batch()

    preds_aligned = np.zeros((len(pdsubmit), len(iter_dict)), dtype=np.float64)
    for i, eeg_id in enumerate(pdsubmit["eeg_id"].to_numpy()):
        p = pred_map.get(int(eeg_id), None)
        if p is None:
            preds_aligned[i] = (1.0 - PRIOR_ALPHA) * default + PRIOR_ALPHA * prior
        else:
            preds_aligned[i] = np.asarray(p, dtype=np.float64)

    preds_aligned = np.clip(preds_aligned, 1e-8, 1.0)
    preds_aligned = preds_aligned / preds_aligned.sum(axis=1, keepdims=True)

    for k, col in enumerate(iter_dict):
        pdsubmit[col] = preds_aligned[:, k].astype(np.float32)

    row_sums = pdsubmit[iter_dict].sum(axis=1).to_numpy(dtype=np.float64)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        vals = pdsubmit[iter_dict].to_numpy(dtype=np.float64)
        vals = np.clip(vals, 1e-8, 1.0)
        vals = vals / vals.sum(axis=1, keepdims=True)
        pdsubmit.loc[:, iter_dict] = vals.astype(np.float32)

    pdsubmit = pdsubmit[["eeg_id"] + iter_dict]
    pdsubmit.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", pdsubmit.shape)
    print(pdsubmit.head())
    print(
        "Min/Max row-sum:",
        float(pdsubmit[iter_dict].sum(axis=1).min()),
        float(pdsubmit[iter_dict].sum(axis=1).max()),
    )
    print("Unique eeg_id in submission:", bool(pdsubmit["eeg_id"].is_unique))
    print("Prior used (from FULL train.csv votes):", prior)
