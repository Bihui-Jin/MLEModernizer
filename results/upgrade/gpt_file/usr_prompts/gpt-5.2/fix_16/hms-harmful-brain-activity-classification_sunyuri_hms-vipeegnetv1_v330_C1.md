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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.286529252669616

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The runtime error comes from a known incompatibility between TensorFlow and newer `protobuf` builds, which triggers `MessageFactory.GetPrototype` failures before your code can execute end-to-end. I add a minimal, Kaggle-safe environment workaround at the very top (before importing TensorFlow) to force the pure-Python protobuf implementation, which avoids that crash. I also make the inference path robust when no model weights are found (fallback to uniform probabilities) so you always get a valid `submission.csv` with rows summing to 1, even if the “models*” dataset isn’t attached. Finally, I fix a small indexing bug in the test batching slice that can lead to wrong/empty batches.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import* and by explicitly importing `google.protobuf` early so the env var takes effect. I also fix training-mode data loading so the notebook doesn’t fail when `READ_EEG_FILES=False` but `train.csv` hasn’t been generated yet (it fall back to creating the needed consolidated `train` dataframe in-memory). Finally, to move the score down toward the target (lower is better) without changing the model architecture/training loop, I remove the “uniform fallback submission” when weights are missing and instead run inference with the provided training logic when possible; if weights truly are unavailable, it still produce a valid submission but warn that score be poor.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by strengthening the early-environment workaround so it applies before TensorFlow (and its deps) initialize, which is the root cause of the `MessageFactory.GetPrototype` error. I also make the Kaggle path resolution deterministic (prefer the official competition dataset folder directly) to avoid accidentally pointing at a non-existent `modelsxxxxxxx` directory. To move the score down toward the target (lower is better) without changing the model/training loop, I prevent the “uniform fallback” from triggering just because fold-weights are missing, and instead gracefully fall back to using stage1 weights when stage2 are absent (same architecture/semantics; just more realistic predictions). Finally, I ensure the inference batching concatenation stays aligned to `test.eeg_id` order and always writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing a compatible protobuf implementation and (if needed) monkey-patching the missing `MessageFactory.GetPrototype` method before importing TensorFlow. I also make model weight discovery deterministic (prefer the official competition dataset path and only use `models*` folders that actually contain fold weights) so inference uses real weights instead of falling back to uniform predictions (which is why your score is far from the target). Finally, I fix the test batching accumulation so predictions always align 1:1 with `test.csv` order, and I keep the submission probabilities properly normalized and clipped for KL-divergence.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by ensuring the protobuf monkey-patch is applied correctly (both at the module level and on the MessageFactory instance) before importing TensorFlow, so execution can proceed end-to-end on Kaggle’s Python 3.13 environment. I also make model-weight discovery more robust by preferring an attached `/kaggle/input/models*` directory that actually contains fold weight files, which should prevent unintended uniform fallback and move the score down toward the target (lower is better). Finally, I keep the existing architecture/training/inference logic intact, but add a small safety normalization at submission time to guarantee probabilities sum to 1 and avoid invalid submissions.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.2865), and the biggest likely cause is that inference is running with randomly-initialized models (because `NEEDTRAIN=False` prevents loading the EfficientNet pre-trained backbone weights). I keep your architecture and inference loop intact, but allow backbone weights to load during inference when the pre-trained weights dataset is present; this should materially reduce KL without changing evaluation semantics. I also ensure models are created under the same distribution strategy scope used elsewhere, and I harden the test-time generator by defining `TARGETS_RAW` even in inference mode (to avoid accidental NameErrors if modes change) without affecting predictions. Finally, I keep the existing probability clipping/renormalization so the submission remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.2865), and the highest-impact minimal fix is to ensure inference uses real pretrained EfficientNet weights (instead of `weights=None`) by default; otherwise the backbone is effectively random and predictions are near-uniform/poor. I keep your architecture and training/inference flow intact, but switch to `weights="imagenet"` when the local `.h5` pretrained files are not present, which is the smallest change that should materially reduce KL. I also make test-time DataGenerator robust by ensuring `sign_id` exists even in `mode="test"` (so it won’t crash if you later include `img`), and I keep your submission clipping+renormalization to guarantee valid probabilities. No early stopping, sampling, or training loop changes are introduced.'
- What this solution (achieved 1.40995) has done: 'The current score is far worse than the target (lower-is-better), and the most likely cause is that your inference is unintentionally using randomly-initialized fold models because `clone_model()` does not reliably preserve the built state and any loaded backbone weights in a way that matches the template (especially under mixed precision + strategy). I keep your exact model architecture and inference semantics, but change inference to build a fresh model per fold under the same `strategy.scope()` and then load fold weights, avoiding `clone_model()` entirely. I also make the weight-set detection more permissive (use all folds that exist instead of requiring a complete set) so you don’t fall back to uniform predictions when some folds are missing, which should reduce KL toward the target. Finally, I keep your existing clipping+renormalization to ensure a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower is better), and the most likely reason is that test-time predictions are being generated with incomplete/incorrect test features: in `mode="test"` your generator hard-codes `r_eeg=0`, but the inference loop fills `eegs_test` with the *full* 50s array (so slicing `0:50s` yields empty/short arrays depending on how the parquet is stored), causing unstable/near-garbage inputs. I make a minimal, score-relevant fix by ensuring the inference cache stores exactly the same 50s window shape the generator expects (always 50*RSFREQ samples) and by clipping/padding if needed, without changing the model/training logic. I also stop deleting only `eegs_test` while leaving `stfts_test/imgs_test` to grow across batches (memory pressure can silently degrade runtime/inference), and instead clear all test caches consistently each batch. Finally, I keep the existing probability clipping+renormalization so the submission stays valid for KL divergence.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), and the most likely reason is that test-time EEG preprocessing does not match what the model saw during training: in training you compute *bipolar montage* from 19 raw channels, but in inference you subtract already-bipolar columns like `Fp1-F7`, which don’t exist in the parquet (so you end up with wrong/near-zero signals and poor predictions). I keep your model/training logic intact and make the smallest fix: build the bipolar montage at inference using the same `BRAIN` pairs computed from raw 10-20 channels, matching the training preprocessing exactly. I also clear `spectrograms_test` alongside the other caches per batch to prevent memory growth that can destabilize inference, without changing semantics. The submission writing/clipping/renormalization stays the same to remain valid for KL-divergence.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), so we should improve predictions rather than tweak calibration. The biggest score-relevant issue in your inference path is that `DataGenerator(mode="test")` always uses `r_eeg=0`, but the test EEGs are already exactly 50s long; your generator then slices `0:10000` and *later center-crops again*, which can accidentally misalign with how training windows are formed and can also break if any loaded array length isn’t exactly as expected. I make a minimal, semantics-preserving fix: in test mode, explicitly set `r_eeg` so that the subsequent slice produces the same centered 50s window the rest of the pipeline expects (robust even if a parquet is slightly longer/shorter). I also ensure the test generator’s `sign_id` matches the global `test.sign_id` (rather than being re-numbered from 0 each batch), which prevents any possible mismatch for `img` mode and keeps sample identity consistent without changing model logic.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), so we should improve prediction correctness with minimal, metric-aligned fixes rather than tune calibration. The highest-impact issue is that the inference EEG preprocessing does **not** match training: training loads already-bipolar parquet columns like `Fp1-F7`, but your `BRAIN_PAIRS` inference path subtracts `Fp1 - F7` (raw channels), which typically don’t exist, yielding wrong/near-zero signals and poor KL. I minimally change both train/test EEG loading to use the exact same montage logic: if bipolar columns exist, use them directly; otherwise compute from raw pairs—this preserves core model/training/inference semantics while fixing a major data mismatch. I also keep your submission clipping+renormalization and ensure test batching stays aligned, without changing the model architecture, loss, or training loop.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), so we should improve correctness rather than calibration. The most likely score-killer here is that test-time EEGs are being force-truncated to the first 50s, while your generator is designed to center-crop when longer/shorter; keeping only the first 50s removes the ability to center and can mismatch what training/validation effectively uses. I make a minimal inference-only change to keep the full resampled EEG length in the cache and let `DataGenerator(mode="test")` compute a centered `r_eeg`, while still padding if shorter than 50s. I also ensure the final submission uses the exact `sample_submission.csv` column order to avoid any subtle column/order issues, without changing the model, loss, or training loop.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), so we should improve prediction correctness with minimal, semantics-preserving fixes. The biggest likely score-killer in this script is that `DATATYPE=["eeg"]` but inference never applies the same “train-style” reordering/swap logic used in `mode!="train"` when the EEG montage has 18 channels: training/valid effectively assumes the 16 kept channels correspond to the first 8 and last 8 after swapping, while test currently just takes the first 8 and last 8 without first applying the 4:8/12:16/8:12 swap on the full montage (this mismatch can badly hurt generalization). I add a tiny inference-only preprocessing step to apply the exact same swap on the full montage (18ch) before the generator selects 16 channels, matching training/valid semantics without changing the model or training loop. I also fix a subtle but important batching alignment issue by using the original `sign_id` indices for `preds_all` assignment (instead of relying on `start:end` slices), ensuring predictions always land on the correct rows even if batching changes.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.2865), so we should improve prediction correctness with minimal changes. The biggest score-relevant issue is that your test-time EEG montage “swap” is applied to the full 18-channel montage, but your generator later selects only the first 8 and last 8 channels; this means the swap has no effect in inference (unlike train/valid where the swap is applied after selecting 16 channels), creating a train/test preprocessing mismatch. I remove the ineffective pre-swap in the test cache and instead apply the exact same 16-channel swap inside `DataGenerator` for `mode="test"` (matching valid semantics, not changing architecture/loss/training loops). I also make the test EEG slice robust to any slight length mismatch by padding when a centered crop would run short (keeping semantics but avoiding accidental shorter inputs).'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import google.protobuf  # noqa: F401

try:
    import google.protobuf.message_factory as _mf_mod
    from google.protobuf.message_factory import MessageFactory as _MessageFactory

    def _compat_get_prototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        raise AttributeError(
            "MessageFactory has neither GetPrototype nor GetMessageClass"
        )

    if not hasattr(_MessageFactory, "GetPrototype"):
        _MessageFactory.GetPrototype = _compat_get_prototype  # type: ignore[attr-defined]

    if hasattr(_mf_mod, "message_factory") and not hasattr(
        _mf_mod.message_factory, "GetPrototype"
    ):
        try:
            _mf_mod.message_factory.GetPrototype = _compat_get_prototype.__get__(
                _mf_mod.message_factory, _mf_mod.message_factory.__class__
            )
        except Exception:
            pass
except Exception:
    pass

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"
elif len(cwd_parts) > 1 and cwd_parts[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

if PLATFORM == "kaggle" and not NEEDTRAIN:
    models_root = "/kaggle/input"
    candidate_dirs = []
    try:
        for dir_name in os.listdir(models_root):
            if dir_name.startswith("models"):
                candidate_dirs.append(dir_name)
    except FileNotFoundError:
        candidate_dirs = []

    def _count_weight_files(d):
        p = os.path.join(models_root, d)
        try:
            files = os.listdir(p)
        except Exception:
            return 0
        return sum(
            (
                f.startswith("fold")
                and (f.endswith("_stage1.h5") or f.endswith("_stage2.h5"))
            )
            for f in files
        )

    if candidate_dirs:
        candidate_dirs = sorted(candidate_dirs, key=_count_weight_files, reverse=True)
        if _count_weight_files(candidate_dirs[0]) > 0:
            LOAD_MODELS_FROM = candidate_dirs[0]

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"/input/{LOAD_MODELS_FROM}"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100
SPE_WIDE = 256

STFT_LENGTH = 50
STFT_HIGH = 64
STFT_WIDE = round(STFT_LENGTH / 0.5)

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024

BATCHSIZE = 16

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]

BRAIN_PAIRS = [(p.split("-")[0], p.split("-")[1]) for p in BRAIN]

TEST_BATCHSIZE = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
from tensorflow.keras import optimizers
from tensorflow.python.framework.ops import reset_default_graph

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enabling skipped:", repr(e))

MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
TARGETS_RAW = [i + "_raw" for i in TARGETS]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

PRETRAIN_ROOT = (
    "./input/pre-trained-weights"
    if PLATFORM == "local"
    else "/kaggle/input/pre-trained-weights"
)


def _extract_montage(eeg_df: pd.DataFrame) -> np.ndarray:
    out = []
    cols = eeg_df.columns
    if all(ch in cols for ch in BRAIN):
        for ch in BRAIN:
            v = eeg_df.loc[:, ch].values
            v[np.isnan(v)] = 0
            out.append(v[None, :])
    else:
        for a_ch, b_ch in BRAIN_PAIRS:
            if (a_ch in cols) and (b_ch in cols):
                v = (eeg_df.loc[:, a_ch] - eeg_df.loc[:, b_ch]).values
            else:
                v = np.zeros((len(eeg_df),), dtype=np.float32)
            v[np.isnan(v)] = 0
            out.append(v[None, :])
    return np.concatenate(out, axis=0).astype(np.float32)


if NEEDTRAIN:
    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)
        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
            train = df.drop_duplicates(
                [
                    "eeg_id",
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ).reset_index(drop=True)
            train["sign_id"] = train.index.values
            df["sign_id"] = df.index.values

            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data



## === cell 1
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        if ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):

            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = _extract_montage(eeg_default)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:

                eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=100, nfft=640
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)

                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
                    eeg_plot = eeg_plot[
                        :,
                        round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                            (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                        ),
                    ]

                    img_save = np.zeros(
                        (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                    )
                    for ii in range(eeg_plot.shape[0]):
                        fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                        fig.patch.set_facecolor("black")

                        plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)

                        plt.xlim(-5, eeg_plot.shape[1] + 5)
                        plt.ylim(0, 200)
                        plt.axis("off")

                        byte_stream = io.BytesIO()
                        plt.savefig(
                            byte_stream, format="png", bbox_inches="tight", dpi=100
                        )
                        byte_stream.seek(0)
                        img = Image.open(byte_stream)
                        img = np.array(img)[:, :, :1]
                        img = img / 255
                        img = np.array(img, dtype=np.float32)
                        byte_stream.truncate()
                        plt.close("all")

                        if img.shape != (36, IMG_WIDE, 1):
                            img = np.concatenate((img, img, img), 2)
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 2
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()




## === cell 3
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stfts=None,
        specs=None,
        imgs=None,
    ):

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs

        if "sign_id" not in self.dataframe.columns:
            self.dataframe = self.dataframe.copy()
            self.dataframe["sign_id"] = np.arange(len(self.dataframe), dtype=np.int64)

        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.dataframe) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        targets_batch = list()

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = row.sign_id
            if self.mode != "test":
                sample_weight = float(np.sum(row[TARGETS_RAW].values) / 20.0)
                targets_batch.append(row.expert_consensus)

            if self.mode == "test":
                r_spe = 0
                r_stft = 0
                if self.eegs is not None and row.eeg_id in self.eegs:
                    eeg_len = int(self.eegs[row.eeg_id].shape[1])
                    expected_len = int(EEG_LENGTH * RSFREQ)
                    r_eeg = max(0.0, (eeg_len - expected_len) / (2.0 * RSFREQ))
                else:
                    r_eeg = 0.0
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw),
                    :,
                ].reset_index(drop=True)
                if self.mode == "train":
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )
                    row = rows.loc[0, :]
                elif self.mode == "valid":
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = row.eeg_label_offset_seconds

            if "spe" in DATATYPE:
                spe = list()
                for k in range(4):
                    spe.append(
                        np.reshape(
                            self.specs[row.spectrogram_id][
                                r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                            ].T,
                            (1, 100, 300),
                        )
                    )
                spe = np.concatenate(spe, axis=0)

            if "eeg" in DATATYPE:
                start = int(round(r_eeg * RSFREQ))
                end = int(round((r_eeg + 50) * RSFREQ))
                eeg_full = self.eegs[row.eeg_id]
                if end > eeg_full.shape[1]:
                    pad = end - eeg_full.shape[1]
                    eeg_full = np.pad(
                        eeg_full, ((0, 0), (0, pad)), mode="constant", constant_values=0
                    )
                eeg = eeg_full[:, start:end]

            if "stft" in DATATYPE:
                stft_t = self.stfts[-row.eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

            if "spe" in DATATYPE:
                spe[np.isnan(spe)] = 0
                exp_min, exp_max = -4, 6
                spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                spe = np.log(spe)

                spe = spe[
                    :,
                    :,
                    round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                        (spe.shape[2] - SPE_WIDE) / 2
                    ),
                ]

                if self.mode == "train":
                    spe2 = spe.copy()
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[2]
                        spe[2] = spe2[0]
                    if np.random.rand() > 0.5:
                        spe[1] = spe2[3]
                        spe[3] = spe2[1]
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[1]
                        spe[2] = spe2[3]
                        spe[1] = spe2[0]
                        spe[3] = spe[2]

                spe = (spe - exp_min) / (exp_max - exp_min) * 255
                spe = np.clip(spe, a_min=0, a_max=255)

                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)

                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                eeg = np.clip(eeg_save, a_min=-255, a_max=255)
                eeg = eeg + 255
                eeg = eeg / 2

                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                exp_min, exp_max = -6, 6
                stft = np.clip(stft, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                stft = np.log(stft)

                stft = np.concatenate(
                    (
                        stft[0 : round(EEG_CHANNEL_USED / 2), :, :],
                        stft[-round(EEG_CHANNEL_USED / 2) :, :, :],
                    ),
                    axis=0,
                )

                if self.mode == "train":
                    stft[0 : round(EEG_CHANNEL_USED / 2), :] = stft[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    stft[-round(EEG_CHANNEL_USED / 2) :, :] = stft[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]
                    stft2 = stft.copy()
                    stft[4:8, :] = stft2[12:16, :]
                    stft[8:12, :] = stft2[4:8, :]
                    stft[12:16, :] = stft2[8:12, :]

                    if np.random.rand() > 0.5:
                        stft = stft[::-1, :]
                else:
                    stft2 = stft.copy()
                    stft[4:8, :] = stft2[12:16, :]
                    stft[8:12, :] = stft2[4:8, :]
                    stft[12:16, :] = stft2[8:12, :]

                stft_save = np.zeros(
                    (stft.shape[0] // 4, stft.shape[1], stft.shape[2] * 4),
                    dtype=np.float32,
                )

                for ii in range(stft.shape[0]):
                    stft_save[ii // 4, :, (ii % 4) :: 4] = stft[ii, :, :]

                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * stft_save.shape[2])
                        stft_save[
                            :,
                            mask : round(
                                mask + np.random.rand() * stft_save.shape[2] * 0.02
                            ),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * stft_save.shape[2])
                        stft_save[
                            :,
                            mask : round(
                                mask + np.random.rand() * stft_save.shape[2] * 0.02
                            ),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * stft_save.shape[2])
                        stft_save[
                            :,
                            mask : round(
                                mask + np.random.rand() * stft_save.shape[2] * 0.02
                            ),
                        ] = 0

                stft = (stft_save - exp_min) / (exp_max - exp_min) * 255
                stft = np.clip(stft, a_min=0, a_max=255)

                if "stft" in DATATYPE:
                    if j == 0:
                        x_stft = np.zeros(
                            (len(indexes), stft.shape[0], stft.shape[1], stft.shape[2]),
                            dtype="float32",
                        )
                        x_stft[j] = stft
                    else:
                        x_stft[j] = stft

            if "img" in DATATYPE:
                img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                if self.mode == "train":
                    img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                    img[10:18, :, :] = img[10:18, :, :][np.random.permutation(8), :, :]
                    if np.random.rand() > 0.5:
                        img = img[::-1, :, :]

                for ii in range(img.shape[0]):
                    axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                    start_temp = round(
                        max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                    )
                    end_temp = round(
                        min(
                            img_save.shape[0],
                            axis_temp + img_save.shape[1] / img.shape[0],
                        )
                    )
                    temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                    img_save[start_temp:end_temp, :] = (
                        img_save[start_temp:end_temp, :]
                        + img[
                            ii, temp_temp : round(temp_temp + end_temp - start_temp), :
                        ]
                    )
                img_save = np.clip(img_save, a_min=0, a_max=1)

                img = np.reshape(img_save, (img_save.shape[0], img_save.shape[1], 1))
                img = np.concatenate((img, img, img), -1)

                img = (img - np.mean(img)) / (np.std(img) + 1e-6)

                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / float(np.sum(row[TARGETS].values))

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1.0

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "stft" in DATATYPE:
            x["stft"] = x_stft
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 4
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step

        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)

        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self, step):
        step = step + 1
        if step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0
                + tf.cos(
                    (step - self.warm_step) / (self.total_step - self.warm_step) * np.pi
                )
            )

        return np.float32(lr)


class IniToOne(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOne, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOne(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOne, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class IniToOneAtten(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOneAtten, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOneAtten(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOneAtten, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


def _resolve_effnet_weights(model_name: str):
    """
    Change (score-relevant): if the custom pre-trained .h5 exists, load it (original behavior).
    Otherwise, fall back to ImageNet weights to avoid random-initialized backbones, which should
    improve predictions and reduce KL substantially.
    """
    pretrain_path = os.path.join(PRETRAIN_ROOT, f"{model_name}_notop.h5")
    if os.path.exists(pretrain_path):
        return None, pretrain_path  # build with weights=None then load_weights(path)
    return "imagenet", None  # build with imagenet weights


def build_model():
    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
        x_spe = tf.keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        w, wpath = _resolve_effnet_weights("efficientnetv2-b0")
        base_model_spe = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=w, include_preprocessing=True
        )
        if wpath is not None:
            base_model_spe.load_weights(wpath)

        base_model_spe._name = "spe_extractor"

        base_model_spe_pre = tf.keras.Model(
            base_model_spe.input, base_model_spe.get_layer("block3b_add").output
        )
        base_model_spe_pre._name = "spe_extractor_pre"
        x_spe1 = base_model_spe_pre(x_spe[:, 0, :, :, :])
        x_spe2 = base_model_spe_pre(x_spe[:, 1, :, :, :])
        x_spe3 = base_model_spe_pre(x_spe[:, 2, :, :, :])
        x_spe4 = base_model_spe_pre(x_spe[:, 3, :, :, :])

        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])
        base_model_spe_after = tf.keras.Model(
            base_model_spe_pre.output, base_model_spe.output
        )
        base_model_spe_after._name = "spe_extractor_after"
        x_spe = base_model_spe_after(x_spe)

        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

        inp.append(inp_spe)
        y_spe = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_spe)

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
                filters=30,
                kernel_size=10,
                strides=10,
                padding="same",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOne(),
                kernel_constraint=SumToOne(),
                input_shape=(None, 1),
            )
        else:
            eeg_embed = tf.keras.layers.Conv1D(
                filters=30,
                kernel_size=10,
                strides=10,
                padding="same",
                use_bias=False,
                activation=None,
            )
        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg)

        x_eeg = tf.keras.layers.Concatenate(axis=-1)(
            [
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 0:10]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 10:20]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 20:30]
                ),
            ]
        )

        x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        w, wpath = _resolve_effnet_weights("efficientnetv2-b0")
        base_model_eeg = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=w, include_preprocessing=True
        )
        if wpath is not None:
            base_model_eeg.load_weights(wpath)

        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)

        x_eeg = tf.reduce_mean(x_eeg, axis=1, keepdims=True)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        if PLATFORM == "local":
            eeg_atten = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=x_eeg.shape[2],
                strides=x_eeg.shape[2],
                padding="valid",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOneAtten(),
                kernel_constraint=SumToOneAtten(),
                input_shape=(None, 1),
            )
        else:
            eeg_atten = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=x_eeg.shape[2],
                strides=x_eeg.shape[2],
                padding="valid",
                use_bias=False,
                activation=None,
            )
        x_eeg = tf.keras.layers.TimeDistributed(eeg_atten)(x_eeg)
        x_eeg = tf.keras.layers.Flatten()(x_eeg)

        x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

        inp.append(inp_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(4, STFT_HIGH, STFT_WIDE * 4), name="stft")
        x_stft = tf.keras.layers.Reshape(
            (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
        )(inp_stft)
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        w, wpath = _resolve_effnet_weights("efficientnetv2-b0")
        base_model_stft = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=w, include_preprocessing=True
        )
        if wpath is not None:
            base_model_stft.load_weights(wpath)

        base_model_stft._name = "stft_extractor"

        base_model_stft_pre = tf.keras.Model(
            base_model_stft.input, base_model_stft.get_layer("block3b_add").output
        )
        base_model_stft_pre._name = "stft_extractor_pre"
        x_stft1 = base_model_stft_pre(x_stft[:, 0, :, :, :])
        x_stft2 = base_model_stft_pre(x_stft[:, 1, :, :, :])
        x_stft3 = base_model_stft_pre(x_stft[:, 2, :, :, :])
        x_stft4 = base_model_stft_pre(x_stft[:, 3, :, :, :])

        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )
        base_model_stft_after = tf.keras.Model(
            base_model_stft_pre.output, base_model_stft.output
        )
        base_model_stft_after._name = "stft_extractor_after"
        x_stft = base_model_stft_after(x_stft)

        x_stft = x_stft[
            :, :, (x_stft.shape[2] - 1) // 2 : (x_stft.shape[2]) // 2 + 1, :
        ]
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = tf.keras.layers.Dropout(0.8)(x_stft)

        inp.append(inp_stft)
        y_stft = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_stft)

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")

        pretrain_path = os.path.join(PRETRAIN_ROOT, "efficientnetb0_notop.h5")
        img_weights = None if os.path.exists(pretrain_path) else "imagenet"
        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=img_weights, input_tensor=inp_img
        )
        if os.path.exists(pretrain_path):
            base_model_img.load_weights(pretrain_path)

        base_model_img._name = "img_extractor"
        x_img = base_model_img.output

        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        inp.append(inp_img)
        y_img = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_img)

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 5
if NEEDTRAIN:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K
    import itertools

    gkf = GroupKFold(n_splits=SPLITS)

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.expert_consensus, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
        df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

        df_train_stage2 = df_train_stage1[
            np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)

        train_gen = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="loss")
        plt.plot(epochs, val_loss, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage1.svg"))
        plt.close()

        valid_stage1 = df_valid_stage1[TARGETS].values
        predict_stage1 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        plt.yticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    round(EPOCHS / 3), LEARN_RATE * 0.1, LEARN_RATE * 0.1 * 0.1, 0
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage2.h5"))

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="loss")
        plt.plot(epochs, val_loss, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage2.svg"))
        plt.close()

        valid_stage2 = df_valid_stage2[TARGETS].values
        predict_stage2 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        plt.yticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()



## === cell 6
if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    stage2_paths_all = [
        os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        for model_i in range(SPLITS)
    ]
    stage1_paths_all = [
        os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage1.h5")
        for model_i in range(SPLITS)
    ]

    stage2_paths = [p for p in stage2_paths_all if os.path.exists(p)]
    stage1_paths = [p for p in stage1_paths_all if os.path.exists(p)]

    if len(stage2_paths) > 0:
        weight_paths = stage2_paths
        print(f"Using stage2 weights for {len(weight_paths)} folds")
    elif len(stage1_paths) > 0:
        weight_paths = stage1_paths
        print(f"Using stage1 weights for {len(weight_paths)} folds")
    else:
        weight_paths = []

    if len(weight_paths) == 0:
        print("WARNING: No model weights found in:", LOAD_MODELS_FROM)
        print("Falling back to uniform probabilities (valid submission, poor score).")
        preds_all = np.full(
            (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub[TARGETS] = sub[TARGETS].clip(1e-7, 1.0)
        sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(
            axis=1, keepdims=True
        )

        sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        sub = sub[["eeg_id"] + [c for c in sample_sub.columns if c != "eeg_id"]]

        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
    else:
        print("Weights dir:", LOAD_MODELS_FROM)

        models = []
        with strategy.scope():
            for wp in weight_paths:
                print("Loading:", os.path.basename(wp))
                m = build_model()
                m.load_weights(wp)
                models.append(m)

        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")

            for i, f in enumerate(files_test):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
        if (
            ("spe" in DATATYPE)
            or ("eeg" in DATATYPE)
            or ("stft" in DATATYPE)
            or ("img" in DATATYPE)
        ):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

            expected_len = int(EEG_LENGTH * RSFREQ)

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = _extract_montage(eeg_default)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if eeg.shape[1] < expected_len:
                    pad = expected_len - eeg.shape[1]
                    eeg = np.pad(
                        eeg, ((0, 0), (0, pad)), mode="constant", constant_values=0
                    )

                if "stft" in DATATYPE:
                    eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                    ff, tt, ss = signal.spectrogram(
                        eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=100, nfft=640
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]

                if "img" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                    train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]

                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")

                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)

                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")

                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")

                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]

                            img_save[ii, :, :] = img

                        imgs_test[train_plot.sign_id[j]] = img_save

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = ss
                    stfts_test[-eeg_id] = tt

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    end = i + 1
                    start = end - TEST_BATCHSIZE
                    if start < 0:
                        start = 0

                    test_batch = test.iloc[start:end, :].copy().reset_index(drop=True)

                    test_gen = DataGenerator(
                        test_batch,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )
                    preds = []
                    for model_i in range(len(models)):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0).astype(np.float32)

                    batch_idx = test.iloc[start:end]["sign_id"].values
                    preds_all[batch_idx, :] = pred[: len(batch_idx), :]

                    spectrograms_test = {}
                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

        preds_all = np.clip(preds_all, 1e-7, 1.0).astype(np.float32)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all

        sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        sub = sub[["eeg_id"] + [c for c in sample_sub.columns if c != "eeg_id"]]

        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
