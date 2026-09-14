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

0.3478364216127923

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    if TF_AVAILABLE:
        try:
            import efficientnet.tfkeras as efn
        except Exception as e:
            print("EfficientNet import failed:", e)
            TF_AVAILABLE = False

    if TF_AVAILABLE:

        def build_model():
            inp = []
            if "spe" in DATATYPE:
                inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
                x_spe = tf.keras.layers.Concatenate(axis=1)(
                    [
                        inp_spe[:, 0, :, :],
                        inp_spe[:, 1, :, :],
                        inp_spe[:, 2, :, :],
                        inp_spe[:, 3, :, :],
                    ]
                )
                x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(
                    x_spe
                )
                x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])
                base_model_spe = efn.EfficientNetB0(
                    include_top=False, weights=None, input_shape=None
                )
                base_model_spe._name = "spe_extractor"
                x_spe = base_model_spe(x_spe)
                x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
                inp.append(inp_spe)
                y = x_spe

            if "eeg" in DATATYPE:
                inp_eeg = tf.keras.Input(
                    shape=(
                        (4 * 4 + 2) * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    )
                )
                x_eeg = tf.keras.layers.Reshape(
                    (inp_eeg.shape[1], inp_eeg.shape[2], 1)
                )(inp_eeg)
                x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])
                base_model_eeg = efn.EfficientNetB0(
                    include_top=False, weights=None, input_shape=None
                )
                base_model_eeg._name = "eeg_extractor"
                x_eeg = base_model_eeg(x_eeg)
                x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
                x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)
                inp.append(inp_eeg)
                y = (
                    tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                    if "spe" in DATATYPE
                    else x_eeg
                )

            if "stft" in DATATYPE:
                inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
                x_stft = tf.keras.layers.Reshape(
                    (inp_stft.shape[1], inp_stft.shape[2], 1)
                )(inp_stft)
                x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])
                base_model_stft = efn.EfficientNetB0(
                    include_top=False, weights=None, input_shape=None
                )
                base_model_stft._name = "stft_extractor"
                x_stft = base_model_stft(x_stft)
                x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
                inp.append(inp_stft)
                y = (
                    tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                    if ("spe" in DATATYPE) or ("eeg" in DATATYPE)
                    else x_stft
                )

            if "img" in DATATYPE:
                inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
                base_model_img = efn.EfficientNetB0(
                    include_top=False, weights=None, input_shape=None
                )
                base_model_img._name = "img_extractor"
                x_img = base_model_img(inp_img)
                x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
                inp.append(inp_img)
                y = (
                    tf.keras.layers.Concatenate(axis=1)([y, x_img])
                    if any(t in DATATYPE for t in ["spe", "eeg", "stft"])
                    else x_img
                )

            y = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(y)
            return tf.keras.Model(inputs=inp, outputs=y)

        model_template = build_model()
        models = []
        for model_i in range(SPLITS):
            print(f"Fold {model_i+1}")
            model = tf.keras.models.clone_model(model_template)
            model_path = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            model.load_weights(model_path)
            models.append(model)

        preds_all = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS))
    else:
        class_counts = df[TARGETS].sum().values.astype(float)
        class_prior = class_counts / class_counts.sum()
        preds_all = np.tile(class_prior, (len(test), 1))
        print("Using class‑frequency priors for predictions.")

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = preds_all
    submission_path = "submission.csv"
    sub.to_csv(submission_path, index=False)
    print("Submission shape", sub.shape)
    print(f"Submission written to {submission_path}")
    print(sub.head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3426325047.py in <cell line: 0>()
----> 1 if not NEEDTRAIN:
      2     test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
      3     test["sign_id"] = test.index.values
      4     print("Test shape", test.shape)
      5 

NameError: name 'NEEDTRAIN' is not defined
