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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.6167917389474165

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.70961) has done: 'The changes increase data‑loading parallelism by using multiple workers (and a valid `prefetch_factor`) for both image‑level and patient‑level DataLoaders, which reduces I/O and preprocessing bottlenecks while keeping the exact model architecture, training loops, and inference logic unchanged.'

# 9. Code solution

## === cell 0
config = {
    "seq_len": 192,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 320,
    "num_classes": 7,
    "crop_size": 320,
    "batch_size_image_level": 128,
    "batch_size_patient_level": 32,
    "num_workers": 4,
    "prefetch_factor": 2,
}




## === cell 1
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

zero_mean = np.zeros(7, dtype=np.float32)
for uid, img_list in selected_image_dict.items():
    if len(img_list) == 0:
        feature_array_dict[uid] = np.zeros(
            (0, config["feature_size"]), dtype=np.float32
        )
        for idx, label in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_dict["row_id"].append(f"{uid}_{label}")
            submission_dict["fractured"].append(float(zero_mean[idx]))

master_items = []
for uid in study_id_list:
    img_list = selected_image_dict.get(uid, [])
    if len(img_list) == 0:
        continue
    for sl in img_list:
        master_items.append((uid, sl))

feat_lists = {uid: [] for uid in study_id_list if selected_image_dict.get(uid)}
preds_lists = {uid: [] for uid in feat_lists}
sum_preds = {uid: np.zeros(7, dtype=np.float32) for uid in feat_lists}
cnt_preds = {uid: 0 for uid in feat_lists}

master_dataset = CSFMasterImageDataset(
    items=master_items,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
)
master_loader = DataLoader(
    master_dataset,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    num_workers=config["num_workers"],  # now >0
    pin_memory=True,
    prefetch_factor=config["prefetch_factor"],
)

amp_autocast = (
    torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cuda"
    else None
)

for batch in tqdm(master_loader, desc="Processing all slices"):
    uids, imgs = batch  # uids is a list, imgs is a tensor
    imgs = imgs.to(device, non_blocking=True)
    with torch.no_grad():
        if amp_autocast:
            with amp_autocast:
                feats, preds = lv1_model(imgs)
        else:
            feats, preds = lv1_model(imgs)
    feats_np = feats.cpu().numpy()
    preds_np = preds.sigmoid().cpu().numpy()
    for i, uid in enumerate(uids):
        feat_lists[uid].append(feats_np[i])
        sum_preds[uid] += preds_np[i]
        cnt_preds[uid] += 1
        preds_lists[uid].append(preds_np[i])

for uid in study_id_list:
    if uid not in feat_lists:
        continue  # already handled empty‑slice case
    feature_array = (
        np.vstack(feat_lists[uid])
        if feat_lists[uid]
        else np.zeros((0, config["feature_size"]), dtype=np.float32)
    )
    feature_array_dict[uid] = feature_array
    if preds_lists[uid]:
        median_preds = np.median(np.stack(preds_lists[uid]), axis=0)
    else:
        median_preds = zero_mean
    for idx, label in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{label}")
        submission_dict["fractured"].append(float(median_preds[idx]))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3332701081.py in <cell line: 0>()
      2 feature_array_dict = {}
      3 
----> 4 zero_mean = np.zeros(7, dtype=np.float32)
      5 for uid, img_list in selected_image_dict.items():
      6     if len(img_list) == 0:

NameError: name 'np' is not defined

## === cell 2
instance_dataset = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
instance_loader = DataLoader(
    dataset=instance_dataset,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=True,
    num_workers=config["num_workers"],  # now >0
    prefetch_factor=config["prefetch_factor"],
)

for features, uids in tqdm(instance_loader, desc="Patient overall"):
    features = features.to(device, non_blocking=True)
    with torch.no_grad():
        if amp_autocast:
            with amp_autocast:
                preds = lv2_model(features)  # (batch, 1)
        else:
            preds = lv2_model(features)
        preds = preds.squeeze(1).sigmoid().cpu().numpy()
    for uid, p in zip(uids, preds):
        submission_dict["row_id"].append(f"{uid}_patient_overall")
        submission_dict["fractured"].append(float(p))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2908998138.py in <cell line: 0>()
----> 1 instance_dataset = CSFInstanceDataset(
      2     feature_array_dict=feature_array_dict,
      3     study_id_list=study_id_list,
      4     seq_len=config["seq_len"],
      5 )

NameError: name 'CSFInstanceDataset' is not defined

## === cell 3
sub_df = pd.DataFrame(submission_dict)
sub_df = sub_df.sort_values("row_id").reset_index(drop=True)
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/39147344.py in <cell line: 0>()
----> 1 sub_df = pd.DataFrame(submission_dict)
      2 sub_df = sub_df.sort_values("row_id").reset_index(drop=True)
      3 sub_df.to_csv("submission.csv", index=False)

NameError: name 'pd' is not defined
