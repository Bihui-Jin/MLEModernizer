# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.198602

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
train_root = vesuvius_data_path + "train/"
available_fragments = [
    int(d)
    for d in os.listdir(train_root)
    if os.path.isdir(os.path.join(train_root, d)) and d.isdigit()
]

test_patches = {
    1: [2000, 400, 2500, 1000],
    2: [1500, 1200, 2200, 1000],
    3: [1800, 500, 2300, 1200],
}

train_patches = [[1, [200, 1500, 4500, 6500]]]

fig, ax = plt.subplots(
    1, len(available_fragments), figsize=(5 * len(available_fragments), 5)
)
if len(available_fragments) == 1:
    ax = [ax]  # make iterable
for idx, fragment in enumerate(available_fragments):
    mask, label = load_mask_label(fragment)
    ax[idx].imshow(label, cmap="gray")
    ax[idx].imshow(mask, cmap="gray", alpha=0.5)

    test_rect = test_patches.get(fragment)
    if test_rect:
        test_patch = matplotlib.patches.Rectangle(
            (test_rect[0], test_rect[1]),
            test_rect[2],
            test_rect[3],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax[idx].add_patch(test_patch)

    for train_patch in train_patches:
        if train_patch[0] == fragment:
            train_rect = train_patch[1]
            tp = matplotlib.patches.Rectangle(
                (train_rect[0], train_rect[1]),
                train_rect[2],
                train_rect[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax[idx].add_patch(tp)

plt.show()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1451725743.py in <cell line: 0>()
      1 # Visualise only fragments that actually exist in the training data
----> 2 train_root = vesuvius_data_path + "train/"
      3 available_fragments = [
      4     int(d)
      5     for d in os.listdir(train_root)

NameError: name 'vesuvius_data_path' is not defined

## === cell 1
def generate_predictions(fragment="a", stride=99, patch=None):
    """Return predictions for the test fragments.
    Handles missing mask files by creating a zero mask of the expected size.
    """
    t_load = time.time()
    if fragment in ["a", "b"]:  # test set
        mask_filepath = vesuvius_data_path + f"test/{fragment}/mask.png"
        if os.path.exists(mask_filepath):
            mask = torch.from_numpy(
                np.array(PIL.Image.open(mask_filepath).convert("1"))
            )
        else:
            sample_stack_path = vesuvius_data_path + f"test/{fragment}/surface_volume/"
            first_tif = sorted(
                [f for f in os.listdir(sample_stack_path) if f.endswith(".tif")]
            )[0]
            img = np.array(PIL.Image.open(os.path.join(sample_stack_path, first_tif)))
            mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.uint8)
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, folder="test")
    elif fragment in [1, 2, 3]:
        mask, label = load_mask_label(fragment, rect=patch)
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect=patch)
    else:
        raise ValueError(f"Unexpected fragment identifier: {fragment}")

    test_pixels = get_pixels(mask, img_size, stride=stride)
    print(
        f"Mask size {mask.shape}, Striding by {stride}, we have {len(test_pixels)} pixels to test | Time to load: {time.time() - t_load:.2f}s"
    )
    test_dataset = SubvolumeDataset(image_stack, mask, test_pixels, img_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    print(
        f"Length of test dataloader: {len(test_dataloader)} batches of size {batch_size}"
    )

    t_generate = time.time()
    output = torch.zeros_like(mask).float()
    model.eval()
    radius = stride // 2
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(test_dataloader):
            preds = model(subvolumes.to(device).permute(0, 2, 3, 1).unsqueeze(dim=1))
            for j, value in enumerate(preds):
                y, x = test_pixels[i * batch_size + j]
                output[y - radius : y + radius + 1, x - radius : x + radius + 1] = value
    print(f"Generated pixels!! Time taken: {time.time() - t_generate:.2f}s")
    return output.cpu()




## === cell 2
test_root = vesuvius_data_path + "test/"
test_fragments = [
    d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))
]

pred_list = []
for fragment in test_fragments:
    train_pred = generate_predictions(fragment=fragment, stride=19)
    threshold = best_threshold if "best_threshold" in globals() else 0.5
    plt.imshow(train_pred.gt(threshold), cmap="gray")
    plt.title(f"Fragment {fragment}")
    plt.show()
    starts_ix, lengths = rle(train_pred, thr=threshold)
    inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
    pred_list.append({"Id": fragment, "Predicted": inklabels_rle})



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/506681210.py in <cell line: 0>()
      1 # Determine which test fragments actually exist
----> 2 test_root = vesuvius_data_path + "test/"
      3 test_fragments = [
      4     d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))
      5 ]

NameError: name 'vesuvius_data_path' is not defined

## === cell 3
import pandas as pd

pd.DataFrame(pred_list).to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1841997450.py in <cell line: 0>()
      1 import pandas as pd
      2 
----> 3 pd.DataFrame(pred_list).to_csv("submission.csv", index=False)

NameError: name 'pred_list' is not defined
