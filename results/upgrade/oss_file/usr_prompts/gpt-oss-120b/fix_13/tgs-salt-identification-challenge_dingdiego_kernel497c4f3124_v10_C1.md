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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.8219558632626561

# 6. Current score

0.1301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing pydensecrf import by making it optional and falling back to a no‑op CRF implementation, correct the data‑loading paths, ensure all required libraries are imported, and adjust the test image directory. These changes unblock the script, let it run end‑to‑end, and produce a valid .csv submission while keeping the original workflow intact.'
- What this solution (achieved 0.0305) has done: 'I replace the mask‑refinement loop with a lightweight mask generation that skips the DenseCRF (which can degrade performance) and instead builds a simple threshold‑based mask from each test image. This keeps the overall workflow unchanged while producing a valid submission and moving the score from 0.0 toward the target.'
- What this solution (achieved 0.0647) has done: 'I add Otsu‑based thresholding with small‑object/hole cleaning to generate more realistic masks, which should raise the mean average precision toward the target while keeping the original workflow unchanged. The new imports are placed with the other imports, and the `simple_threshold_mask` function is updated to use this smarter thresholding.'
- What this solution (achieved 0.0994) has done: 'I enhance the unsupervised mask creation by adding a light Gaussian blur before Otsu, applying opening/closing to clean the mask, and keeping only the largest connected component (salt is usually a single region). These modest image‑processing tweaks stay within the original workflow but should raise the mean average precision closer to the target score.'
- What this solution (achieved 0.0) has done: 'The fix normalizes the input image before applying `equalize_adapthist` (the function requires float images in the [0, 1] range) and removes the auxiliary “z” column before writing the submission so the CSV matches Kaggle’s required `['id','rle_mask']` schema. No core logic is changed; only the preprocessing and final save steps are corrected, which resolves the runtime error and produces a valid submission file.'
- What this solution (achieved 0.1301) has done: 'I added a safe clipping step before adaptive histogram equalization and wrapped the equalize_adapthist call in a fallback that uses simple histogram equalization when the image values are out of the expected range. This prevents the ValueError that stopped the pipeline, allowing mask generation to continue and a valid CSV submission to be written, while keeping the original processing logic unchanged.'
- What this solution (achieved 0.1301) has done: 'I modestly enhance the mask‑generation step by adding a complementary local‑threshold mask (which often captures salt regions Otsu misses) and merging it with the Otsu mask. This small change keeps the overall workflow intact while improving mask recall, which should raise the mean average precision toward the target score.'
- What this solution (achieved 0.1301) has done: 'I adjust the mask‑generation routine to be less aggressive (keep all plausible salt regions instead of only the largest one), lower the size thresholds for small‑object removal, and add a secondary, more‑blurred Otsu mask that is unioned with the existing mask. These modest tweaks preserve the overall pipeline while increasing recall, which should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.1301) has done: 'I modestly tighten the post‑processing in `simple_threshold_mask` by (1) using a slightly larger size filter for small objects/holes (scaled with depth) to cut down spurious regions, (2) keeping only the largest connected component (salt is usually a single blob), and (3) adjusting the Gaussian blur scale a bit for better smoothing. These small tweaks preserve the overall pipeline while raising precision, moving the score upward toward the target.'
- What this solution (achieved 0.1301) has done: 'I loosen the post‑processing in **simple_threshold_mask** to increase recall:‑ use a smaller Gaussian blur, keep all mask components (remove the “largest‑region only” step), and apply a stronger dilation. These tweaks stay within the existing pipeline and are expected to raise the mean average precision toward the target without altering the overall workflow.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu, gaussian, threshold_local
from skimage import morphology, measure, exposure, img_as_float
import os

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    CRF_AVAILABLE = True
except ImportError:
    dcrf = None
    CRF_AVAILABLE = False




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run‑length encoded mask string into a 101×101 binary numpy array.
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 2
base_submission_path = os.path.abspath("../input/sample_submission.csv")
if not os.path.exists(base_submission_path):
    base_submission_path = os.path.abspath("../input/k-fold-merger/submission.csv")
df = pd.read_csv(base_submission_path)

test_path = os.path.abspath("../input/test/images/")
if not os.path.isdir(test_path):
    test_path = os.path.abspath(
        "../input/tgs-salt-identification-challenge/test/images/"
    )

depth_path_candidates = [
    os.path.abspath("../input/depths.csv"),
    os.path.abspath("../input/tgs-salt-identification-challenge/depths.csv"),
    os.path.abspath("../input/k-fold-merger/depths.csv"),
]
depth_path = next((p for p in depth_path_candidates if os.path.exists(p)), None)
if depth_path is not None:
    depths_df = pd.read_csv(depth_path)
else:
    depths_df = pd.DataFrame(columns=["id", "z"])

df = df.merge(depths_df, on="id", how="left")

median_depth = df["z"].median() if not df["z"].isnull().all() else 1.0




## === cell 3
def crf(original_image, mask_img):
    """
    Apply DenseCRF to refine a binary mask.
    If pydensecrf is not installed, simply return the original mask.
    """
    if not CRF_AVAILABLE:
        return mask_img.astype(np.uint8)

    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
    )
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2  # background / salt
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
        np.uint8
    )




## === cell 4
def rle_encode(im):
    """
    Encode a binary mask (numpy array) as a run‑length string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
def simple_threshold_mask(image, depth=None):
    """
    Generate a binary mask using adaptive histogram equalization, Otsu thresholds at two
    blur scales, a local‑threshold mask, and light morphological cleaning.
    The function now keeps **all** candidate regions (instead of only the largest)
    and uses a stronger dilation to improve recall.
    """
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = img_as_float(gray)
    gray = np.clip(gray, 0.0, 1.0)

    try:
        gray = exposure.equalize_adapthist(gray)
    except ValueError:
        gray = exposure.equalize_hist(gray)

    if depth is not None and not np.isnan(depth):
        scale = np.sqrt(depth / median_depth)
        sigma = 1.0 * scale  # smaller blur to retain details
        min_obj = max(6, int(12 * scale))
        hole_area = max(6, int(12 * scale))
    else:
        sigma = 1.0
        min_obj = 12
        hole_area = 12

    blurred = gaussian(gray, sigma=sigma)
    thresh = threshold_otsu(blurred)
    otsu_mask = (blurred > thresh).astype(np.uint8)

    blurred2 = gaussian(gray, sigma=sigma * 2)
    thresh2 = threshold_otsu(blurred2)
    otsu_mask2 = (blurred2 > thresh2).astype(np.uint8)

    local_thresh = threshold_local(gray, block_size=45, offset=0)
    local_mask = (gray > local_thresh).astype(np.uint8)

    mask = np.logical_or.reduce([otsu_mask, otsu_mask2, local_mask]).astype(np.uint8)

    selem = morphology.disk(2)
    mask = morphology.opening(mask, selem)
    mask = morphology.closing(mask, selem)

    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_obj)
    mask = morphology.remove_small_holes(mask, area_threshold=hole_area)


    mask = morphology.dilation(mask, morphology.disk(5))

    return mask.astype(np.uint8)




## === cell 6
for i in tqdm(range(df.shape[0]), desc="Generating masks"):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_path):
        continue
    orig_img = imread(img_path)

    depth_val = df.loc[i, "z"] if "z" in df.columns else None
    mask = simple_threshold_mask(orig_img, depth=depth_val)

    if CRF_AVAILABLE:
        mask = crf(orig_img, mask)

    df.at[i, "rle_mask"] = rle_encode(mask)




## === cell 7
output_path = "crf_correction.csv"
submission_df = df[["id", "rle_mask"]]
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
