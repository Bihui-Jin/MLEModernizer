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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0
tqdm==4.67.1

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

0.68137

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.1091) has done: 'I fixed the import errors, replaced the old Keras imports with `tensorflow.keras`, corrected the missing `load_img` and tqdm utilities, added safe handling for missing depth values, and ensured the model, training, prediction, and submission steps all run sequentially and produce a valid `submission.csv` file.'
- What this solution (achieved 0.5221) has done: 'I added an environment flag to avoid the protobuf import error, fixed the ModelCheckpoint filename to meet Keras requirements, and introduced a quick validation‑threshold search so the model uses the best binary cutoff (based on simple IoU) before creating the final submission. This resolves the runtime crashes and should lift the validation performance toward the target score.'

# 9. Code solution

## === cell 0
im_width = 128
im_height = 128
border = 5
im_chan = 2  # original + cumulative sum channel
n_features = 1  # depth feature

path_train = "../input/train/"
path_test = "../input/test/"



## === cell 1
df_depths = pd.read_csv("../input/depths.csv", index_col="id")
df_depths.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1974133975.py in <cell line: 0>()
----> 1 df_depths = pd.read_csv("../input/depths.csv", index_col="id")
      2 df_depths.head()
      3 

NameError: name 'pd' is not defined

## === cell 2
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(30, 15))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img(
        os.path.join(path_train, "images", img_name + ".png"), color_mode="grayscale"
    )
    img_mask = load_img(
        os.path.join(path_train, "masks", img_name + ".png"), color_mode="grayscale"
    )
    img = np.array(img)
    img_cumsum = (np.float32(img) - img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)

    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 2)
    plt.imshow(img, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3)
    plt.imshow(img_mask, cmap="gray")
plt.show()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1882732197.py in <cell line: 0>()
      1 ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
----> 2 plt.figure(figsize=(30, 15))
      3 for j, img_name in enumerate(ids):
      4     q = j + 1
      5     img = load_img(

NameError: name 'plt' is not defined

## === cell 3
train_ids = next(os.walk(os.path.join(path_train, "images")))[2]
test_ids = next(os.walk(os.path.join(path_test, "images")))[2]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1754668587.py in <cell line: 0>()
----> 1 train_ids = next(os.walk(os.path.join(path_train, "images")))[2]
      2 test_ids = next(os.walk(os.path.join(path_test, "images")))[2]
      3 

NameError: name 'os' is not defined

## === cell 4
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)

print("Getting and resizing train images and masks ...")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    depth_val = (
        df_depths.loc[id_.replace(".png", ""), "z"]
        if id_.replace(".png", "") in df_depths.index
        else 0.0
    )
    X_feat[n, 0] = depth_val

    img = load_img(os.path.join(path_train, "images", id_), color_mode="grayscale")
    x_img = img_to_array(img)  # (h, w, 1)
    x_img = resize(
        x_img, (im_height, im_width, 1), mode="constant", preserve_range=True
    )

    x_center = x_img[border:-border, border:-border, 0]
    x_center_mean = x_center.mean()
    x_csum = (np.float32(x_img) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border, 0].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border, 0].std())

    mask = load_img(os.path.join(path_train, "masks", id_), color_mode="grayscale")
    mask = img_to_array(mask)
    mask = resize(mask, (im_height, im_width, 1), mode="constant", preserve_range=True)

    X[n, ..., 0] = x_img.squeeze() / 255.0
    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255.0
print("Done!")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1908587320.py in <cell line: 0>()
----> 1 X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
      2 y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
      3 X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
      4 
      5 print("Getting and resizing train images and masks ...")

NameError: name 'np' is not defined

## === cell 5
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(
    X, X_feat, y, test_size=0.15, random_state=42
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/330319045.py in <cell line: 0>()
----> 1 X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(
      2     X, X_feat, y, test_size=0.15, random_state=42
      3 )
      4 

NameError: name 'train_test_split' is not defined

## === cell 6
x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
x_feat_std = X_feat_train.std(axis=0, keepdims=True) + 1e-8

X_feat_train = (X_feat_train - x_feat_mean) / x_feat_std
X_feat_valid = (X_feat_valid - x_feat_mean) / x_feat_std



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2733396815.py in <cell line: 0>()
----> 1 x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
      2 x_feat_std = X_feat_train.std(axis=0, keepdims=True) + 1e-8
      3 
      4 X_feat_train = (X_feat_train - x_feat_mean) / x_feat_std
      5 X_feat_valid = (X_feat_valid - x_feat_mean) / x_feat_std

NameError: name 'X_feat_train' is not defined

## === cell 7
ix = random.randint(0, len(X_train) - 1)
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(X_train[ix, ..., 0], cmap="seismic")
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[0].set_title("Seismic")

ax[1].imshow(X_train[ix, ..., 1], cmap="seismic")
if has_mask:
    ax[1].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[1].set_title("Seismic cumsum")

ax[2].imshow(y_train[ix].squeeze(), cmap="gray")
ax[2].set_title("Salt")
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1576955273.py in <cell line: 0>()
----> 1 ix = random.randint(0, len(X_train) - 1)
      2 has_mask = y_train[ix].max() > 0
      3 
      4 fig, ax = plt.subplots(1, 3, figsize=(20, 10))
      5 ax[0].imshow(X_train[ix, ..., 0], cmap="seismic")

NameError: name 'random' is not defined

## === cell 8
def mean_iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.int32)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    metric.update_state(y_true, y_pred)
    return metric.result()




## === cell 9
input_img = Input((im_height, im_width, im_chan), name="img")
input_feat = Input((n_features,), name="feat")

c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(input_img)
c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
p4 = MaxPooling2D((2, 2))(c4)

f_repeat = RepeatVector(8 * 8)(input_feat)
f_reshape = Reshape((8, 8, n_features))(f_repeat)

p4_feat = concatenate([p4, f_reshape], axis=-1)

c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(p4_feat)
c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = concatenate([u6, c4], axis=-1)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = concatenate([u7, c3], axis=-1)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = concatenate([u8, c2], axis=-1)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=-1)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

output = Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = Model(inputs=[input_img, input_feat], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3052463964.py in <cell line: 0>()
----> 1 input_img = Input((im_height, im_width, im_chan), name="img")
      2 input_feat = Input((n_features,), name="feat")
      3 
      4 c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(input_img)
      5 c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)

NameError: name 'Input' is not defined

## === cell 10
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(
        "model-tgs-salt-1.weights.h5",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]

results = model.fit(
    {"img": X_train, "feat": X_feat_train},
    y_train,
    batch_size=16,
    epochs=50,
    callbacks=callbacks,
    validation_data=({"img": X_valid, "feat": X_feat_valid}, y_valid),
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3640394912.py in <cell line: 0>()
      1 callbacks = [
----> 2     EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
      3     ReduceLROnPlateau(patience=3, verbose=1),
      4     ModelCheckpoint(
      5         "model-tgs-salt-1.weights.h5",

NameError: name 'EarlyStopping' is not defined

## === cell 11
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []

print("Getting and resizing test images ...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    depth_val = (
        df_depths.loc[id_.replace(".png", ""), "z"]
        if id_.replace(".png", "") in df_depths.index
        else 0.0
    )
    X_feat_test[n, 0] = depth_val

    img = load_img(os.path.join(path_test, "images", id_), color_mode="grayscale")
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])

    x = resize(x, (im_height, im_width, 1), mode="constant", preserve_range=True)

    x_center = x[border:-border, border:-border, 0]
    x_center_mean = x_center.mean()
    x_csum = (np.float32(x) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border, 0].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border, 0].std())

    X_test[n, ..., 0] = x.squeeze() / 255.0
    X_test[n, ..., 1] = x_csum.squeeze()
print("Done!")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/819708740.py in <cell line: 0>()
----> 1 X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
      2 X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
      3 sizes_test = []
      4 
      5 print("Getting and resizing test images ...")

NameError: name 'np' is not defined

## === cell 12
X_feat_test = (X_feat_test - x_feat_mean) / x_feat_std



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1756896648.py in <cell line: 0>()
----> 1 X_feat_test = (X_feat_test - x_feat_mean) / x_feat_std
      2 

NameError: name 'X_feat_test' is not defined

## === cell 13
val_loss = model.evaluate({"img": X_valid, "feat": X_feat_valid}, y_valid, verbose=1)
print("Validation loss:", val_loss)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4104872768.py in <cell line: 0>()
----> 1 val_loss = model.evaluate({"img": X_valid, "feat": X_feat_valid}, y_valid, verbose=1)
      2 print("Validation loss:", val_loss)
      3 

NameError: name 'model' is not defined

## === cell 14
preds_train = model.predict({"img": X_train, "feat": X_feat_train}, verbose=0)
preds_val = model.predict({"img": X_valid, "feat": X_feat_valid}, verbose=0)
preds_test = model.predict({"img": X_test, "feat": X_feat_test}, verbose=0)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4071183333.py in <cell line: 0>()
----> 1 preds_train = model.predict({"img": X_train, "feat": X_feat_train}, verbose=0)
      2 preds_val = model.predict({"img": X_valid, "feat": X_feat_valid}, verbose=0)
      3 preds_test = model.predict({"img": X_test, "feat": X_feat_test}, verbose=0)
      4 
      5 

NameError: name 'model' is not defined

## === cell 15
def simple_iou(y_true, y_pred_bin):
    """Pixel‑wise IoU for a single image."""
    intersection = np.logical_and(y_true, y_pred_bin).sum()
    union = np.logical_or(y_true, y_pred_bin).sum()
    if union == 0:
        return 1.0
    return intersection / union


def competition_metric(y_true_arr, y_pred_bin_arr, thresholds=None):
    """
    Approximate the competition metric:
    for each IoU threshold compute TP / (TP+FP+FN) and average over thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    ious = np.array(
        [
            simple_iou(y_true_arr[i].squeeze(), y_pred_bin_arr[i].squeeze())
            for i in range(len(y_true_arr))
        ]
    )

    per_thr_precisions = []
    for t in thresholds:
        TP = FP = FN = 0
        for i, iou in enumerate(ious):
            true_sum = y_true_arr[i].sum()
            pred_sum = y_pred_bin_arr[i].sum()
            if iou > t:
                TP += 1
            else:
                if true_sum > 0 and pred_sum > 0:
                    FP += 1
                    FN += 1
                elif pred_sum > 0:
                    FP += 1
                elif true_sum > 0:
                    FN += 1
        denom = TP + FP + FN
        per_thr_precisions.append(TP / denom if denom > 0 else 0.0)

    return np.mean(per_thr_precisions)


best_thr = 0.5
best_score = 0.0
for thr in np.arange(0.3, 0.71, 0.05):
    preds_val_bin = (preds_val > thr).astype(np.uint8)
    score = competition_metric(y_valid, preds_val_bin)
    if score > best_score:
        best_score = score
        best_thr = thr

print(
    f"Best threshold on validation (competition metric): {best_thr:.2f} with score {best_score:.4f}"
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1720606035.py in <cell line: 0>()
     49 best_thr = 0.5
     50 best_score = 0.0
---> 51 for thr in np.arange(0.3, 0.71, 0.05):
     52     preds_val_bin = (preds_val > thr).astype(np.uint8)
     53     score = competition_metric(y_valid, preds_val_bin)

NameError: name 'np' is not defined

## === cell 16
preds_train_t = (preds_train > best_thr).astype(np.uint8)
preds_val_t = (preds_val > best_thr).astype(np.uint8)
preds_test_t = (preds_test > best_thr).astype(np.uint8)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3695658153.py in <cell line: 0>()
----> 1 preds_train_t = (preds_train > best_thr).astype(np.uint8)
      2 preds_val_t = (preds_val > best_thr).astype(np.uint8)
      3 preds_test_t = (preds_test > best_thr).astype(np.uint8)
      4 

NameError: name 'preds_train' is not defined

## === cell 17
preds_test_upsampled = []
for i in tqdm(range(len(preds_test_t))):
    up = resize(
        np.squeeze(preds_test_t[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
    )
    preds_test_upsampled.append(up.astype(np.uint8))
print("Upsampling done.")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/53168737.py in <cell line: 0>()
      1 preds_test_upsampled = []
----> 2 for i in tqdm(range(len(preds_test_t))):
      3     up = resize(
      4         np.squeeze(preds_test_t[i]),
      5         (sizes_test[i][0], sizes_test[i][1]),

NameError: name 'tqdm' is not defined

## === cell 18
def RLenc(img, order="F", format=True):
    """Run‑length encoding.  img must be binary (0 / 1)."""
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1  # 1‑based indexing
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    if r != 0:
        runs.append((pos, r))

    if not format:
        return runs
    return " ".join(f"{p} {l}" for p, l in runs)


pred_dict = {}
for i, fn in enumerate(test_ids):
    mask = preds_test_upsampled[i]
    rle = RLenc(mask)
    pred_dict[fn[:-4]] = rle



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1739414789.py in <cell line: 0>()
     23 
     24 pred_dict = {}
---> 25 for i, fn in enumerate(test_ids):
     26     mask = preds_test_upsampled[i]
     27     rle = RLenc(mask)

NameError: name 'test_ids' is not defined

## === cell 19
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv", index=True)
print("Submission file written to submission.csv")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/967654881.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      2 sub.index.name = "id"
      3 sub.columns = ["rle_mask"]
      4 sub.to_csv("submission.csv", index=True)
      5 print("Submission file written to submission.csv")

NameError: name 'pd' is not defined
