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

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.28106

# 6. Current score

5.10396

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.03981) has done: 'I wrap the TensorFlow import in a safe try/except and fall back to a lightweight dummy model that returns uniform probabilities, ensuring the script runs without the protobuf error. I also filter the test directory to only process image files and guarantee the submission DataFrame matches the sample‑submission columns, so a valid `submission.csv` is written.'
- What this solution (achieved 5.03794) has done: 'I fixed the TensorFlow import failure by keeping the original TF‑based model only when TF loads successfully, and added a lightweight scikit‑learn logistic‑regression model that trains on down‑sampled training images when TF is unavailable. The new model learns from the actual labels, so predictions are far better than the uniform dummy probabilities, reducing the log‑loss toward the target. I also adjusted the inference loop to use whichever model is available and ensured the submission file keeps the exact required columns.'
- What this solution (achieved 5.03157) has done: 'I fix the TensorFlow import failure by keeping the lightweight sklearn fallback, but replace the simple LogisticRegression model with a centroid‑based classifier that uses down‑sampled images to compute a mean vector for each breed and predicts probabilities from the Euclidean distance to these centroids. This change improves prediction quality without altering the overall pipeline, ensures a valid CSV is written, and moves the log‑loss closer to the target.'
- What this solution (achieved 4.99815) has done: 'I replace the centroid fallback with a lightweight LogisticRegression model trained on down‑sampled images, and update the inference step to use this classifier when TensorFlow is unavailable. This keeps the overall pipeline intact, fixes the poor‑performing fallback, and should substantially lower the log‑loss toward the target while still producing a correct `submission.csv`.'
- What this solution (achieved 5.04694) has done: 'I replaced the LogisticRegression fallback with a lightweight centroid‑based classifier that stores the mean feature vector for each breed using 64×64 down‑sampled images. During inference the code now computes Euclidean distances from each test image to every class centroid, converts the negative distances to a soft‑max probability distribution, and writes these probabilities to the required submission CSV. This fix removes the poor‑performing logistic model, fixes the prediction path, and should lower the log‑loss toward the target while keeping the original pipeline intact.'
- What this solution (achieved 5.1332) has done: 'Implemented a lightweight `LogisticRegression` fallback model (trained on 64×64 down‑sampled images) to replace the centroid classifier, and updated the inference logic to use this model when TensorFlow is unavailable. Added proper handling for cases where the fallback model isn’t trained, keeping the dummy uniform predictions as a safe default.'
- What this solution (achieved 5.11832) has done: 'Implemented a more effective fallback classifier using RandomForest instead of the weaker LogisticRegression. Added the necessary import and training logic, while keeping the TensorFlow branch untouched. This improves predictive quality when TensorFlow isn’t available, ensuring a valid `submission.csv` is written and moving the log‑loss closer to the target. No other core logic was altered.'
- What this solution (achieved 5.09136) has done: 'I fixed the fallback path so it now trains a multinomial LogisticRegression (which works better on the flattened 64 × 64 RGB vectors) and uses the exact same preprocessing during inference. The TensorFlow branch is left untouched; when TF cannot be imported the script reliably fall back to the improved sklearn model and still write a correctly‑formatted `submission.csv`. This change removes the previous poor RandomForest fallback and should lower the log‑loss toward the target.'
- What this solution (achieved 4.99868) has done: 'I fixed the import failure by ensuring scikit‑learn utilities are available, replaced the weak LogisticRegression fallback with a more expressive K‑Nearest‑Neighbors pipeline (StandardScaler + KNeighborsClassifier) which uses the same 64 × 64 RGB flattening as training, and kept the rest of the pipeline unchanged. This improves the quality of the fallback predictions, bringing the log‑loss closer to the target while still producing a correctly‑formatted `submission.csv`.'
- What this solution (achieved 5.02955) has done: 'The fix replaces the weak K‑Nearest‑Neighbors fallback with a stronger RandomForest classifier trained on the same 64×64 RGB pixel vectors. RandomForest handles high‑dimensional data better and usually yields much lower log‑loss, moving the score toward the target while keeping the original pipeline unchanged. No other logic is altered; the script now always trains and uses the RandomForest model when TensorFlow is unavailable, and still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 5.04678) has done: 'The script’s TensorFlow import fails, so it falls back to a scikit‑learn model. The previous fallback used a RandomForest which gives a high log‑loss. We replace that with a much stronger K‑Nearest‑Neighbors classifier (k‑1, distance‑weighted) that works on the same 64×64 flattened RGB vectors. This change keeps the overall pipeline unchanged, fixes the model‑training step, and provides better‑calibrated probabilities, moving the score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 5.0022) has done: 'Implemented a stronger fallback classifier: switched from a 1‑NN model to a balanced RandomForest with many trees, which yields much better probability estimates on down‑sampled 64×64 images. This improves prediction quality while keeping the original pipeline unchanged and still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 5.06006) has done: 'Implemented a more effective sklearn fallback: replaced the RandomForest with a scaled K‑Nearest‑Neighbors classifier (distance‑weighted) which generally yields better calibrated probabilities on flattened 64×64 RGB vectors. Added a StandardScaler in the pipeline and kept all other logic unchanged, ensuring the script runs without TensorFlow and writes a correct `submission.csv`. This modest change should lower the log‑loss toward the target while preserving the original workflow.'
- What this solution (achieved 5.10396) has done: 'I replace the sklearn‑based fallback with a pure‑numpy K‑Nearest‑Neighbors implementation that trains on the 64×64 down‑sampled images when TensorFlow cannot be used. This removes the dependency on scikit‑learn (which isn’t installed) and provides a more informed probability estimate than the uniform dummy, helping to lower the log‑loss while keeping the original pipeline intact. I also adjust the inference loop to use this new KNN fallback.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import (
        Dense,
        Activation,
        Dropout,
        BatchNormalization,
        Input,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.initializers import he_normal

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); using fallback model.")
    TF_AVAILABLE = False

SKLEARN_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
classes = np.unique(labels.breed)
classes_num = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}




## === cell 2
train_dir = "../input/dog-breed-identification/train"
test_dir = "../input/dog-breed-identification/test"
sample_submission_path = "../input/dog-breed-identification/sample_submission.csv"




## === cell 3
def dense_block(x, neurons, layer_no):
    x = Dense(neurons, kernel_initializer=he_normal(), name=f"topDense{layer_no}")(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x


def create_model(input_shape):
    input_layer = Input(input_shape, name="input_layer")
    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


height, width, channels_num = 512, 512, 3

if TF_AVAILABLE:
    model = create_model((height, width, channels_num))
    optimizer = Adam(learning_rate=0.004)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
else:
    model = None




## === cell 4
fallback_X = None  # shape (n_samples, n_features)
fallback_y = None  # shape (n_samples,)
K_FALLBACK = 5  # number of neighbors used for prediction

if not TF_AVAILABLE:
    print("Training pure‑numpy KNN fallback model on down‑sampled images...")
    tiny_h, tiny_w = 64, 64
    train_files = sorted(
        [
            f
            for f in os.listdir(train_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    X_list = []
    y_list = []

    for fname in train_files:
        img_path = os.path.join(train_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.resize(img_bgr[:, :, ::-1], (tiny_w, tiny_h))
        vec = (img_rgb.flatten() / 255.0).astype(np.float32)
        breed = labels.loc[labels["id"] == fname[:-4], "breed"].values
        if len(breed) == 0:
            continue
        cls_idx = class_to_idx[breed[0]]
        X_list.append(vec)
        y_list.append(cls_idx)

    if X_list:
        fallback_X = np.stack(X_list, axis=0)  # (n_samples, n_features)
        fallback_y = np.array(y_list, dtype=np.int32)  # (n_samples,)
        print(f"KNN fallback trained on {fallback_X.shape[0]} images.")
    else:
        print("No training data found for fallback model.")
else:
    fallback_X = None
    fallback_y = None




## === cell 5
sample_sub = pd.read_csv(sample_submission_path)
submission = pd.DataFrame(columns=sample_sub.columns)

test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
)

batch_size = 32
preds = []

tiny_h, tiny_w = 64, 64  # used for fallback models


def knn_predict_proba(batch_vecs, X_train, y_train, k=K_FALLBACK):
    """
    Distance‑weighted KNN probability prediction.
    batch_vecs: (batch, n_features)
    X_train:   (n_samples, n_features)
    y_train:   (n_samples,)
    Returns:   (batch, classes_num) probability matrix.
    """
    a2 = np.sum(batch_vecs**2, axis=1, keepdims=True)  # (batch,1)
    b2 = np.sum(X_train**2, axis=1)  # (n_samples,)
    ab = batch_vecs @ X_train.T  # (batch, n_samples)
    dists = np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0.0))  # (batch, n_samples)

    neigh_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]  # (batch, k)
    neigh_dist = np.take_along_axis(dists, neigh_idx, axis=1)  # (batch, k)
    neigh_labels = y_train[neigh_idx]  # (batch, k)

    eps = 1e-8
    weights = 1.0 / (neigh_dist + eps)  # (batch, k)

    prob = np.zeros((batch_vecs.shape[0], classes_num), dtype=np.float32)
    for i in range(k):
        np.add.at(prob, (np.arange(prob.shape[0]), neigh_labels[:, i]), weights[:, i])

    prob_sum = prob.sum(axis=1, keepdims=True) + eps
    prob /= prob_sum
    return prob


for start_idx in range(0, len(test_files), batch_size):
    batch_files = test_files[start_idx : start_idx + batch_size]
    batch_imgs = []
    for fname in batch_files:
        img_path = os.path.join(test_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        img_rgb = cv2.resize(img_bgr[:, :, ::-1], (width, height)) / 255.0
        batch_imgs.append(img_rgb)
    batch_array = np.stack(batch_imgs, axis=0)  # (batch, H, W, C)

    if TF_AVAILABLE:
        batch_pred = model.predict(batch_array, verbose=0)
    elif fallback_X is not None and fallback_y is not None:
        tiny_batch = (
            np.stack(
                [
                    cv2.resize((img * 255).astype(np.uint8), (tiny_w, tiny_h)).flatten()
                    for img in batch_array
                ],
                axis=0,
            )
            / 255.0
        )
        batch_pred = knn_predict_proba(tiny_batch, fallback_X, fallback_y, k=K_FALLBACK)
    else:
        batch_pred = np.full((len(batch_files), classes_num), 1.0 / classes_num)

    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

submission["id"] = [fname[:-4] for fname in test_files]
for idx, breed in enumerate(classes):
    submission[breed] = preds[:, idx]

submission = submission[sample_sub.columns]

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with shape {submission.shape}")
