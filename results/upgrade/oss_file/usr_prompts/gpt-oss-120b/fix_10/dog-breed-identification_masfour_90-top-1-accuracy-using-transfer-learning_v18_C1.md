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

5.09136

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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    print("scikit‑learn not available; will fall back to dummy predictions.")




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
decoder_path = "/root/decoder_weights.npy"
if TF_AVAILABLE and os.path.exists(decoder_path):
    decoder_weights = np.load(decoder_path, allow_pickle=True).item()
    for layer_name, layer_weights in decoder_weights.items():
        try:
            model.get_layer(layer_name).set_weights(layer_weights)
        except Exception:
            pass
else:
    if TF_AVAILABLE:
        print("Decoder weights not found – the model will use ImageNet weights only.")
    else:
        print(
            "Running without TensorFlow model; will train a fallback model if possible."
        )




## === cell 5
fallback_centroids = None  # deprecated, kept for safety
fallback_model = None  # generic fallback (LogisticRegression or RandomForest)

if not TF_AVAILABLE and SKLEARN_AVAILABLE:
    print("Training fallback model on down‑sampled images...")
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
        vec = (img_rgb.flatten() / 255.0).astype(np.float64)  # (tiny_h*tiny_w*3,)
        breed = labels.loc[labels["id"] == fname[:-4], "breed"].values
        if len(breed) == 0:
            continue
        cls_idx = class_to_idx[breed[0]]
        X_list.append(vec)
        y_list.append(cls_idx)

    if X_list:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        lr = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            multi_class="multinomial",
            solver="saga",
            C=1.0,
            verbose=0,
        )
        lr.fit(X, y)
        fallback_model = lr
        print("LogisticRegression fallback model training complete.")
    else:
        print("No training data found for fallback model.")
else:
    fallback_model = None




## === cell 6
sample_sub = pd.read_csv(sample_submission_path)
submission = pd.DataFrame(columns=sample_sub.columns)

test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
)

batch_size = 32
preds = []

tiny_h, tiny_w = 64, 64  # used for fallback models

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
    batch_array = np.stack(batch_imgs, axis=0)

    if TF_AVAILABLE:
        batch_pred = model.predict(batch_array, verbose=0)
    elif fallback_model is not None:
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
        batch_pred = fallback_model.predict_proba(tiny_batch)
    elif fallback_centroids is not None:
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
        dists = np.linalg.norm(
            tiny_batch[:, None, :] - fallback_centroids[None, :, :], axis=2
        )
        logits = -dists
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        batch_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
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
