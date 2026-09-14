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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

2.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6242067089755213

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix two execution blockers so the notebook can run end-to-end and actually write `submission.csv`: (1) the protobuf/TensorFlow import crash (your environment is effectively Python 3 + TF2, despite the “2.7” note), and (2) the TTA pipeline’s reliance on `ImageDataGenerator.principal_components`, which doesn’t exist unless ZCA whitening is enabled. I make `_make_tta_step_tf` robust by safely reading optional attributes and standardizing only with what’s present, keeping the same TTA core logic. I also add a minimal attempt to load the provided saved models if the directories exist (falling back to the current small CNNs if not), which is score-improving but still within your intended ensemble design. Finally, I ensure predictions are always produced and the submission format matches `sample_submission.csv`.'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (which commonly triggers `MessageFactory.GetPrototype` errors with TF’s bundled protobuf) and by importing TensorFlow only after setting safe environment flags. This is an execution blocker preventing any submission from being generated in the current environment, so fixing it is required before score can improve. I also add a small robustness fix so the gambler-model custom objects are loaded with the correct function names expected by `load_model`, without changing the model logic. Everything else (TTA logic, ensemble weighting, submission formatting/paths) is kept the same.'
- What this solution (achieved 0.10762) has done: 'The crash happens before any modeling code runs because TensorFlow’s bundled protobuf is incompatible with the runtime protobuf version, triggering `MessageFactory.GetPrototype` during `import tensorflow`. To unblock execution end-to-end, I add a minimal early import-order/env fix: force the pure-Python protobuf implementation **before** importing TensorFlow, and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` which is the most compatible combo on Kaggle. I keep your model/TTA/ensemble logic unchanged, and I also add a small safeguard so the submission is written to `/kaggle/working/submission.csv` even if `/kaggle/working` is missing (it be created). These changes are execution/stability fixes and should allow the intended saved-model ensemble to run, which should move accuracy up toward your target compared to the current mostly-fallback behavior.'
- What this solution (achieved 0.10762) has done: 'You’re currently crashing at `import tensorflow` due to a protobuf/TensorFlow incompatibility triggered by forcing the pure-Python protobuf implementation; removing those environment overrides is the minimal fix to unblock execution. I keep your ensemble/TTA/prediction logic the same, only adjusting the import-time environment so TensorFlow can load reliably in Kaggle’s TF2 runtime. I also add a tiny compatibility fallback: if TensorFlow still fails to import, the script stop with a clear error before doing any work (rather than failing later), ensuring you always get a deterministic failure mode. With TensorFlow import fixed, your saved-model loading + TTA ensemble should run end-to-end and should move accuracy upward from the current near-random score toward your target.'
- What this solution (achieved 0.11024) has done: 'The main blocker is that TensorFlow can’t import because `google.protobuf.pyext._message` is missing in this environment; no amount of `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` toggling fix a missing compiled extension. To make the script run end-to-end and still produce a valid `submission.csv`, I add a safe fallback path that does not require TensorFlow: it creates a reasonable baseline prediction using only `train.csv` label priors (majority-class), which score far above random and move toward your target compared to “no submission.” I keep all your TF/TTA/model code in place and only gate it behind a runtime check; if TensorFlow imports successfully in another environment, the original ensemble/TTA logic run unchanged. I also ensure all later cells have the variables they need even when running in the no-TF fallback, and always write `/kaggle/working/submission.csv` with correct columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"

if not os.path.exists(WORKING_DIR):
    try:
        os.makedirs(WORKING_DIR)
    except Exception:
        pass

np.random.seed(42)

TF_AVAILABLE = False
tf = None
layers = None
ImageDataGenerator = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras import layers  # noqa: F401
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: F401

    TF_AVAILABLE = True
    try:
        tf.random.set_seed(42)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

except Exception as e:
    print("WARNING: TensorFlow unavailable; falling back to non-TF submission.")
    print("TF import error:", repr(e))

print("TF_AVAILABLE:", TF_AVAILABLE)
print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Working dir exists:", os.path.exists(WORKING_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TF_AVAILABLE:

    def acc_gambler(y_true, y_pred):
        y_temp = y_pred[:, 1:]
        count = tf.constant(0, dtype=tf.int32)
        for i in range(tf.shape(y_true)[0]):
            if tf.equal(tf.argmax(y_temp[i]), tf.argmax(y_true[i])):
                count = count + 1
        return tf.cast(count, tf.float32) / tf.cast(tf.shape(y_true)[0], tf.float32)

    def loss_gambler(label_smoothing=0.0):
        def loss_gamb(y_true, y_pred):
            y_true = tf.add(
                y_true,
                tf.add(
                    tf.multiply(label_smoothing / 2.0, tf.add(1.0, -1.0 * y_true)),
                    tf.multiply(-1.0 * label_smoothing / 2.0, y_true),
                ),
            )
            y_temp = y_pred[:, 1:]
            f0 = y_pred[:, 0]
            lamb = tf.math.divide(
                tf.math.multiply(tf.reduce_sum(y_temp), tf.reduce_sum(y_temp)),
                tf.reduce_sum(tf.math.multiply(y_temp, y_temp)),
            )
            loss = tf.constant(0.0, dtype=tf.float32)
            num_classes = tf.shape(y_true)[1]
            batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
            for i in range(num_classes):
                loss = tf.add(
                    loss,
                    (-1.0 * (1.0 / batch_size))
                    * tf.reduce_sum(
                        y_true[:, i] * tf.math.log(y_temp[:, i] + f0 / lamb)
                    ),
                )
            return loss

        return loss_gamb




## === cell 2
if TF_AVAILABLE:

    def build_fallback_model(
        num_classes=5, input_shape=(448, 448, 3), name="fallback_cnn"
    ):
        inp = tf.keras.Input(shape=input_shape, dtype=tf.float32, name="image")
        x = layers.Rescaling(1.0 / 255.0)(inp)
        x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inp, out, name=name)
        return model

    MODEL1_DIR = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
    MODEL2_DIR = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"
    MODEL3_DIR = (
        "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-15-0.839"
    )

    def _safe_load_saved_model(model_dir, fallback_model, custom_objects=None):
        if os.path.exists(model_dir):
            try:
                m = tf.keras.models.load_model(
                    model_dir, custom_objects=custom_objects, compile=False
                )
                print("Loaded saved model:", model_dir)
                return m
            except Exception as e:
                print(
                    "WARNING: failed to load model from %s due to: %r. Using fallback."
                    % (model_dir, e)
                )
                return fallback_model
        else:
            print("WARNING: model dir not found:", model_dir, "Using fallback.")
            return fallback_model

    model_v1 = _safe_load_saved_model(
        MODEL1_DIR,
        build_fallback_model(name="model_v1_fallback"),
        custom_objects=None,
    )

    _gamb_loss_fn = loss_gambler()
    model_v2 = _safe_load_saved_model(
        MODEL2_DIR,
        build_fallback_model(name="model_v2_fallback"),
        custom_objects={
            "loss_gamb": _gamb_loss_fn,
            "loss_gambler": _gamb_loss_fn,
            "acc_gambler": acc_gambler,
        },
    )

    model_v3 = _safe_load_saved_model(
        MODEL3_DIR,
        build_fallback_model(name="model_v3_fallback"),
        custom_objects=None,
    )

    print("Loaded models:", model_v1.name, model_v2.name, model_v3.name)
    print("Model v1 output shape:", model_v1.output_shape)
else:
    model_v1 = None
    model_v2 = None
    model_v3 = None



## === cell 3
if TF_AVAILABLE:

    def random_crop(img, random_crop_size):
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y : (y + dy), x : (x + dx), :]

    def _batch_random_crops(batch_x, crop_length):
        b, h, w, c = batch_x.shape
        dy = dx = int(crop_length)
        if h < dy or w < dx:
            raise ValueError("Input smaller than crop: %r vs %r" % ((h, w), (dy, dx)))

        xs = np.random.randint(0, w - dx + 1, size=b)
        ys = np.random.randint(0, h - dy + 1, size=b)

        rows = ys[:, None] + np.arange(dy)[None, :]
        cols = xs[:, None] + np.arange(dx)[None, :]
        crops = batch_x[
            np.arange(b)[:, None, None], rows[:, :, None], cols[:, None, :], :
        ]
        return crops

    def _load_test_images_to_numpy(
        test_dir, test_ids, target_size=(512, 512), batch_size=64
    ):
        paths = [os.path.join(test_dir, img_id) for img_id in test_ids]
        h, w = target_size

        def _read_decode_resize(p):
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(
                img, [h, w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
            )
            img = tf.cast(img, tf.float32)
            return img

        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(_read_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        out = np.empty((len(test_ids), h, w, 3), dtype=np.float32)
        idx = 0
        for batch in ds:
            b = int(batch.shape[0])
            out[idx : idx + b] = batch.numpy()
            idx += b
        return out

    def _make_tta_step_tf(datagen, crop_length, seed_base=12345):
        mean = getattr(datagen, "mean", None)
        std = getattr(datagen, "std", None)
        principal_components = getattr(datagen, "principal_components", None)

        mean_t = tf.constant(mean, dtype=tf.float32) if mean is not None else None
        std_t = tf.constant(std, dtype=tf.float32) if std is not None else None
        pc_t = (
            tf.constant(principal_components, dtype=tf.float32)
            if principal_components is not None
            else None
        )

        @tf.function
        def _standardize_tf(x):
            if getattr(datagen, "rescale", None) is not None:
                x = x * tf.cast(datagen.rescale, tf.float32)
            if mean_t is not None:
                x = x - mean_t
            if std_t is not None:
                x = x / (std_t + tf.keras.backend.epsilon())
            if pc_t is not None:
                flat = tf.reshape(x, [tf.shape(x)[0], -1])
                flat = tf.matmul(flat, pc_t)
                x = tf.reshape(flat, tf.shape(x))
            return x

        @tf.function
        def _tta_batch(batch, pass_id):
            b = tf.shape(batch)[0]
            seed = tf.stack(
                [
                    tf.cast(seed_base + pass_id, tf.int32),
                    tf.cast(42 + pass_id * 997, tf.int32),
                ],
                axis=0,
            )

            batch = tf.image.stateless_random_flip_left_right(batch, seed=seed)

            zoom = tf.random.stateless_uniform(
                [], minval=0.6, maxval=1.4, seed=seed + tf.constant([11, 17], tf.int32)
            )
            h = tf.shape(batch)[1]
            w = tf.shape(batch)[2]
            new_h = tf.cast(tf.cast(h, tf.float32) / zoom, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) / zoom, tf.int32)
            new_h = tf.clip_by_value(new_h, 1, h)
            new_w = tf.clip_by_value(new_w, 1, w)

            batch = tf.image.stateless_random_crop(
                batch,
                size=tf.stack([b, new_h, new_w, 3]),
                seed=seed + tf.constant([23, 29], tf.int32),
            )
            batch = tf.image.resize(
                batch, [h, w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
            )

            batch = _standardize_tf(batch)

            crop = tf.image.stateless_random_crop(
                batch,
                size=tf.stack([b, crop_length, crop_length, 3]),
                seed=seed + tf.constant([31, 37], tf.int32),
            )
            return crop

        return _tta_batch

    def _tta_predict_from_memory(
        model, x512, datagen, tta_passes=5, batch_size=32, crop_length=448, verbose=1
    ):
        n = x512.shape[0]
        steps = (n + batch_size - 1) // batch_size
        preds = None

        tta_step = _make_tta_step_tf(datagen, crop_length=crop_length)

        for p in range(tta_passes):
            cur_preds = []
            for s in range(steps):
                b0 = s * batch_size
                b1 = min(n, (s + 1) * batch_size)
                batch = x512[b0:b1]

                crops = tta_step(
                    tf.convert_to_tensor(batch, dtype=tf.float32),
                    tf.constant(p, dtype=tf.int32),
                ).numpy()

                logits = model(
                    tf.convert_to_tensor(crops, dtype=tf.float32), training=False
                ).numpy()
                cur_preds.append(logits)

            cur = np.concatenate(cur_preds, axis=0)
            preds = cur if preds is None else (preds + cur)

            if verbose:
                print("TTA pass %d/%d done" % (p + 1, tta_passes))

        return preds / float(tta_passes)




## === cell 4
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["image_id"].tolist()

if TF_AVAILABLE:
    test_dir = os.path.join(INPUT_DIR, "test_images")
    test_datagen = ImageDataGenerator(zoom_range=0.4, horizontal_flip=True)

    _PRED_BATCH_SIZE = 32

    x_test_512 = _load_test_images_to_numpy(
        test_dir, test_ids, target_size=(512, 512), batch_size=64
    )
    print("Loaded test images into memory:", x_test_512.shape, x_test_512.dtype)

    def predict_with_tta(model, tta_passes=5):
        return _tta_predict_from_memory(
            model,
            x_test_512,
            test_datagen,
            tta_passes=tta_passes,
            batch_size=_PRED_BATCH_SIZE,
            crop_length=448,
            verbose=1,
        )

else:
    train_path = os.path.join(INPUT_DIR, "train.csv")
    train_df = pd.read_csv(train_path)
    prior = train_df["label"].value_counts(normalize=True).sort_index()
    num_classes = int(train_df["label"].nunique())
    prior_vec = np.zeros((num_classes,), dtype=np.float32)
    for k, v in prior.items():
        prior_vec[int(k)] = float(v)

    def predict_with_tta(model, tta_passes=5):
        return np.tile(prior_vec[None, :], (len(test_ids), 1))




## === cell 5
pred_v1 = predict_with_tta(model_v1, tta_passes=5)



## === cell 6
temp_v2 = predict_with_tta(model_v2, tta_passes=5)
pred_v2 = temp_v2[:, 1:] if temp_v2.ndim == 2 and temp_v2.shape[1] == 6 else temp_v2



## === cell 7
if pred_v1.shape[1] != pred_v2.shape[1]:
    raise ValueError(
        "Ensemble prediction shape mismatch: %r vs %r" % (pred_v1.shape, pred_v2.shape)
    )

pred_new = 0.5 * pred_v1 + 0.5 * pred_v2
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": predicted_class_indices_new})
sub_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

print("Saved:", sub_path)
print(submission.head())
print("Rows:", len(submission), "Expected:", len(sample_sub))
assert len(submission) == len(sample_sub)
assert list(submission.columns) == ["image_id", "label"]
assert submission["label"].dtype.kind in ("i", "u")
