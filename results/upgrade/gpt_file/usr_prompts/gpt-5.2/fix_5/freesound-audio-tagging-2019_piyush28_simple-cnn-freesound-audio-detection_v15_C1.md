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
Develop a model to tag audio data automatically using a diverse vocabulary of 80 categories.

## Metric
The task consists of predicting the audio labels (tags) for every test clip. Some test clips bear one label while others bear several labels. The predictions are to be done at the clip level, i.e., no start/end timestamps for the sound events are required.

The primary metric is label-weighted label-ranking average precision. 

The  "label-weighted" part means that the overall score is the average over all the *labels* in the test set, where each label receives equal weight (by contrast, plain *lrap* gives each *test item* equal weight).

## Submission Format
For each `fname` in the test set, you must predict the probability of each label. The file should contain a header and have the following format:

```
fname,Accelerating_and_revving_and_vroom,...Zipper_(clothing)
000ccb97.wav,0.1,....,0.3
0012633b.wav,0.0,...,0.8
```

## Dataset
The following 5 audio files in the curated train set have a wrong label, due to a bug in the file renaming process:\
`f76181c4.wav, 77b925c2.wav, 6a1f682a.wav, c7db12aa.wav, 7752cc8a.wav`

The audio file `1d44b0bd.wav` in the curated train set was found to be corrupted (contains no signal) due to an error in format conversion.

- **train_curated.csv** - ground truth labels for the curated subset of the training audio files (see Data Fields below)
- **train_noisy.csv** - ground truth labels for the noisy subset of the training audio files (see Data Fields below)
- **sample_submission.csv** - a sample submission file in the correct format, including the correct sorting of the sound categories; it contains the list of audio files found in the test.zip folder (corresponding to the public leaderboard)
- **train_curated.zip** - a folder containing the audio (.wav) training files of the curated subset
- **train_noisy.zip** - a folder containing the audio (.wav) training files of the noisy subset
- **test.zip** - a folder containing the audio (.wav) test files for the public leaderboard

### Columns
Each row of the train_curated.csv and train_noisy.csv files contains the following information:

- **fname**: the audio file name, eg, `0006ae4e.wav`
- **labels**: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        input/
            description.md (276 lines)
            sample_submission.csv (3362 lines)
            sample_submission.csv.zip (20.7 kB)
            test.zip (2.2 GB)
            train_curated.csv (4971 lines)
            train_curated.csv.zip (39.3 kB)
            train_curated.zip (2.4 GB)
            train_noisy.csv (19816 lines)
            train_noisy.csv.zip (154.2 kB)
            train_noisy.zip (21.5 GB)
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
            test/
                4260ebea.wav (1.0 MB)
                426eb1e0.wav (654.5 kB)
                ... and 3359 other files
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
            train_curated/
                0006ae4e.wav (621.0 kB)
                0019ef41.wav (181.3 kB)
                ... and 4968 other files
            train_noisy/
                00097e21.wav (1.3 MB)
                000b6cfb.wav (1.3 MB)
                ... and 19813 other files
        working/
            freesound-audio-tagging-2019/
                description.md (276 lines)
                sample_submission.csv (3362 lines)
                ... and 8 other files
                freesound-audio-tagging-2019/
                test/
                    4260ebea.wav (1.0 MB)
                    426eb1e0.wav (654.5 kB)
                    ... and 3359 other files
                    test/
                train_curated/
                    0006ae4e.wav (621.0 kB)
                    0019ef41.wav (181.3 kB)
                    ... and 4968 other files
                train_noisy/
                    00097e21.wav (1.3 MB)
                    000b6cfb.wav (1.3 MB)
                    ... and 19813 other files
```

-> data/freesound-audio-tagging-2019/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/freesound-audio-tagging-2019/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/freesound-audio-tagging-2019/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> data/sample_submission.csv has 3361 rows and 81 columns.
The columns are: fname, Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Applause, Bark, Bass_drum, Bass_guitar, Bathtub_(filling_or_washing), Bicycle_bell, Burping_and_eructation, Bus, Buzz, Car_passing_by, Cheering... and 66 more columns

-> data/train_curated.csv has 4970 rows and 2 columns.
The columns are: fname, labels

-> data/train_noisy.csv has 19815 rows and 2 columns.
The columns are: fname, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.4151301852907714

# 6. Current score

0.07307

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06442) has done: 'I fix the TensorFlow import/runtime crash by removing the eager-disable path and forcing TF1-style graph mode in a way that’s compatible with the Kaggle TF/protobuf stack. Then I fix test file enumeration so directories like `test/test` are excluded (that caused the `ReadFile ... Is a directory` error and the submission length mismatch). Finally, I ensure the submission rows exactly match `sample_submission.csv` ordering/length by using its `fname` list to drive prediction, which is score-neutral but guarantees a valid submission file is produced.'
- What this solution (achieved 0.06533) has done: 'We fix the TensorFlow/protobuf crash by avoiding the TF1 `disable_v2_behavior()` path (which is triggering the `MessageFactory.GetPrototype` issue in this environment) and instead run in TF2 with `tf.compat.v1` graph execution using `disable_eager_execution()` only. We also ensure the graph is reset at the right times and that sessions use the compat-v1 API consistently, without changing your model, preprocessing, loss, or training loop semantics. Finally, we keep submission ordering driven strictly by `sample_submission.csv` so the output is always valid and aligned, while keeping the approach score-neutral aside from enabling training/inference to run correctly.'
- What this solution (achieved 0.07307) has done: 'We fix the immediate TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by switching the script to use `tf.compat.v1` graph mode only after importing TensorFlow through a safer Keras path and by avoiding the problematic mixed TF/Keras imports. Then we ensure the graph/session lifecycle is consistent (reset graph before building Trainor/Predictor, and avoid stale graphs) so training/inference run end-to-end. Finally, we keep submission ordering strictly driven by `sample_submission.csv` and always write `submission.csv` with the exact required columns, which is score-neutral but guarantees a valid file. These changes are minimal and do not alter your model architecture, preprocessing, loss, or training loop semantics.'

# 9. Code solution

## === cell 0
import os
import json
import gc
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer

tf1 = tf.compat.v1
tf1.disable_eager_execution()

np.random.seed(42)
tf1.set_random_seed(42)

INPUT_ROOT = "/kaggle/input/freesound-audio-tagging-2019"
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

print("TF version:", tf.__version__)
print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
do_run = False



## === cell 2
train_curated_csv = os.path.join(INPUT_ROOT, "train_curated.csv")
train_noisy_csv = os.path.join(INPUT_ROOT, "train_noisy.csv")
sample_sub_csv = os.path.join(INPUT_ROOT, "sample_submission.csv")

train_curated_path = os.path.join(INPUT_ROOT, "train_curated")
train_noisy_path = os.path.join(INPUT_ROOT, "train_noisy")
test_path = os.path.join(INPUT_ROOT, "test")

assert os.path.exists(train_curated_csv), train_curated_csv
assert os.path.exists(train_noisy_csv), train_noisy_csv
assert os.path.exists(sample_sub_csv), sample_sub_csv
assert os.path.isdir(train_curated_path), train_curated_path
assert os.path.isdir(train_noisy_path), train_noisy_path
assert os.path.isdir(test_path), test_path

print("n_test_entries_in_dir:", len(os.listdir(test_path)))



## === cell 3
sample_sub = pd.read_csv(sample_sub_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}

train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_curated_df["fname"] = train_curated_df["fname"].apply(
    lambda x: os.path.join(train_curated_path, x)
)
train_noisy_df["fname"] = train_noisy_df["fname"].apply(
    lambda x: os.path.join(train_noisy_path, x)
)

noisy_labels = train_noisy_df["labels"].str.split(",")
new_noisy_labels = []
for labs in noisy_labels:
    new_noisy_labels.append(",".join([lab for lab in labs if lab in labels_dict]))
train_noisy_df["labels"] = new_noisy_labels

split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_indices = indices[:split]
train_indices = indices[split:]
val_curated_df = train_curated_df.iloc[val_indices].reset_index(drop=True)
train_curated_df = train_curated_df.iloc[train_indices].reset_index(drop=True)

train_df = pd.concat([train_curated_df, train_noisy_df], axis=0).reset_index(drop=True)

os.makedirs("./preprocessed", exist_ok=True)
train_curated_df.to_csv("./preprocessed/train_curated.csv", index=False)
train_noisy_df.to_csv("./preprocessed/train_noisy.csv", index=False)
val_curated_df.to_csv("./preprocessed/val_curated.csv", index=False)
train_df.to_csv("./preprocessed/train.csv", index=False)
with open("./preprocessed/labels_dict.json", "w") as fp:
    json.dump(labels_dict, fp)

print("Prepared:", train_df.shape, "labels:", len(labels_dict))



## === cell 4
for _name in [
    "train_curated_df",
    "train_noisy_df",
    "train_df",
    "train_indices",
    "val_indices",
    "indices",
    "noisy_labels",
    "new_noisy_labels",
]:
    if _name in globals():
        del globals()[_name]
gc.collect()




## === cell 5
def wav_to_spectogram(wav_filename):
    wav_bytes = tf.io.read_file(wav_filename)
    audio_decoded, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    audio_decoded = tf.squeeze(audio_decoded, axis=-1)  # [samples]

    stft = tf.signal.stft(
        audio_decoded,
        frame_length=1024,
        frame_step=64,
        fft_length=1024,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spec = tf.abs(stft)  # [time, freq]

    spec = tf.minimum(spec, 255.0)

    spec = tf.expand_dims(spec, axis=-1)  # [time, freq, 1]
    spec = tf.image.resize(spec, [300, 300], method=tf.image.ResizeMethod.BILINEAR)
    return spec  # [300, 300, 1]


def csv_pipe(csv, labels_dict):
    df = pd.read_csv(csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    files = df["fname"].values
    labels = df["labels"].fillna("").str.split(",")

    classes = list(labels_dict.keys())
    binarizer = MultiLabelBinarizer(classes=classes)
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)
    return files, labels_1hot


def wav_data_generators(csv, labels_dict, batch_size=32, shuffle=False, repeat=True):
    files, labels = csv_pipe(csv, labels_dict)

    def _map_fn(file, label):
        return wav_to_spectogram(file), label

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if repeat:
        dataset = dataset.repeat()

    return dataset




## === cell 6
def conv_relu_bn(filters, kernels, strides, padding="valid", in_shape=None):
    ls = tf.keras.layers
    if in_shape is not None:
        conv = ls.Conv2D(
            filters, kernels, strides, padding=padding, input_shape=in_shape
        )
    else:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding)
    return tf.keras.models.Sequential([conv, ls.ReLU(), ls.BatchNormalization()])


class Model(tf.keras.Model):
    def __init__(self, in_shape, n_classes=80):
        super(Model, self).__init__()
        ls = tf.keras.layers
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding="same", in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding="same")
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding="same")
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding="same")
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
        self.conv_relu_bn9 = conv_relu_bn(1024, 3, 2)
        self.gpool = ls.GlobalAveragePooling2D()
        self.classifier = ls.Dense(n_classes)

    def call(self, inputs):
        x = self.conv_relu_bn1(inputs)
        x = self.conv_relu_bn2(x)
        x = self.conv_relu_bn3(x)
        x = self.conv_relu_bn4(x)
        x = self.conv_relu_bn5(x)
        x = self.conv_relu_bn6(x)
        x = self.conv_relu_bn7(x)
        x = self.conv_relu_bn8(x)
        x = self.conv_relu_bn9(x)
        x = self.gpool(x)
        return self.classifier(x)




## === cell 7
class Trainor:
    def __init__(
        self,
        train_csv,
        val_csv,
        labels_dict,
        lr=1e-3,
        batch_size=16,
        shuffle=True,
        model_dir="ds2",
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = Model(in_shape=(300, 300, 1), n_classes=len(labels_dict))
        self._define_dataset_iterators(
            train_csv, val_csv, labels_dict, batch_size, shuffle
        )

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        preds = self.probas > 0.5

        self.loss = tf1.losses.sigmoid_cross_entropy(self.labels, logits)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(preds, tf.int32), self.labels), tf.float32)
        )

        optimizer = tf1.train.AdamOptimizer(lr)
        self.global_step = tf1.train.get_or_create_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = tf1.Session(config=cfg)

        self._define_summaries()
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()
        self._may_be_load_model()

    def _define_summaries(self):
        tf1.summary.scalar("loss", self.loss)
        tf1.summary.scalar("accuracy", self.accuracy)
        flip = tf.image.flip_left_right(self.features)
        transpose = tf.image.transpose(flip, perm=[0, 2, 1, 3])
        tf1.summary.image("spectogram", transpose)
        self.merged = tf1.summary.merge_all()
        self.summary_writer = tf1.summary.FileWriter(self.model_dir, self.sess.graph)

    def _define_dataset_iterators(
        self, train_csv, val_csv, labels_dict, batch_size, shuffle
    ):
        train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size, shuffle, repeat=True
        )
        val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size, shuffle=False, repeat=True
        )

        it = tf1.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes
        )
        self.features, self.labels = it.get_next()
        self.train_init_op = it.make_initializer(train_dataset)
        self.val_init_op = it.make_initializer(val_dataset)

    def _may_be_load_model(self):
        if os.path.isdir(self.model_dir):
            files = os.listdir(self.model_dir)
            steps = []
            for f in files:
                if "model.ckpt-" in f and f.endswith(".index"):
                    try:
                        steps.append(int(f.split("model.ckpt-")[-1].split(".")[0]))
                    except Exception:
                        pass
            if len(steps) > 0:
                self._load_model(os.path.join(self.model_dir, "model.ckpt"), max(steps))

    def _save_model(self):
        return self.saver.save(
            self.sess,
            os.path.join(self.model_dir, "model.ckpt"),
            global_step=self.global_step,
        )

    def _load_model(self, ckpt_path, step):
        self.saver.restore(self.sess, ckpt_path + "-{}".format(step))

    def train(
        self,
        steps=500,
        ckpt_every_n_steps=100,
        log_every_n_steps=1500,
        log_val_n_steps=5,
    ):
        self.sess.run(self.train_init_op)
        for step in range(1, steps + 1):
            summary, _ = self.sess.run([self.merged, self.train_op])
            self.summary_writer.add_summary(summary, step)

            if step % log_every_n_steps == 0:
                self.sess.run(self.val_init_op)
                val_losses, val_accs = [], []
                for _ in range(log_val_n_steps):
                    l, a = self.sess.run([self.loss, self.accuracy])
                    val_losses.append(l)
                    val_accs.append(a)
                print(
                    "STEP: {}, Loss: {:.4f}, Acc: {:.4f}".format(
                        step, np.mean(val_losses), np.mean(val_accs)
                    )
                )
                self.sess.run(self.train_init_op)

            if step % ckpt_every_n_steps == 0:
                path = self._save_model()
                print("STEP: {}, Model saved at: {}".format(step, path))

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 8
with open("./preprocessed/labels_dict.json") as fp:
    labels_dict = json.load(fp)

sample_sub = pd.read_csv(sample_sub_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}

print("Loaded labels:", len(labels_dict))



## === cell 9
if do_run:
    tf1.reset_default_graph()
    trainor = Trainor(
        "./preprocessed/train.csv", "./preprocessed/val_curated.csv", labels_dict
    )



## === cell 10
if do_run:
    trainor.train(10000)




## === cell 11
class Predictor:
    def __init__(self, files, batch_size=16, model_dir="ds2"):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1), n_classes=len(labels_dict))
        self._define_dataset_iterator(files, batch_size)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)

        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = tf1.Session(config=cfg)
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()

        self._maybe_load_model()

    def _define_dataset_iterator(self, files, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(
            wav_to_spectogram, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.iterator = tf1.data.make_one_shot_iterator(dataset)
        self.features = self.iterator.get_next()

    def _maybe_load_model(self):
        if not os.path.isdir(self.model_dir):
            print("No model_dir found, using random weights:", self.model_dir)
            return
        files = os.listdir(self.model_dir)
        steps = []
        for f in files:
            if "model.ckpt-" in f and f.endswith(".index"):
                try:
                    steps.append(int(f.split("model.ckpt-")[-1].split(".")[0]))
                except Exception:
                    pass
        if len(steps) == 0:
            print(
                "No checkpoint found in model_dir, using random weights:",
                self.model_dir,
            )
            return
        ckpt_path = os.path.join(self.model_dir, "model.ckpt")
        step = max(steps)
        self.saver.restore(self.sess, ckpt_path + "-{}".format(step))
        print("Loaded checkpoint:", ckpt_path + "-{}".format(step))

    def pred_generator(self):
        while True:
            try:
                yield self.sess.run(self.probas)
            except tf.errors.OutOfRangeError:
                break
        print("Prediction over")

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 12
tf1.reset_default_graph()

sample_sub = pd.read_csv(sample_sub_csv)
orig_files = sample_sub["fname"].tolist()
files = [os.path.join(test_path, f) for f in orig_files]

bad = [p for p in files if (not os.path.isfile(p))]
if bad:
    raise FileNotFoundError(
        "Some test paths are not files (first 5 shown):\n{}".format("\n".join(bad[:5]))
    )

predictor = Predictor(files, batch_size=16, model_dir="ds2")



## === cell 13
pred_gen = predictor.pred_generator()



## === cell 14
n_test = len(orig_files)
n_classes = len(labels_dict)
all_preds = np.zeros((n_test, n_classes), dtype=np.float32)

offset = 0
for probs in pred_gen:
    bs = probs.shape[0]
    all_preds[offset : offset + bs, :] = probs
    offset += bs

assert offset == n_test, (offset, n_test)



## === cell 15
sub_df = pd.DataFrame(all_preds, columns=label_cols)
sub_df.insert(0, "fname", orig_files)
sub_df = sub_df[["fname"] + label_cols]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.head())
