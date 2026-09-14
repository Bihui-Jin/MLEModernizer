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

0.3601589627320324

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import gc
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as ls
from sklearn.preprocessing import MultiLabelBinarizer

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

np.random.seed(1337)
tf1.set_random_seed(1337)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/freesound-audio-tagging-2019",
    "/kaggle/data/freesound-audio-tagging-2019",
    "../input/freesound-audio-tagging-2019",
]

DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.isdir(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root. Tried: {CANDIDATE_ROOTS}")

train_curated_csv = os.path.join(DATA_ROOT, "train_curated.csv")
train_noisy_csv = os.path.join(DATA_ROOT, "train_noisy.csv")
sample_submission_csv = os.path.join(DATA_ROOT, "sample_submission.csv")

train_curated_path = os.path.join(DATA_ROOT, "train_curated")
train_noisy_path = os.path.join(DATA_ROOT, "train_noisy")
test_path = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("train_curated_csv exists:", os.path.exists(train_curated_csv))
print("train_noisy_csv exists:", os.path.exists(train_noisy_csv))
print("sample_submission_csv exists:", os.path.exists(sample_submission_csv))
print("test_path exists:", os.path.isdir(test_path))



## === cell 2
do_run = True

PREP_DIR = "./preprocessed"
os.makedirs(PREP_DIR, exist_ok=True)



## === cell 3
sample_sub = pd.read_csv(sample_submission_csv)
label_cols = [c for c in sample_sub.columns if c != "fname"]
labels_dict = {label: i for i, label in enumerate(label_cols)}

with open(os.path.join(PREP_DIR, "labels_dict.json"), "w") as fp:
    json.dump(labels_dict, fp)

print("n_labels:", len(labels_dict))
print("First 5 labels:", label_cols[:5])



## === cell 4
train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_curated_df["fname"] = train_curated_df["fname"].apply(
    lambda x: os.path.join(train_curated_path, x)
)
train_noisy_df["fname"] = train_noisy_df["fname"].apply(
    lambda x: os.path.join(train_noisy_path, x)
)

train_curated_df = train_curated_df[
    ~train_curated_df["fname"].str.endswith("1d44b0bd.wav")
].reset_index(drop=True)

split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_indices = set(indices[:split].tolist())
train_indices = set(indices[split:].tolist())

val_curated_df = train_curated_df[train_curated_df.index.isin(val_indices)].reset_index(
    drop=True
)
train_curated_df_split = train_curated_df[
    train_curated_df.index.isin(train_indices)
].reset_index(drop=True)

noisy_labels = train_noisy_df["labels"].astype(str).str.split(",")
new_noisy_labels = []
for labs in noisy_labels:
    keep = [lab for lab in labs if lab in labels_dict]
    new_noisy_labels.append(",".join(keep))
train_noisy_df["labels"] = new_noisy_labels
train_noisy_df = train_noisy_df[train_noisy_df["labels"].str.len() > 0].reset_index(
    drop=True
)

train_df = pd.concat([train_curated_df_split, train_noisy_df], axis=0).reset_index(
    drop=True
)

train_curated_df_split.to_csv(os.path.join(PREP_DIR, "train_curated.csv"), index=False)
train_noisy_df.to_csv(os.path.join(PREP_DIR, "train_noisy.csv"), index=False)
val_curated_df.to_csv(os.path.join(PREP_DIR, "val_curated.csv"), index=False)
train_df.to_csv(os.path.join(PREP_DIR, "train.csv"), index=False)

print("Saved preprocessed CSVs:")
print("train.csv rows:", len(train_df))
print("val_curated.csv rows:", len(val_curated_df))



## === cell 5
del train_curated_df, train_noisy_df, train_df
gc.collect()




## === cell 6
def random_rolls(audio_1d):
    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    roll_amount = tf.random.uniform((1,), 1200, 2000)[0]
    new_audio = tf.cond(
        cond,
        lambda: tf.roll(audio_1d, shift=tf.cast(roll_amount, tf.int32), axis=0),
        lambda: audio_1d,
    )
    return new_audio


def random_speedx(audio_1d):
    def get_audio():
        factor = tf.random.uniform((1,), 0.7, 1.7)[0]
        indices = tf.round(
            tf.range(0.0, tf.cast(tf.shape(audio_1d)[0], tf.float32), factor)
        )
        indices = tf.boolean_mask(
            indices, indices < tf.cast(tf.shape(audio_1d)[0], tf.float32)
        )
        sound = tf.gather(audio_1d, tf.cast(indices, tf.int32))
        return sound

    random_num = tf.random.uniform((1,), 0, 10)[0]
    cond = random_num >= 5
    return tf.cond(cond, get_audio, lambda: audio_1d)


def wav_to_spectogram(wav_filename, random_perturbs=False):
    wav_bytes = tf.io.read_file(wav_filename)
    audio_tensor, sample_rate = tf.audio.decode_wav(
        wav_bytes, desired_channels=1
    )  # [samples,1]
    sound = tf.squeeze(audio_tensor, axis=-1)  # [samples]

    if random_perturbs:
        sound = random_speedx(random_rolls(sound))

    stft = tf.signal.stft(
        sound,
        frame_length=1024,
        frame_step=64,
        fft_length=1024,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    spectogram = tf.abs(stft)  # [frames, freq_bins]

    minimum = tf.minimum(spectogram, 255.0)
    expand_dims = tf.expand_dims(minimum, -1)  # [frames, freq_bins, 1]
    expand_dims = tf.expand_dims(expand_dims, 0)  # [1, frames, freq_bins, 1]
    resize = tf.image.resize(
        expand_dims, [300, 300], method=tf.image.ResizeMethod.BILINEAR
    )
    squeeze = tf.squeeze(resize, axis=0)  # [300,300,1]
    return squeeze


def csv_pipe(csv_path, labels_dict):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=1337).reset_index(drop=True)

    files = df["fname"].values.astype(str)
    labels = df["labels"].astype(str).str.split(",")

    class_order = [k for k, _ in sorted(labels_dict.items(), key=lambda kv: kv[1])]
    binarizer = MultiLabelBinarizer(classes=class_order)
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)
    return files, labels_1hot


def wav_data_generators(
    csv_path, labels_dict, batch_size=32, shuffle=False, repeat=True
):
    files, labels = csv_pipe(csv_path, labels_dict)

    def _map_fn(file, label):
        return wav_to_spectogram(file, random_perturbs=shuffle), label

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(800, seed=1337)

    dataset = dataset.batch(batch_size).prefetch(10)

    if repeat:
        dataset = dataset.repeat()

    return dataset




## === cell 7
def conv_relu_bn(filters, kernels, strides, padding="valid", in_shape=None):
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
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding="same", in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding="same")
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding="same")
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding="same")
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
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
        x = self.gpool(x)
        return self.classifier(x)




## === cell 8
class Trainor:
    def __init__(
        self,
        train_csv,
        val_csv,
        labels_dict,
        lr=1e-3,
        batch_size=8,
        shuffle=True,
        model_dir="ds2",
    ):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
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

        optimizer = tf1.train.RMSPropOptimizer(lr)
        self.global_step = tf1.train.get_or_create_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess = tf1.Session()
        self._define_summaries()
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver(max_to_keep=5)
        self._may_be_load_model()

    def _define_summaries(self):
        tf1.summary.scalar("loss", self.loss)
        tf1.summary.scalar("accuracy", self.accuracy)
        flip = tf.image.flip_left_right(self.features)
        transpose = tf.image.transpose(flip, perm=[0, 2, 1, 3])
        tf1.summary.image("spectogram", transpose, max_outputs=2)
        self.merged = tf1.summary.merge_all()
        self.summary_writer = tf1.summary.FileWriter(self.model_dir, self.sess.graph)

    def _define_dataset_iterators(
        self, train_csv, val_csv, labels_dict, batch_size, shuffle
    ):
        train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size, shuffle=True, repeat=True
        )
        val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size, shuffle=False, repeat=True
        )

        out_types = tf1.data.get_output_types(train_dataset)
        out_shapes = tf1.data.get_output_shapes(train_dataset)

        it = tf1.data.Iterator.from_structure(out_types, out_shapes)
        self.features, self.labels = it.get_next()
        self.train_init_op = it.make_initializer(train_dataset)
        self.val_init_op = it.make_initializer(val_dataset)

    def _may_be_load_model(self):
        ckpt = tf1.train.latest_checkpoint(self.model_dir)
        if ckpt is not None:
            self.saver.restore(self.sess, ckpt)
            print("Restored checkpoint:", ckpt)

    def _save_model(self):
        return self.saver.save(
            self.sess,
            os.path.join(self.model_dir, "model.ckpt"),
            global_step=self.global_step,
        )

    def train(
        self,
        steps=500,
        ckpt_every_n_steps=100,
        log_every_n_steps=500,
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

        path = self._save_model()
        print("Final Model saved at:", path)

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 9
with open(os.path.join(PREP_DIR, "labels_dict.json")) as fp:
    labels_dict = json.load(fp)

ordered_labels = [c for c in sample_sub.columns if c != "fname"]
assert len(labels_dict) == len(ordered_labels) == 80
for i, lab in enumerate(ordered_labels):
    assert labels_dict[lab] == i

print("labels_dict OK")



## === cell 10
train_steps = 10000 if do_run else 700

tf1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1
tf1.set_random_seed(1337)

trainor = Trainor(
    os.path.join(PREP_DIR, "train.csv"),
    os.path.join(PREP_DIR, "val_curated.csv"),
    labels_dict,
    model_dir="ds2",
)

trainor.train(steps=train_steps)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1678799646.py in <cell line: 0>()
      7 tf1.set_random_seed(1337)
      8 
----> 9 trainor = Trainor(
     10     os.path.join(PREP_DIR, "train.csv"),
     11     os.path.join(PREP_DIR, "val_curated.csv"),

/tmp/ipykernel_11/4055670259.py in __init__(self, train_csv, val_csv, labels_dict, lr, batch_size, shuffle, model_dir)
     33 
     34         self.sess = tf1.Session()
---> 35         self._define_summaries()
     36         self.sess.run(tf1.global_variables_initializer())
     37         self.saver = tf1.train.Saver(max_to_keep=5)

/tmp/ipykernel_11/4055670259.py in _define_summaries(self)
     42         tf1.summary.scalar("accuracy", self.accuracy)
     43         flip = tf.image.flip_left_right(self.features)
---> 44         transpose = tf.image.transpose(flip, perm=[0, 2, 1, 3])
     45         tf1.summary.image("spectogram", transpose, max_outputs=2)
     46         self.merged = tf1.summary.merge_all()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/dispatch.py in op_dispatch_handler(*args, **kwargs)
   1252         if iterable_params is not None:
   1253           args, kwargs = replace_iterable_params(args, kwargs, iterable_params)
-> 1254         result = api_dispatcher.Dispatch(args, kwargs)
   1255         if result is not NotImplemented:
   1256           return result

TypeError: Got an unexpected keyword argument 'perm'

## === cell 11
class Predictor:
    def __init__(self, files, batch_size=16, model_dir="ds2"):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1), n_classes=80)
        self._define_dataset_iterator(files, batch_size)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)

        self.sess = tf1.Session()
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()
        self._load_model()

    def _define_dataset_iterator(self, files, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(files.astype(str))
        dataset = dataset.map(
            wav_to_spectogram, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(10)
        it = tf1.data.make_one_shot_iterator(dataset)
        self.features = it.get_next()

    def _load_model(self):
        ckpt = tf1.train.latest_checkpoint(self.model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found in {self.model_dir}. Ensure training ran and saved checkpoints."
            )
        self.saver.restore(self.sess, ckpt)
        print("Loaded checkpoint:", ckpt)

    def pred_generator(self):
        try:
            while True:
                yield self.sess.run(self.probas)
        except tf.errors.OutOfRangeError:
            print("Prediction over (dataset exhausted).")

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass




## === cell 12
test_fnames = sample_sub["fname"].values.astype(str)
test_files = np.array([os.path.join(test_path, f) for f in test_fnames])

tf1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1
tf1.set_random_seed(1337)

predictor = Predictor(test_files, batch_size=16, model_dir="ds2")
pred_gen = predictor.pred_generator()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/696335343.py in <cell line: 0>()
      7 tf1.set_random_seed(1337)
      8 
----> 9 predictor = Predictor(test_files, batch_size=16, model_dir="ds2")
     10 pred_gen = predictor.pred_generator()
     11 

/tmp/ipykernel_11/432544901.py in __init__(self, files, batch_size, model_dir)
     11         self.sess.run(tf1.global_variables_initializer())
     12         self.saver = tf1.train.Saver()
---> 13         self._load_model()
     14 
     15     def _define_dataset_iterator(self, files, batch_size):

/tmp/ipykernel_11/432544901.py in _load_model(self)
     25         ckpt = tf1.train.latest_checkpoint(self.model_dir)
     26         if ckpt is None:
---> 27             raise FileNotFoundError(
     28                 f"No checkpoint found in {self.model_dir}. Ensure training ran and saved checkpoints."
     29             )

FileNotFoundError: No checkpoint found in ds2. Ensure training ran and saved checkpoints.

## === cell 13
n_test = len(test_fnames)
preds = np.zeros((n_test, 80), dtype=np.float32)

offset = 0
for probs in pred_gen:
    bs = probs.shape[0]
    preds[offset : offset + bs, :] = probs
    offset += bs

assert offset == n_test, f"Predicted {offset} rows, expected {n_test}"

sub_df = pd.DataFrame(preds, columns=ordered_labels)
sub_df.insert(0, "fname", test_fnames)

assert list(sub_df.columns) == list(sample_sub.columns)
sub_df = sub_df.fillna(0.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2407062148.py in <cell line: 0>()
      3 
      4 offset = 0
----> 5 for probs in pred_gen:
      6     bs = probs.shape[0]
      7     preds[offset : offset + bs, :] = probs

NameError: name 'pred_gen' is not defined
