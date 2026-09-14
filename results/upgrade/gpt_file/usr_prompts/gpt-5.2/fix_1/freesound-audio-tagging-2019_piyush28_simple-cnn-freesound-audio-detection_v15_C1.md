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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
import tensorflow.keras.layers as ls
from tensorflow.contrib.framework.python.ops import audio_ops as audio
from sklearn.preprocessing import MultiLabelBinarizer

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.listdir('../input')

## === cell 2
do_run=False

## === cell 3
if not do_run:
    print(os.listdir('../input/cnnfat2019'))
    train_curated_csv = '../input/freesound-audio-tagging-2019/train_curated.csv'
    train_noisy_csv = '../input/freesound-audio-tagging-2019/train_noisy.csv'
    test_csv = '../input/freesound-audio-tagging-2019/test.csv'
    train_curated_path = '../input/freesound-audio-tagging-2019/train_curated/'
    train_noisy_path = '../input/freesound-audio-tagging-2019/train_noisy/'
else:
    train_curated_csv = '../input/train_curated.csv'
    train_noisy_csv = '../input/train_noisy.csv'
    test_csv = '../input/test.csv'
    train_curated_path = '../input/train_curated/'
    train_noisy_path = '../input/train_noisy/'

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3216338074.py in <cell line: 0>()
      1 if not do_run:
----> 2     print(os.listdir('../input/cnnfat2019'))
      3     train_curated_csv = '../input/freesound-audio-tagging-2019/train_curated.csv'
      4     train_noisy_csv = '../input/freesound-audio-tagging-2019/train_noisy.csv'
      5     test_csv = '../input/freesound-audio-tagging-2019/test.csv'

FileNotFoundError: [Errno 2] No such file or directory: '../input/cnnfat2019'

## === cell 4
train_curated_df = pd.read_csv(train_curated_csv)
train_noisy_df = pd.read_csv(train_noisy_csv)

train_curated_df['fname'] = train_curated_path + train_curated_df['fname']
train_noisy_df['fname'] = train_noisy_path + train_noisy_df['fname']

labels = np.concatenate(train_curated_df['labels'].str.split(','))
unique_labels = np.unique(labels)
labels_dict = {label:i for i, label in enumerate(unique_labels)}

split = int(0.2 * train_curated_df.shape[0])
indices = np.random.permutation(train_curated_df.shape[0])
val_indices = indices[:split]
train_indices = indices[split:]
val_curated_df = train_curated_df[train_curated_df.index.isin(val_indices)]
train_curated_df = train_curated_df[train_curated_df.index.isin(train_indices)]

train_df = pd.concat([train_curated_df, train_noisy_df])

noisy_labels = train_noisy_df['labels'].str.split(',')
new_noisy_labels = []
for labels in noisy_labels:
    new_noisy_labels.append(','.join([label for label in labels if label in labels_dict]))
train_noisy_df['labels'] = new_noisy_labels

if not os.path.isdir('./preprocessed'):
    os.makedirs('./preprocessed')
train_curated_df.to_csv('./preprocessed/train_curated.csv', index=False)
train_noisy_df.to_csv('./preprocessed/train_noisy.csv', index=False)
val_curated_df.to_csv('./preprocessed/val_curated.csv', index=False)
train_df.to_csv('./preprocessed/train.csv', index=False)

with open('./preprocessed/labels_dict.json', 'w+') as fp:
    json.dump(labels_dict, fp)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/602726434.py in <cell line: 0>()
      1 # Read train files
----> 2 train_curated_df = pd.read_csv(train_curated_csv)
      3 train_noisy_df = pd.read_csv(train_noisy_csv)
      4 
      5 # Append path to train files

NameError: name 'train_curated_csv' is not defined

## === cell 5
del train_curated_df
del train_df
del train_noisy_df
del train_indices
del labels_dict
import gc; gc.collect()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1744170857.py in <cell line: 0>()
----> 1 del train_curated_df
      2 del train_df
      3 del train_noisy_df
      4 del train_indices
      5 del labels_dict

NameError: name 'train_curated_df' is not defined

## === cell 6
def wav_to_spectogram(wav_filename):
    wav_bytes = tf.read_file(wav_filename)
    wav_decoder = audio.decode_wav(wav_bytes, 1)
    spectogram = audio.audio_spectrogram(
        wav_decoder.audio, window_size=1024, stride=64)
    
    minimum =  tf.minimum(spectogram, 255.)
    expand_dims = tf.expand_dims(minimum, -1)
    resize = tf.image.resize_bilinear(expand_dims, [300, 300])
    squeeze = tf.squeeze(resize, 0)

    return squeeze

def csv_pipe(csv, labels_dict):
    df = pd.read_csv(csv)
    df = df.sample(frac=1).reset_index(drop=True)
    
    files = df['fname']
    labels = df['labels'].str.split(',')

    binarizer = MultiLabelBinarizer(list(labels_dict.keys()))
    labels_1hot = binarizer.fit_transform(labels).astype(np.int32)
    return files, labels_1hot

def wav_data_generators(csv, labels_dict, batch_size=32, shuffle=False, repeat=True):
    files, labels = csv_pipe(csv, labels_dict)

    wav_to_spectogram_applier = lambda file, label: (wav_to_spectogram(file), label)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(wav_to_spectogram_applier)

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(10)

    if repeat:
        dataset = dataset.repeat()

    return dataset

## === cell 7
def conv_relu_bn(filters, kernels, strides, padding='valid', in_shape=None):
    if in_shape is not None:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding, input_shape=in_shape)
    else:
        conv = ls.Conv2D(filters, kernels, strides, padding=padding)
    return tf.keras.models.Sequential([
        conv,
        ls.ReLU(),
        ls.BatchNormalization()
    ])
    

class Model(tf.keras.Model):
    def __init__(self, in_shape, n_classes=80):
        super(Model, self).__init__()
        self.conv_relu_bn1 = conv_relu_bn(64, 3, 1, padding='same', in_shape=in_shape)
        self.conv_relu_bn2 = conv_relu_bn(64, 3, 2)
        self.conv_relu_bn3 = conv_relu_bn(128, 3, 1, padding='same')
        self.conv_relu_bn4 = conv_relu_bn(128, 3, 2)
        self.conv_relu_bn5 = conv_relu_bn(256, 3, 1, padding='same')
        self.conv_relu_bn6 = conv_relu_bn(256, 3, 2)
        self.conv_relu_bn7 = conv_relu_bn(512, 3, 1, padding='same')
        self.conv_relu_bn8 = conv_relu_bn(512, 3, 2)
        self.conv_relu_bn9 = conv_relu_bn(1024, 3, 2)
        self.gpool = ls.GlobalAveragePooling2D()
        self.classifier = ls.Dense(n_classes)

    def call(self, inputs):
        conv_relu_bn1 = self.conv_relu_bn1(inputs)
        conv_relu_bn2 = self.conv_relu_bn2(conv_relu_bn1)
        conv_relu_bn3 = self.conv_relu_bn3(conv_relu_bn2)
        conv_relu_bn4 = self.conv_relu_bn4(conv_relu_bn3)
        conv_relu_bn5 = self.conv_relu_bn5(conv_relu_bn4)
        conv_relu_bn6 = self.conv_relu_bn6(conv_relu_bn5)
        conv_relu_bn7 = self.conv_relu_bn7(conv_relu_bn6)
        conv_relu_bn8 = self.conv_relu_bn8(conv_relu_bn7)
        conv_relu_bn9 = self.conv_relu_bn9(conv_relu_bn8)
        pooled = self.gpool(conv_relu_bn9)
        return self.classifier(pooled)


## === cell 8
class Trainor:
    def __init__(self,
                 train_csv,
                 val_csv,
                 labels_dict,
                 lr=1e-3,
                 batch_size=16,
                 shuffle=True,
                 model_dir='ds2'):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model = Model(in_shape=(300, 300, 1))
        self._define_dataset_iterators(
            train_csv, val_csv, labels_dict, batch_size, shuffle)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        preds = self.probas > 0.5

        self.loss = tf.losses.sigmoid_cross_entropy(self.labels, logits)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(preds, tf.int32), self.labels), tf.float32))
        optimizer = tf.train.AdamOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        self.sess = tf.Session()
        self._define_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._may_be_load_model()

    def _define_summaries(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        flip = tf.image.flip_left_right(self.features)
        transpose = tf.image.transpose_image(flip)
        tf.summary.image('spectogram', transpose)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

    def _define_dataset_iterators(self, train_csv, val_csv, labels_dict, batch_size, shuffle):
        train_dataset = wav_data_generators(
            train_csv, labels_dict, batch_size, shuffle)
        val_dataset = wav_data_generators(
            val_csv, labels_dict, batch_size, shuffle)
        iter = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        self.features, self.labels = iter.get_next()
        self.train_init_op = iter.make_initializer(train_dataset)
        self.val_init_op = iter.make_initializer(val_dataset)

    def _may_be_load_model(self):
        if os.path.isdir(self.model_dir):
            files = os.listdir(self.model_dir)
            steps = [int(file.split('-')[-1].split('.')[0]) for file in files if 'model' in file]
            if len(steps) > 0:
                self._load_model(os.path.join(self.model_dir, 'model.ckpt'), max(steps))

    def _save_model(self):
        return self.saver.save(
            self.sess, os.path.join(self.model_dir, 'model.ckpt'),
            global_step=self.global_step)

    def _load_model(self, ckpt_path, step):
        self.saver.restore(self.sess, ckpt_path + '-{}'.format(step))

    def train(self, steps=500, ckpt_every_n_steps=100, log_every_n_steps=1500, log_val_n_steps=5):
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)
        self.sess.run(self.train_init_op)
        for step in range(1, steps+1):
            
            summary, _ = self.sess.run([self.merged, self.train_op])
            self.summary_writer.add_summary(summary, step)

            if step % log_every_n_steps == 0:
                self.sess.run(self.val_init_op)
                val_losses, val_accs = [], []
                for _ in range(log_val_n_steps):
                    l, a = self.sess.run([self.loss, self.accuracy])
                    val_losses.append(l)
                    val_accs.append(a)
                print('STEP: {}, Loss: {:.4f}, Acc: {:.4f}'.format(
                    step, np.mean(val_losses), np.mean(val_accs)))
                self.sess.run(self.train_init_op)

            if step % ckpt_every_n_steps == 0:
                path = self._save_model()
                print('STEP: {}, Model saved at: {}'.format(step, path))

    def __del__(self):
        self.sess.close()

## === cell 9
with open('./preprocessed/labels_dict.json') as fp:
    labels_dict = json.load(fp)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3059799244.py in <cell line: 0>()
----> 1 with open('./preprocessed/labels_dict.json') as fp:
      2     labels_dict = json.load(fp)

FileNotFoundError: [Errno 2] No such file or directory: './preprocessed/labels_dict.json'

## === cell 10
if do_run:
    trainor = Trainor(
        './preprocessed/train.csv',
        './preprocessed/val_curated.csv',
        labels_dict)

## === cell 11
if do_run:
    trainor.train(10000)

## === cell 12
class Predictor:
    def __init__(self,
                 files,
                 lr=1e-3,
                 batch_size=16,
                 model_dir='ds2'):
        self.model_dir = model_dir
        self.model = Model((300, 300, 1))
        self._define_dataset_iterator(
            files, batch_size)

        logits = self.model(self.features)
        self.probas = tf.nn.sigmoid(logits)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self._load_model()

    def _define_dataset_iterator(self, files, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((files,))
        dataset = dataset.map(wav_to_spectogram)

        dataset = dataset.batch(batch_size).prefetch(10)

        dataset = dataset.repeat(1)
        self.features = dataset.make_one_shot_iterator().get_next()

    def _load_model(self):
        files = os.listdir(self.model_dir)
        steps = [int(file.split('-')[-1].split('.')[0]) for file in files if 'model' in file]
        ckpt_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.restore(self.sess, ckpt_path + '-{}'.format(max(steps)))

    def pred_generator(self):
        try:
            while True:
                yield self.sess.run(self.probas)
        except:
            print('Prediction over')

    def __del__(self):
        self.sess.close()

## === cell 13
tf.reset_default_graph()
if do_run:
    orig_files = os.listdir('../input/test')
    files = [os.path.join('../input/test', file) for file in orig_files]
    predictor = Predictor(files)
else:
    orig_files = os.listdir('../input/freesound-audio-tagging-2019/test')
    files = [os.path.join('../input/freesound-audio-tagging-2019/test', file) for file in orig_files]
    predictor = Predictor(files, model_dir='../input/cnnfat2019/')

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2754937507.py in <cell line: 0>()
----> 1 tf.reset_default_graph()
      2 if do_run:
      3     orig_files = os.listdir('../input/test')
      4     files = [os.path.join('../input/test', file) for file in orig_files]
      5     predictor = Predictor(files)

AttributeError: module 'tensorflow' has no attribute 'reset_default_graph'

## === cell 14
pred_gen = predictor.pred_generator()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1581715784.py in <cell line: 0>()
----> 1 pred_gen = predictor.pred_generator()

NameError: name 'predictor' is not defined

## === cell 15
pred_dict = {}
for label in labels_dict:
    pred_dict[label] = []

for probs in pred_gen:
    for label in labels_dict:
        pred_dict[label].extend(list(probs[:, labels_dict[label]]))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1520965193.py in <cell line: 0>()
      1 pred_dict = {}
----> 2 for label in labels_dict:
      3     pred_dict[label] = []
      4 
      5 for probs in pred_gen:

NameError: name 'labels_dict' is not defined

## === cell 16
pred_dict['fname'] = orig_files
df = pd.DataFrame(pred_dict, columns=['fname'].extend(list(labels_dict.keys())))
df.to_csv('submission.csv', index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3242947170.py in <cell line: 0>()
----> 1 pred_dict['fname'] = orig_files
      2 df = pd.DataFrame(pred_dict, columns=['fname'].extend(list(labels_dict.keys())))
      3 df.to_csv('submission.csv', index=False)

NameError: name 'orig_files' is not defined
