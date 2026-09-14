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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

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
protobuf==6.33.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.5178905448621252

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

data = pandas.read_table(
    "../input/movie-review-sentiment-analysis-kernels-only/train.tsv"
)
labels = ["0", "1", "2", "3", "4"]

numberPhrase = data.groupby("Sentiment").count().PhraseId
plt.pie(list(numberPhrase), labels=labels, autopct="%.2f%%", shadow=True)

plt.title("Ratio diagram of Phrase")
plt.show()

numberSentence = data.groupby("Sentiment").SentenceId.nunique()
plt.pie(list(numberSentence), labels=labels, autopct="%.2f%%", shadow=True)
plt.title("Ratio diagram of Sentence")
plt.show()

numberLengthText = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
for i in range(len(data.PhraseId)):
    phrase = data.Phrase[i]
    words = phrase.split(" ")
    numberLengthText[min(len(words) - 1, 2)][int(data.Sentiment[i])] += 1

numberLengthText = numpy.array(numberLengthText, dtype=numpy.float32)
for i in range(3):
    numberLengthText[i, :] /= numpy.sum(numberLengthText[i, :])
x = ["1", "2", ">=3"]
x_ = numpy.arange(len(x))
plt.bar(x_ - 0.2, numberLengthText[:, 0], 0.1, label=labels[0])
plt.bar(x_ - 0.1, numberLengthText[:, 1], 0.1, label=labels[1])
plt.bar(x_, numberLengthText[:, 2], 0.1, label=labels[2])
plt.bar(x_ + 0.1, numberLengthText[:, 3], 0.1, label=labels[3])
plt.bar(x_ + 0.2, numberLengthText[:, 4], 0.1, label=labels[4])
plt.xticks(x_, x)
plt.legend()
plt.xlabel("Number of words in phrase")
plt.ylabel("Percentage")
plt.title("Ratio diagram of Phrase by word numbers")
plt.show()



## === cell 2
import os
import numpy as np


class Preprocessor:
    @staticmethod
    def lemmatize(word):
        if word[-4:] == "sess":
            return word[:-4] + "ss"
        if word[-3:] == "ies":
            return word[:-3] + "y"
        if word[-2:] == "ss":
            return word
        if word[-1:] == "s":
            return word[:-1]
        if word == "'s":
            return "be"
        return word

    @staticmethod
    def tokenize(sentence):
        return sentence.lower().split(" ")


class W2VProcessor:
    def __init__(self, originData=None, dataFolder="", vectorSize=200):
        self.__model = None
        self.__vectorSize = vectorSize
        if originData is not None:
            pass

    def load(self, wordVectorFile):
        self.__model = None
        self.__vectorSize = self.__vectorSize

    def getVectorSize(self):
        return self.__vectorSize

    def process(self, sentence, length=None):
        """
        Returns a (length, vectorSize) array.
        Unknown words are represented by zero vectors.
        """
        if length is None:
            length = len(sentence.split())
        tokens = Preprocessor.tokenize(sentence)[:length]

        vecs = np.zeros((length, self.__vectorSize), dtype=np.float32)

        return vecs




## === cell 4
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class CNN:
    def __init__(
        self, wordVectSize=200, numOfWords=25, learningRate=0.001, numFilters=32
    ):
        self.__numberOfWords = numOfWords
        self.__sizeOfWordVectors = wordVectSize

        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__globalStep = tf.Variable(0, trainable=False, name="global_step")
            self.__learningRate = tf.train.exponential_decay(
                learningRate,
                self.__globalStep,
                decay_steps=100,
                decay_rate=0.98,
                staircase=True,
            )

            self.__input = tf.placeholder(
                tf.float32, shape=[None, numOfWords, wordVectSize], name="input"
            )
            self.__label = tf.placeholder(tf.int32, shape=[None], name="label")
            self.__dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

            conv1 = tf.layers.conv1d(
                self.__input,
                filters=numFilters,
                kernel_size=1,
                padding="valid",
                activation=tf.nn.relu,
            )
            conv3 = tf.layers.conv1d(
                self.__input,
                filters=numFilters,
                kernel_size=3,
                padding="valid",
                activation=tf.nn.relu,
            )
            conv5 = tf.layers.conv1d(
                self.__input,
                filters=numFilters,
                kernel_size=5,
                padding="valid",
                activation=tf.nn.relu,
            )

            pool1 = tf.layers.max_pooling1d(
                conv1, pool_size=self.__numberOfWords, strides=1, padding="valid"
            )
            pool3 = tf.layers.max_pooling1d(
                conv3, pool_size=self.__numberOfWords - 2, strides=1, padding="valid"
            )
            pool5 = tf.layers.max_pooling1d(
                conv5, pool_size=self.__numberOfWords - 4, strides=1, padding="valid"
            )

            conca = tf.concat(
                [pool1, pool3, pool5], axis=2
            )  # shape [batch, 1, 3*numFilters]
            conca = tf.squeeze(conca, axis=1)  # shape [batch, 3*numFilters]

            dropo = tf.nn.dropout(conca, keep_prob=self.__dropout)

            self.__logits = tf.layers.dense(dropo, units=5)

            self.__classes = tf.argmax(self.__logits, axis=1, output_type=tf.int32)

            onehot_labels = tf.one_hot(self.__label, depth=5)
            self.__loss = tf.losses.softmax_cross_entropy(onehot_labels, self.__logits)

            optimizer = tf.train.GradientDescentOptimizer(self.__learningRate)
            self.__trainOp = optimizer.minimize(
                self.__loss, global_step=self.__globalStep
            )

            self.__init_op = tf.global_variables_initializer()
            self.__saver = tf.train.Saver()

        self.__session = tf.Session(graph=self.__graph)
        self.__session.run(self.__init_op)

    def fit_on_batch(self, trainData, label):
        feed_dict = {self.__input: trainData, self.__label: label, self.__dropout: 0.5}
        _, loss = self.__session.run([self.__trainOp, self.__loss], feed_dict=feed_dict)
        return loss

    def predict(self, testData):
        feed_dict = {self.__input: testData, self.__dropout: 1.0}
        preds = self.__session.run(self.__classes, feed_dict=feed_dict)
        return preds




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
import random
import pandas
import numpy as np

sentenceSize = 25
processor = W2VProcessor(vectorSize=200)  # vector size must match CNN default
processor.load("../input/word2vec-model/vectors.bin")  # fallback load does nothing

clf = CNN(numOfWords=sentenceSize)
epochNum = 5
batchSize = 15
trainRatio = 1.0

processedData = [processor.process(phrase, sentenceSize) for phrase in data.Phrase]
trainingData = processedData[: int(trainRatio * len(processedData))]
testingData = processedData[int(trainRatio * len(processedData)) :]

X = trainingData
Y = (
    pandas.to_numeric(data.Sentiment[: int(trainRatio * len(processedData))])
    .astype(int)
    .values
)
x = testingData
y = (
    pandas.to_numeric(data.Sentiment[int(trainRatio * len(processedData)) :])
    .astype(int)
    .values
)

for epoch in range(epochNum):
    totalLoss = 0.0
    batches = 0
    for i in range(0, len(trainingData), batchSize):
        batch_X = np.array(X[i : i + batchSize])
        batch_Y = np.array(Y[i : i + batchSize])
        loss = clf.fit_on_batch(batch_X, batch_Y)
        totalLoss += loss
        batches += 1
    print("epoch:", epoch, " loss:", totalLoss / batches if batches else 0)

    if len(testingData) > 0:
        totalTrue = 0
        totalSamples = 0
        for j in range(0, len(testingData), batchSize):
            batch_x = np.array(x[j : j + batchSize])
            batch_y = np.array(y[j : j + batchSize])
            preds = clf.predict(batch_x)
            totalTrue += np.sum(preds == batch_y)
            totalSamples += len(batch_y)
        accuracy = totalTrue / totalSamples if totalSamples else 0
        print("cross-validation accuracy:", accuracy)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/410866069.py in <cell line: 0>()
      7 processor.load("../input/word2vec-model/vectors.bin")  # fallback load does nothing
      8 
----> 9 clf = CNN(numOfWords=sentenceSize)
     10 epochNum = 5
     11 batchSize = 15

/tmp/ipykernel_55/129216147.py in __init__(self, wordVectSize, numOfWords, learningRate, numFilters)
     30 
     31             # Convolution layers
---> 32             conv1 = tf.layers.conv1d(
     33                 self.__input,
     34                 filters=numFilters,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `conv1d` is not available with Keras 3.

## === cell 8
batchSize = 30
blindData = pandas.read_table(
    "../input/movie-review-sentiment-analysis-kernels-only/test.tsv"
)
predictions = []

for i in range(0, len(blindData), batchSize):
    phrases_batch = blindData.Phrase[i : i + batchSize]
    processedBatch = [processor.process(p, sentenceSize) for p in phrases_batch]
    batch_pred = clf.predict(np.array(processedBatch))
    predictions.extend(batch_pred.tolist())

assert len(predictions) == len(blindData), "Prediction length mismatch"

with open("submission.csv", "w") as outFile:
    outFile.write("PhraseId,Sentiment\n")
    for pid, pred in zip(blindData.PhraseId, predictions):
        outFile.write(f"{pid},{int(pred)}\n")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2829719749.py in <cell line: 0>()
      8     phrases_batch = blindData.Phrase[i : i + batchSize]
      9     processedBatch = [processor.process(p, sentenceSize) for p in phrases_batch]
---> 10     batch_pred = clf.predict(np.array(processedBatch))
     11     predictions.extend(batch_pred.tolist())
     12 

NameError: name 'clf' is not defined
