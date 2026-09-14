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

## === cell 1
import pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

data = pandas.read_table("../input/movie-review-sentiment-analysis-kernels-only/train.tsv")
labels = ["0", "1", "2", "3", "4"]

numberPhrase = data.groupby("Sentiment").count().PhraseId
plt.pie(list(numberPhrase), labels=labels, autopct='%.2f%%', shadow=True)

plt.title('Ratio diagram of Phrase')
plt.show()

numberSentence = data.groupby("Sentiment").SentenceId.nunique()
plt.pie(list(numberSentence), labels=labels, autopct='%.2f%%', shadow=True)
plt.title('Ratio diagram of Sentence')
plt.show()

numberLengthText = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
for i in range(len(data.PhraseId)):
    phrase = data.Phrase[i]
    words = phrase.split(" ")
    numberLengthText[min(len(words)-1, 2)][int(data.Sentiment[i])] += 1

numberLengthText = numpy.array(numberLengthText, dtype=numpy.float32)
for i in range(3):
    numberLengthText[i, :] /= numpy.sum(numberLengthText[i, :])
x = ["1", "2", ">=3"]
x_ = numpy.arange(len(x))
plt.bar(x_-0.2, numberLengthText[:, 0], 0.1, label=labels[0])
plt.bar(x_-0.1, numberLengthText[:, 1], 0.1, label=labels[1])
plt.bar(x_, numberLengthText[:, 2], 0.1, label=labels[2])
plt.bar(x_+0.1, numberLengthText[:, 3], 0.1, label=labels[3])
plt.bar(x_+0.2, numberLengthText[:, 4], 0.1, label=labels[4])
plt.xticks(x_, x)
plt.legend()
plt.xlabel('Number of words in phrase')
plt.ylabel('Percentage')
plt.title('Ratio diagram of Phrase by word numbers')
plt.show()

## === cell 3
import os
import word2vec
import numpy

class Preprocessor:
    @staticmethod
    def lemmatize(word):
        if word[-4:] == "sess":
            return word[:-4]+"ss"
        if word[-3:] == "ies":
            return word[:-3]+"y"
        if word[-2:] == "ss":
            return word
        if word[-1:] == "s":
            return word[:-1]
        if word == "'s":
            return "be"

    @staticmethod
    def tokenize(sentence):
        return sentence.lower().split(" ")

class W2VProcessor:
    def __init__(self, originData=None, dataFolder="", vectorSize=100):
        self.__model = None
        self.__vectorSize = vectorSize
        if type(originData) is str:
            word2vec.word2vec(
                originData, 
                os.path.join(dataFolder, "vec.w2v"), 
                size=vectorSize, 
                verbose=True)
            self.__model = word2vec.load(os.path.join(dataFolder, "vec.w2v"))

    def load(self, wordVectorFile):
        self.__model = word2vec.load(wordVectorFile)
        self.__vectorSize = self.__model.vectors.shape[1]

    def getVectorSize(self):
        return self.__vectorSize

    def process(self, sentence, length=None):
        if self.__model is None:
            print("Error: The model is None")
            return None

        if not sentence:
            print("Error: The sentence is None")
            return None
        
        sentence = Preprocessor.tokenize(sentence)

        if length is None:
            length = len(sentence)
        sentence = sentence[:length]
        tensor = []
        for word in sentence:
            try:
                tensor.append(self.__model[word])
            except:
                try:
                    tensor.append(self.__model[Preprocessor.lemmatize(word)])
                except:    
                    tensor.append(numpy.zeros((self.__vectorSize,)))
        for i in range(length-len(sentence)):
            tensor.append(numpy.zeros(tensor[0].shape))
        tensor = numpy.concatenate(tensor).reshape((length, len(tensor[0])))

        return tensor


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3349221081.py in <cell line: 0>()
      1 import os
----> 2 import word2vec
      3 import numpy
      4 
      5 class Preprocessor:

ModuleNotFoundError: No module named 'word2vec'

## === cell 5
import numpy
import tensorflow

class CNN:
    def __init__(self, wordVectSize=200, numOfWords=25, learningRate=0.001, numFilters=32):
        self.__numberOfWords = numOfWords
        self.__sizeOfWordVectors = wordVectSize
        self.__graph = tensorflow.Graph()
        self.__globalStep = tensorflow.Variable(0, trainable=False)
        self.__learningRate = tensorflow.train.exponential_decay(
            learningRate, self.__globalStep, decay_steps=100, decay_rate=0.98, staircase=True)
        
        self.__input = tensorflow.placeholder(tensorflow.float32, shape=[None, numOfWords, wordVectSize])
        self.__label = tensorflow.placeholder(tensorflow.float32, shape=[None])
        self.__dropout = tensorflow.placeholder(tensorflow.float32, shape=[])
        
        conv1 = tensorflow.layers.conv1d(
            self.__input, filters=numFilters, kernel_size=1, padding="valid")
        conv3 = tensorflow.layers.conv1d(
            self.__input, filters=numFilters, kernel_size=3, padding="valid")
        conv5 = tensorflow.layers.conv1d(
            self.__input, filters=numFilters, kernel_size=5, padding="valid")

        pool1 = tensorflow.layers.max_pooling1d(conv1, pool_size=(numOfWords, ), strides=1)
        pool3 = tensorflow.layers.max_pooling1d(conv3, pool_size=(numOfWords-2, ), strides=1)
        pool5 = tensorflow.layers.max_pooling1d(conv5, pool_size=(numOfWords-4, ), strides=1)
        
        conca = tensorflow.concat([pool1, pool3, pool5], axis=2)
        dropo = tensorflow.nn.dropout(conca, self.__dropout)
        self.__logits = tensorflow.layers.dense(dropo, units=5, activation=tensorflow.nn.softmax)
        self.__classes = tensorflow.reshape(tensorflow.argmax(self.__logits, axis=2), [-1])
        
        onehotLabels = tensorflow.one_hot(
            indices=tensorflow.cast(self.__label, tensorflow.int32), depth=5)
        onehotLabels = tensorflow.reshape(onehotLabels, [-1, 1, 5])
        self.__loss = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=onehotLabels, logits=self.__logits)

        self.__trainOp = tensorflow.contrib.layers.optimize_loss(
            loss=self.__loss,
            global_step=self.__globalStep,
            learning_rate=self.__learningRate,
            optimizer="SGD"
        )

        self.__session = tensorflow.Session()
        self.__session.run(tensorflow.global_variables_initializer())
        self.__saver = tensorflow.train.Saver()
        
    def fit_on_batch(self, trainData, label):
        feedDict = {
            self.__input: trainData,
            self.__label: label,
            self.__dropout: 0.5
        }

        _, loss = self.__session.run([self.__trainOp, self.__loss], feed_dict=feedDict)
        return loss

    def predict(self, testData):
        return self.__session.run([self.__classes], feed_dict={self.__input: testData, self.__dropout: 1.0})[0]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
import random
import pandas
import numpy

sentenceSize = 25
processor = W2VProcessor()
processor.load("../input/word2vec-model/vectors.bin")
clf = CNN(numOfWords=sentenceSize)
epochNum = 5
batchSize = 15
trainRatio = 1.0

processedData = [processor.process(phrase, sentenceSize) for phrase in data.Phrase]
trainingData = processedData[:int(trainRatio*len(processedData))]
testingData = processedData[int(trainRatio*len(processedData)):]

X = trainingData
Y = pandas.to_numeric(data.Sentiment[:int(trainRatio*len(processedData))])
x = testingData
y = pandas.to_numeric(data.Sentiment[int(trainRatio*len(processedData)):])

for epoch in range(epochNum):
    totalLoss = 0
    indice = 0
    for i in range(0, len(trainingData), batchSize):
        totalLoss += clf.fit_on_batch(numpy.array(X[i:i+batchSize]), numpy.array(Y[i:i+batchSize]))
        indice += 1
    print("epoch: ", epoch, " loss: ", totalLoss/indice)

    if len(testingData) > 0:
        totalTrue = 0
        for j in range(0, len(testingData), batchSize):
            zj = clf.predict(numpy.array(x[j:j+batchSize]))
            yj = numpy.array(y[j:j+batchSize])
            totalTrue += numpy.count_nonzero(yj-zj==0)

        accuracy = 1.0*totalTrue/len(testingData)
        print("cross-validation accuracy: ", accuracy)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3007353667.py in <cell line: 0>()
      4 
      5 sentenceSize = 25
----> 6 processor = W2VProcessor()
      7 processor.load("../input/word2vec-model/vectors.bin")
      8 clf = CNN(numOfWords=sentenceSize)

NameError: name 'W2VProcessor' is not defined

## === cell 9
batchSize = 30
blindData = pandas.read_table("../input/movie-review-sentiment-analysis-kernels-only/test.tsv")
with open("submission.csv", "w") as outFile:
    outFile.write("PhraseId,Sentiment\n")
    for i in range(0, len(blindData.PhraseId), batchSize):
        processedBlindData = [processor.process(phrase, sentenceSize) for phrase in blindData.Phrase[i:i+batchSize]]
        z = clf.predict(numpy.array(processedBlindData))
        for j in range(len(z)):
            outFile.write(str(blindData.PhraseId[i+j])+","+str(int(z[j]))+"\n")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3983636045.py in <cell line: 0>()
      4     outFile.write("PhraseId,Sentiment\n")
      5     for i in range(0, len(blindData.PhraseId), batchSize):
----> 6         processedBlindData = [processor.process(phrase, sentenceSize) for phrase in blindData.Phrase[i:i+batchSize]]
      7         z = clf.predict(numpy.array(processedBlindData))
      8         for j in range(len(z)):

/tmp/ipykernel_11/3983636045.py in <listcomp>(.0)
      4     outFile.write("PhraseId,Sentiment\n")
      5     for i in range(0, len(blindData.PhraseId), batchSize):
----> 6         processedBlindData = [processor.process(phrase, sentenceSize) for phrase in blindData.Phrase[i:i+batchSize]]
      7         z = clf.predict(numpy.array(processedBlindData))
      8         for j in range(len(z)):

NameError: name 'processor' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
