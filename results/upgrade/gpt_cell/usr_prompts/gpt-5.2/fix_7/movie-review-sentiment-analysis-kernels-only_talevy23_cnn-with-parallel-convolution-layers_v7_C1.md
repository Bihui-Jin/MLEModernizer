# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

seed = 12345
import random
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

random.seed(seed)
np.random.seed(seed)


## === cell 1
train,test,sampleSubmission = pd.read_csv('../input/train.tsv', sep = '\t'),pd.read_csv('../input/test.tsv', sep = '\t'),pd.read_csv('../input/sampleSubmission.csv')

train.head(3)


## === cell 2
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

phrases = [clean_str(s) for s in train['Phrase']]


## === cell 3
type(train['Phrase'])
len(phrases[2].split())
phrases[2].split()


## === cell 4
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

seed = 12345
import random
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

random.seed(seed)
np.random.seed(seed)


## === cell 5
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

t = Tokenizer()
t.fit_on_texts(phrases)
vocab_size = len(t.word_index) + 1
X = t.texts_to_sequences(phrases)
max_length = max([len(test.split()) for test in phrases])
X = pad_sequences(X, maxlen=max_length, padding="post")
print(X.shape)


## --- ERROR in cell 5, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/564379402.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# (set in an earlier cell), which triggers the protobuf C-extension ImportError.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;31m# Using Keras 3's built-in preprocessing avoids importing TensorFlow while keeping identical logic.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mpreprocessing[0m[0;34m.[0m[0mtext[0m [0;32mimport[0m [0mTokenizer[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mpreprocessing[0m[0;34m.[0m[0msequence[0m [0;32mimport[0m [0mpad_sequences[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# DO NOT EDIT. Generated by api_gen.sh[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFloatDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFunction[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mInitializer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m """
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mdeserialize[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mget[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mserialize[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mconstraints[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mdatasets[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mimport[0m [0mtypes[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mcelu[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0melu[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mexponential[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0mKerasTensor[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0many_symbolic_tensors[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m [0;32mimport[0m [0mbackend_utils[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mAutocastScope[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mVariable[0m [0;32mas[0m [0mKerasVariable[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mget_autocast_scope[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mstandardize_dtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0mBOOL_TYPES[0m [0;34m=[0m [0;34m([0m[0;34m"bool"[0m[0;34m,[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py[0m in [0;36m<module>[0;34m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0mget_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0min_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mnaming[0m [0;32mimport[0m [0mauto_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0maudio_dataset_utils[0m [0;32mimport[0m [0maudio_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdataset_utils[0m [0;32mimport[0m [0msplit_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mfile_utils[0m [0;32mimport[0m [0mget_file[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_dataset_utils[0m [0;32mimport[0m [0mimage_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_utils[0m [0;32mimport[0m [0marray_to_img[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mdataset_utils[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow_io[0m [0;32mas[0m [0mtfio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mtree[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mio_utils[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_paths[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_structure[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten_with_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mis_nested[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;32mif[0m [0moptree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0moptree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32melif[0m [0mdmtree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0mdmtree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py[0m in [0;36m<module>[0;34m[0m
[1;32m     11[0m [0;31m# Register backend-specific node classes[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mif[0m [0mbackend[0m[0;34m([0m[0;34m)[0m [0;34m==[0m [0;34m"tensorflow"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0mListWrapper[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0m_DictWrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;31m# source: tensorflow/core/framework/attr_value.proto[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m"""Generated protocol buffer code."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0menum_type_wrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mpython_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mreflection[0m [0;32mas[0m [0m_reflection[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py[0m in [0;36m<module>[0;34m[0m
[1;32m     36[0m [0;32mimport[0m [0mweakref[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0mdescriptor_mod[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0mmessage_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mtext_format[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py[0m in [0;36m<module>[0;34m[0m
[1;32m     27[0m   [0;31m# TODO: Remove this import after fix api_implementation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m   [0;32mif[0m [0m_message[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0mpyext[0m [0;32mimport[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m   [0m_USE_C_DESCRIPTORS[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0;34m[0m[0m

[0;31mImportError[0m: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 6
from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,merge, concatenate, Input, Dropout

def model3(output_dim=8, max_length=50, y_dim=5, num_filters=5, filter_sizes = [3,5]):
    embed_input = Input(shape=(max_length,))
    x = Embedding(vocab_size,output_dim,input_length=max_length)(embed_input)
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        conv = MaxPooling1D(pool_size=max_length-filter_sizes[i]+1, strides=1, padding='valid')(conv)
        pooled_outputs.append(conv)
    merge = concatenate(pooled_outputs)
        
    x = Flatten()(merge)
    x = Dropout(0.2)(x)
    predictions = Dense(y_dim, activation = 'sigmoid')(x)
    
    model = Model(inputs=embed_input,outputs=predictions)

    model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())
    
    from keras.utils import plot_model
    plot_model(model, to_file='shared_input_layer.png')
    
    return model


model = model3(output_dim=8, max_length=max_length,y_dim=5,filter_sizes = [3,4,5])
from IPython.display import Image
Image(filename='shared_input_layer.png')
