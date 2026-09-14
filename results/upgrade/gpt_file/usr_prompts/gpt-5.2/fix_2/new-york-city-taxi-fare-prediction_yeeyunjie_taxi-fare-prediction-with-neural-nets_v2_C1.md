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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.9

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
statsmodels==0.14.5
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

5.07016

# 6. Current score

954.05739

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 954.05739) has done: 'I fix the notebook-breaking issues without changing the core modeling approach: remove Jupyter magics, avoid importing Keras at import-time (it triggers the protobuf `MessageFactory/GetPrototype` crash in this environment), and correct the Pandas `.any(1)` calls to `.any(axis=1)` so the data-cleaning pipeline runs. I also fix the Haversine distance function (it currently applies `math.radians` to Series and fail) by implementing a vectorized numpy version that preserves the same feature intent but runs fast on 1M rows. Finally, I ensure the script trains the same Ridge baseline (already present) and writes a valid `submission.csv` with the exact required columns and a `.csv` suffix in `/kaggle/working/`.'

# 9. Code solution

## === cell 0
import os

for dirname, _, filenames in os.walk(
    "/kaggle/input/new-york-city-taxi-fare-prediction"
):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore")


RANDOM_STATE = 42

DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")



## === cell 2
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test_df = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("Train loaded:", df.shape, "Test loaded:", test_df.shape, "Sample:", sample.shape)
print(df.head(2))
print(test_df.head(2))



## === cell 3
print(f"Number of records: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")



## === cell 4
df.info()



## === cell 5
df.describe()



## === cell 6
df.head()



## === cell 7
null_rows = df[df.isnull().any(axis=1)]
print("Rows with any nulls:", null_rows.shape[0])



## === cell 8
df.columns[df.isnull().any()]



## === cell 9
df1 = df[~df.isnull().any(axis=1)].copy()
print("After dropping null rows:", df1.shape)



## === cell 10
incorrect_location = df1[
    ((df1["dropoff_latitude"] < 0) | (df1["pickup_latitude"] < 0))
    & ((df1["dropoff_longitude"] > 0) | (df1["pickup_longitude"] > 0))
].copy()
print("Incorrect-location rows:", incorrect_location.shape)



## === cell 11
if len(incorrect_location) > 0:
    incorrect_location = incorrect_location.rename(
        columns={
            "pickup_latitude": "pickup_longitude",
            "pickup_longitude": "pickup_latitude",
            "dropoff_latitude": "dropoff_longitude",
            "dropoff_longitude": "dropoff_latitude",
        }
    )
    incorrect_location = incorrect_location[df1.columns]

    df1.loc[
        df1.index.isin(incorrect_location.index),
        [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ],
    ] = incorrect_location[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values



## === cell 12
remaining_odd = df1[
    ((df1["dropoff_latitude"] < 0) | (df1["pickup_latitude"] < 0))
    & ((df1["dropoff_longitude"] > 0) | (df1["pickup_longitude"] > 0))
]
print("Remaining odd coord rows (will drop):", remaining_odd.shape)



## === cell 13
todrop = remaining_odd
df1 = df1[~df1.index.isin(todrop.index)].copy()
print("After dropping remaining odd coords:", df1.shape)



## === cell 14
df1 = df1.drop(
    df1[
        (df1["dropoff_latitude"] == 0)
        & (df1["dropoff_longitude"] == 0)
        & (df1["pickup_latitude"] == 0)
        & (df1["pickup_longitude"] == 0)
    ].index
).copy()
print("After dropping (0,0) coords:", df1.shape)



## === cell 15
df1.head()



## === cell 16
todrop_lat = df1[
    (df1["pickup_latitude"].lt(24) | df1["pickup_latitude"].gt(50))
    | (df1["dropoff_latitude"].lt(24) | df1["dropoff_latitude"].gt(50))
]
print("Odd latitude rows:", todrop_lat.shape)



## === cell 17
todrop_lon = df1[
    (df1["pickup_longitude"].lt(-125) | df1["pickup_longitude"].gt(-67))
    | (df1["dropoff_longitude"].lt(-125) | df1["dropoff_longitude"].gt(-67))
]
print("Odd longitude rows:", todrop_lon.shape)



## === cell 18
df1 = df1.drop(todrop_lat.index).drop(todrop_lon.index).copy()
print("After dropping odd lat/lon:", df1.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/234141308.py in <cell line: 0>()
----> 1 df1 = df1.drop(todrop_lat.index).drop(todrop_lon.index).copy()
      2 print("After dropping odd lat/lon:", df1.shape)
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: '[472, 1181, 1260, 2397, 4278, 4597, 6188, 6302, 8647, 9147, 10215, 10458, 10488, 10672, 11653, 12705, 12983, 13340, 14197, 14308, 15783, 15919, 16690, 17159, 18388, 18842, 19385, 19775, 20508, 21085, 21303, 22090, 22183, 23737, 24519, 26101, 26994, 28809, 28839, 29547, 30934, 31823, 31836, 32350, 32535, 32799, 32802, 32870, 32916, 33042, 33053, 33889, 34425, 34503, 35213, 35222, 35249, 36250, 36493, 37618, 37770, 37798, 37888, 40109, 40673, 42347, 42606, 43208, 44045, 44543, 44754, 44898, 45597, 45845, 46082, 46463, 46790, 46908, 47045, 47311, 47804, 48058, 48996, 49215, 50380, 50678, 50784, 51743, 54350, 56251, 57003, 57653, 57817, 59931, 60411, 60634, 61898, 62877, 63052, 63491, 63888, 64810, 65396, 65945, 66890, 67343, 67511, 67590, 68495, 68501, 69150, 69195, 69832, 70228, 70511, 71502, 71613, 71675, 71878, 72086, 73261, 73285, 73342, 74007, 74092, 76436, 77145, 77621, 77837, 77934, 80442, 80900, 81079, 81208, 82052, 82205, 82402, 82518, 82573, 84362, 84784, 84889, 85267, 86677, 86856, 87165, 87172, 87930, 88191, 89002, 89065, 89188, 89335, 91646, 91721, 93095, 93106, 93280, 93361, 93576, 93784, 93848, 95278, 96204, 96403, 96562, 98052, 98095, 98152, 98167, 98596, 99737, 102303, 103935, 104207, 104811, 105009, 105620, 106497, 107028, 107926, 108050, 108494, 108513, 109020, 109309, 109441, 109529, 110172, 110356, 110645, 111113, 111809, 114102, 115054, 115345, 117800, 122170, 122336, 123453, 123644, 124263, 127017, 127312, 127683, 128184, 128190, 128290, 128488, 129141, 130731, 132235, 132492, 132874, 132941, 134093, 135075, 135369, 135907, 136097, 138000, 138396, 139911, 140392, 140586, 141716, 141760, 142696, 143655, 143668, 144818, 144887, 145940, 146250, 147399, 147950, 148094, 148321, 149364, 150065, 150559, 150704, 150970, 151249, 151302, 151775, 152370, 152650, 154024, 154911, 156390, 156708, 157560, 157994, 158166, 158841, 158859, 158966, 160497, 160601, 160669, 162640, 162780, 162937, 163521, 163949, 164086, 164188, 164955, 164959, 165436, 166426, 167740, 167804, 168376, 169111, 169660, 169719, 170037, 170076, 170178, 170426, 170472, 170628, 170794, 171960, 172206, 172598, 172835, 173862, 174210, 174323, 174670, 174712, 175318, 175987, 177008, 177197, 177656, 178017, 178360, 178401, 178802, 179693, 181091, 181226, 181273, 182782, 182877, 182940, 183810, 183938, 184795, 185234, 185560, 185771, 185916, 186120, 186499, 186745, 187175, 187279, 188196, 188297, 188480, 188729, 189111, 189468, 189488, 189601, 190309, 191083, 191816, 192460, 193490, 193958, 194523, 194927, 197353, 197403, 198033, 198036, 198112, 198211, 198500, 199723, 200654, 200655, 201910, 201937, 202205, 202227, 202252, 202754, 203781, 204303, 204711, 205357, 205407, 205415, 205481, 205569, 205939, 206793, 207243, 207647, 207676, 207939, 207972, 208340, 208413, 208883, 208942, 209455, 209756, 210384, 210390, 210874, 211987, 212213, 212532, 212547, 212808, 213106, 213244, 213307, 213330, 213455, 214008, 214270, 215184, 216030, 216372, 217355, 218221, 218658, 219013, 219218, 220245, 220338, 221326, 221558, 221665, 221770, 221806, 222417, 223498, 223830, 224717, 224848, 225470, 227764, 227776, 227796, 228124, 228861, 229154, 229517, 231792, 232892, 232972, 233081, 233984, 234241, 234669, 234707, 235453, 236086, 236380, 237168, 237619, 238478, 238716, 239350, 239458, 240336, 240605, 241563, 241610, 243362, 243766, 243843, 244041, 244753, 245093, 245206, 245740, 245904, 246245, 246681, 246795, 246870, 246951, 247687, 248068, 248623, 248723, 249173, 249426, 249766, 250277, 250647, 250661, 251107, 252293, 253496, 255016, 255217, 256475, 258861, 259073, 259219, 259272, 260096, 260581, 260635, 262151, 263218, 263569, 264622, 266325, 266801, 268728, 270779, 271332, 271481, 271852, 273500, 274227, 274503, 275656, 276629, 276807, 277070, 277148, 277236, 278233, 278483, 278989, 280896, 281694, 282752, 283271, 283746, 283955, 284004, 284076, 284312, 285273, 285389, 285795, 285931, 286333, 286440, 287796, 288011, 288462, 288735, 289047, 291721, 292877, 293316, 296261, 296342, 298117, 298174, 298693, 298931, 298954, 299467, 299561, 300251, 300412, 301807, 301889, 302362, 302610, 303291, 304253, 304265, 304624, 305167, 305357, 305793, 306221, 307631, 309126, 309259, 309307, 311581, 312346, 312720, 313081, 313932, 314773, 315470, 315682, 316091, 316770, 316845, 316855, 317021, 317368, 319352, 319939, 321401, 321699, 322980, 323168, 323286, 323704, 323853, 323864, 325007, 325697, 325802, 326033, 326856, 327049, 328054, 328065, 328138, 329475, 330202, 330307, 330376, 331074, 331115, 331392, 332337, 333326, 334341, 334883, 335713, 336117, 336392, 337147, 337928, 338281, 339794, 339941, 340445, 341184, 341609, 342356, 343889, 344621, 344636, 344719, 345454, 345887, 345913, 346155, 346416, 346822, 347117, 347150, 348406, 348978, 349093, 349277, 349507, 350486, 350978, 351546, 351625, 353847, 354195, 354221, 354297, 354793, 354821, 355404, 355447, 355577, 356694, 356815, 357526, 358012, 358274, 358530, 359149, 359672, 362464, 362806, 363884, 363963, 364034, 364214, 366906, 367443, 367786, 368496, 368719, 368874, 369020, 370156, 370450, 370560, 371095, 372598, 372624, 372912, 373335, 373470, 373959, 374239, 374296, 374400, 374476, 375744, 375774, 376324, 376740, 377180, 377584, 378529, 379331, 379476, 379880, 380704, 380772, 381771, 381848, 382936, 383569, 383642, 383740, 383748, 384064, 384947, 385110, 385716, 386009, 386226, 386320, 386953, 388089, 388787, 389129, 389185, 389274, 390943, 391017, 391541, 392031, 392059, 392477, 392834, 393460, 393824, 394796, 394899, 397131, 397986, 398366, 398865, 399748, 400047, 401038, 401159, 401178, 401445, 401508, 402109, 402231, 402395, 402744, 403757, 405258, 405343, 405649, 406794, 407070, 407212, 407528, 407992, 408363, 408876, 409091, 409395, 410555, 410792, 412911, 413659, 413667, 413674, 414099, 414704, 415067, 415917, 416259, 416402, 416800, 417138, 417157, 417279, 417417, 418382, 419355, 419362, 419837, 421177, 421344, 422076, 422544, 422898, 423233, 423793, 424948, 425408, 425782, 426286, 426623, 426642, 427085, 427309, 428539, 429307, 429386, 429432, 429525, 429542, 429896, 430044, 430430, 430458, 431339, 431587, 431728, 431791, 431906, 432277, 433136, 435976, 436233, 436733, 437108, 437873, 438442, 438458, 439328, 440198, 440308, 440686, 440943, 441344, 442100, 442390, 443622, 444188, 447221, 447461, 447631, 447771, 448046, 448411, 448651, 448711, 448752, 449635, 449706, 450944, 451398, 451421, 452282, 452313, 452879, 453427, 453488, 453556, 454719, 455653, 456948, 457325, 458550, 459747, 459759, 460770, 461921, 462861, 462874, 462993, 464037, 464191, 464749, 464783, 464996, 466033, 468415, 468463, 468963, 469366, 470260, 472100, 473007, 474220, 474418, 474551, 475979, 475994, 476402, 476996, 477183, 479894, 480754, 480923, 481978, 482652, 482914, 483002, 483214, 484000, 484268, 485324, 485830, 486554, 487276, 487344, 487974, 488006, 488600, 489055, 489328, 489394, 489458, 489747, 490612, 491484, 492074, 492153, 492766, 493113, 494391, 495161, 495881, 496215, 497167, 498392, 499635, 499821, 500537, 500808, 502348, 503218, 503507, 505347, 505792, 507439, 508224, 509762, 510240, 510274, 511334, 512918, 512961, 514011, 514166, 514306, 514550, 514581, 514719, 515579, 515582, 516889, 517258, 520016, 520372, 520528, 523761, 524376, 524673, 524848, 525203, 525575, 526357, 526681, 526880, 526919, 526962, 527604, 527737, 527929, 528421, 528612, 528783, 529193, 529929, 530042, 530060, 530225, 531361, 533398, 534064, 535510, 536290, 538219, 538925, 540574, 540608, 540967, 541686, 542011, 542023, 543001, 543178, 543998, 544989, 545153, 546435, 547378, 547519, 547588, 547974, 548249, 550017, 550536, 551328, 551420, 553508, 555222, 555657, 555978, 556428, 556598, 557038, 557355, 557456, 559195, 560823, 562130, 562654, 562655, 562983, 563132, 563522, 564424, 564773, 566705, 566932, 567153, 567301, 569060, 569562, 571099, 572105, 572130, 572941, 575912, 576516, 576645, 576869, 576916, 576940, 578245, 578297, 579588, 579704, 579901, 580140, 581798, 582027, 582271, 584776, 584891, 585038, 585090, 585144, 585504, 586267, 586758, 588450, 588672, 588849, 589192, 589788, 589851, 590035, 591243, 591362, 591842, 592181, 592222, 592383, 592493, 592634, 593049, 593836, 594618, 595031, 595142, 597497, 597549, 597723, 597732, 598274, 598396, 598890, 599133, 599136, 599318, 601112, 601329, 601389, 601782, 602229, 602826, 602922, 603014, 603060, 603953, 604190, 604328, 605251, 605487, 607713, 608597, 608974, 609684, 610125, 610172, 611145, 611463, 611549, 613756, 617732, 617873, 618671, 618982, 619134, 619425, 619542, 619803, 620072, 620101, 620955, 621392, 622342, 622495, 623044, 623184, 625098, 625161, 625574, 630416, 630445, 630599, 630633, 630695, 630996, 631084, 631141, 632977, 633180, 634437, 634994, 636052, 636547, 637575, 637857, 638984, 639871, 640624, 641219, 641685, 641726, 641734, 643991, 644448, 646012, 646132, 646340, 646550, 647968, 648865, 650230, 650968, 651320, 652821, 653380, 654090, 654748, 656518, 656879, 657505, 657700, 657896, 658181, 658347, 659174, 660366, 660660, 660989, 661946, 662025, 662356, 662877, 663949, 664840, 665263, 665322, 665532, 666706, 667930, 668130, 670466, 670650, 670699, 670999, 671410, 672598, 673808, 673948, 674895, 674907, 675116, 675302, 678080, 678883, 679726, 680341, 681658, 681782, 681916, 682595, 682887, 683233, 683499, 683585, 683739, 684274, 684757, 685176, 685318, 686472, 686749, 687157, 687248, 688719, 688776, 688974, 688986, 690419, 690646, 690854, 691266, 691342, 691362, 691552, 691813, 692079, 692382, 693800, 693924, 695271, 695416, 695679, 696585, 697569, 698534, 698606, 700173, 701482, 701655, 701944, 704603, 704843, 704878, 705357, 706671, 708330, 708700, 709349, 709605, 710245, 711903, 712235, 712935, 713034, 713343, 713566, 715391, 715551, 715954, 716542, 717069, 717329, 718389, 718862, 719267, 719888, 720366, 720801, 721066, 721406, 721842, 722390, 723429, 725272, 725342, 725380, 725833, 726309, 726940, 727603, 727869, 727935, 728510, 730065, 730761, 731491, 731695, 731979, 732836, 733358, 734722, 734761, 735128, 735195, 737083, 737390, 737779, 737876, 737931, 738770, 739365, 740461, 740842, 741013, 741659, 742689, 743744, 743789, 743796, 743934, 744974, 745083, 745458, 745951, 746403, 748464, 748607, 750602, 750615, 750723, 750982, 752228, 752264, 752384, 752797, 752907, 753266, 754260, 755489, 755945, 756039, 756494, 756733, 757673, 758195, 758445, 758669, 758696, 760741, 763488, 764036, 764315, 764636, 766062, 766413, 766999, 768014, 768091, 768632, 769319, 770308, 770533, 771692, 771740, 772962, 773737, 774566, 774647, 775834, 776276, 776305, 776580, 777548, 777956, 779191, 780091, 780241, 781610, 782006, 783133, 785137, 785509, 786669, 786722, 787241, 787870, 788320, 789953, 790605, 790807, 793236, 793247, 795942, 796504, 796531, 797102, 797647, 798079, 798149, 798199, 799524, 800018, 800197, 800298, 800358, 800564, 800761, 801191, 802171, 802444, 802942, 803935, 805620, 805709, 805766, 806515, 806786, 806951, 807175, 807911, 808418, 808914, 809012, 810141, 810753, 810839, 810923, 811231, 812252, 812566, 814485, 814617, 815100, 815705, 815777, 816436, 817248, 817662, 818505, 818937, 819057, 819474, 819673, 819691, 820239, 821024, 821950, 822377, 823088, 823464, 823669, 823943, 826207, 826844, 828054, 828566, 828803, 829039, 829138, 829401, 829492, 830134, 831229, 831923, 832059, 832133, 832721, 832968, 833365, 833583, 833630, 834884, 835755, 835763, 836650, 836869, 837082, 837368, 837619, 837781, 839158, 839221, 839885, 840478, 840551, 840843, 841806, 842046, 842959, 843087, 844261, 845547, 846143, 847039, 847255, 847441, 848688, 848898, 849151, 849494, 849632, 851799, 854230, 854380, 854813, 854999, 856986, 857442, 858371, 858385, 858504, 858733, 858883, 859271, 860247, 861522, 862252, 862855, 863029, 863414, 863476, 864170, 864750, 864795, 865060, 865478, 865928, 866226, 867744, 868195, 870046, 870537, 871078, 871312, 871817, 873159, 873244, 873314, 873624, 874446, 874604, 875441, 876843, 877774, 878023, 878042, 882209, 882553, 882557, 883844, 883934, 884121, 885108, 885113, 885122, 885276, 885599, 885683, 886964, 887183, 887255, 887369, 887919, 889183, 889313, 891042, 891166, 893633, 895012, 895730, 895966, 897199, 897211, 897580, 898152, 899104, 899848, 899963, 900407, 900444, 900494, 902250, 903221, 903790, 904189, 904775, 904913, 905791, 905817, 906715, 907277, 908520, 908897, 908931, 909821, 909830, 909871, 910148, 911595, 911797, 912407, 913933, 913961, 914096, 914757, 914795, 915656, 916697, 919034, 919049, 920384, 920720, 920847, 921116, 921369, 921563, 921716, 921758, 922224, 922365, 922720, 923506, 924006, 925222, 925321, 926283, 926352, 926407, 926507, 926564, 926579, 926710, 927700, 928175, 929546, 929717, 929718, 930315, 930440, 930680, 930725, 931006, 931164, 931220, 932052, 932639, 932671, 932776, 933966, 935580, 936077, 937054, 937364, 937652, 937852, 938288, 940026, 940460, 940652, 940996, 942215, 942783, 942864, 943003, 943372, 943674, 945328, 945356, 945524, 946404, 947283, 947405, 947523, 947897, 947914, 947989, 948106, 949343, 949564, 950480, 952375, 953115, 953871, 953957, 954049, 954276, 954598, 955484, 955628, 955669, 956898, 957782, 959292, 960926, 961451, 961915, 962457, 962630, 962742, 963217, 964450, 965390, 965403, 965697, 967986, 968707, 969008, 970442, 970835, 970845, 971489, 972633, 972991, 973104, 973162, 974428, 976890, 977579, 977865, 978012, 978243, 979168, 979533, 979789, 980066, 980315, 981103, 981606, 981839, 982749, 982889, 982913, 983705, 984285, 985121, 986321, 986518, 986713, 987841, 988080, 988747, 989281, 992056, 992490, 992693, 992846, 993372, 993672, 994162, 994608, 994663, 994860, 994980, 995517, 995910, 996613, 997656, 997786, 998717, 999338] not found in axis'

## === cell 19
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.boxplot(x=test_df["pickup_latitude"], ax=ax[0])
sns.boxplot(x=test_df["pickup_longitude"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 20
test_df.describe()



## === cell 21
df1 = df1[
    ((df1["pickup_longitude"] > -75) & (df1["pickup_longitude"] < -72))
    & ((df1["pickup_latitude"] > 40) & (df1["pickup_latitude"] < 42))
    & ((df1["dropoff_longitude"] > -75) & (df1["dropoff_longitude"] < -72))
    & ((df1["dropoff_latitude"] > 40) & (df1["dropoff_latitude"] < 42))
].copy()
print("After NYC bbox filter:", df1.shape)



## === cell 22
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.boxplot(x=df1["pickup_latitude"], ax=ax[0])
sns.boxplot(x=df1["pickup_longitude"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 23
df1 = df1.drop(df1[df1["fare_amount"] <= 0].index).copy()
print("After dropping non-positive fares:", df1.shape)



## === cell 24
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=df1["passenger_count"])
plt.tight_layout()
plt.show()



## === cell 25
df1[df1["passenger_count"] > 50].head()



## === cell 26
df1 = df1.drop(df1[df1["passenger_count"] > 50].index).copy()
print("After dropping passenger_count>50:", df1.shape)



## === cell 27
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=df1["fare_amount"])
plt.tight_layout()
plt.show()



## === cell 28
df1 = df1.drop(df1[df1["fare_amount"] > 200].index).copy()
print("After dropping fare_amount>200:", df1.shape)



## === cell 29
pd.to_datetime(
    pd.to_datetime(df1.head()["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)



## === cell 30
df1["pickup_datetime"] = pd.to_datetime(
    pd.to_datetime(df1["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)
test_df["pickup_datetime"] = pd.to_datetime(
    pd.to_datetime(test_df["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)



## === cell 31
df1["year"] = df1["pickup_datetime"].dt.year
df1["month"] = df1["pickup_datetime"].dt.month
df1["day"] = df1["pickup_datetime"].dt.day
df1["weekday"] = df1["pickup_datetime"].dt.weekday
df1["hour"] = df1["pickup_datetime"].dt.hour
df1["min"] = df1["pickup_datetime"].dt.minute

test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["weekday"] = test_df["pickup_datetime"].dt.weekday
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["min"] = test_df["pickup_datetime"].dt.minute




## === cell 32
def haversine_np(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    """
    Vectorized haversine distance (km).
    Inputs are arrays/Series in decimal degrees.
    """
    lon1 = np.radians(pickup_lon.astype(float))
    lat1 = np.radians(pickup_lat.astype(float))
    lon2 = np.radians(dropoff_lon.astype(float))
    lat2 = np.radians(dropoff_lat.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r




## === cell 33
df1["distance"] = haversine_np(
    df1["pickup_longitude"],
    df1["pickup_latitude"],
    df1["dropoff_longitude"],
    df1["dropoff_latitude"],
)
print(df1[["distance"]].describe())



## === cell 34
df2 = df1.copy()
df2 = df2.drop(
    columns=[
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
)
print("df2 columns:", df2.columns.tolist())



## === cell 35
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)
test_df = test_df.drop(
    columns=[
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
)
print("test_df columns:", test_df.columns.tolist())



## === cell 36
corr = df2.drop(columns=["key"]).corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, ax=ax, annot=True, cmap=cmap, mask=mask, fmt=".2f")
plt.tight_layout()
plt.show()



## === cell 37
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.violinplot(y=df2["fare_amount"], x=df2["year"], ax=ax[0])
sns.violinplot(y=df2["fare_amount"], x=df2["month"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 38
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.barplot(y=df2["fare_amount"], x=df2["year"], ax=ax[0], palette="Set2")
sns.barplot(y=df2["fare_amount"], x=df2["month"], ax=ax[1], palette="Set2")
plt.tight_layout()
plt.show()



## === cell 39
pass



## === cell 40
pass



## === cell 41
X = df2.drop(columns=["fare_amount", "key", "pickup_datetime"])
y = df2["fare_amount"].astype(float)



## === cell 42
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=RANDOM_STATE)



## === cell 43
print(f"train: {X_train.shape}")
print(f"train target: {y_train.shape}")
print(f"val: {X_valid.shape}")
print(f"val target: {y_valid.shape}")



## === cell 44
X_train.head()



## === cell 45
ss = StandardScaler()
ss.fit(X_train)
X_train_ss = ss.transform(X_train)
X_valid_ss = ss.transform(X_valid)

ridge_tmp = Ridge(alpha=10)
ridge_tmp.fit(X_train_ss, y_train)
valid_pred = ridge_tmp.predict(X_valid_ss)
rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
print("Validation RMSE (Ridge alpha=10):", rmse)



## === cell 46

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    ElasticNet,
    HuberRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def get_models(models=dict()):
    models["lr"] = LinearRegression()
    models["lasso"] = Lasso()
    models["ridge"] = Ridge()
    models["en"] = ElasticNet()
    models["huber"] = HuberRegressor()
    models["pa"] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    return models


def get_models_nl(models=dict()):
    models["svr"] = SVR()
    n_trees = 100
    models["ada"] = AdaBoostRegressor(n_estimators=n_trees)
    models["bag"] = BaggingRegressor(n_estimators=n_trees)
    models["rf"] = RandomForestRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["et"] = ExtraTreesRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["gbm"] = GradientBoostingRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE
    )
    return models


def evaluate_models(models, X_train_ss, y_train, X_test_ss, y_test):
    for name, model in models.items():
        model_fit = model.fit(X_train_ss, y_train)
        train_preds = model_fit.predict(X_train_ss)
        test_preds = model_fit.predict(X_test_ss)
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        print(f"{name}:")
        print(f"----")
        print(f"Train MSE: {round(train_mse, 2)}")
        print(f"Test MSE: {round(test_mse, 2)}\n")


def params(model):
    if model == "lasso":
        return {"alpha": [0.01, 0.1, 1, 2, 5, 10]}
    elif model == "ridge":
        return {"alpha": [0.01, 0.1, 1, 2, 5, 10]}
    elif model == "en":
        return {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6]}
    elif model == "svr":
        return {
            "kernel": ["rbf", "linear", "poly"],
            "C": [1, 20, 50, 100],
            "gamma": ["scale", "auto"],
            "epsilon": [0.1, 1, 10],
        }
    elif model == "ada":
        return {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 1]}
    elif model == "bag":
        return {
            "n_estimators": [20, 50, 100, 150],
            "max_features": [2, 4, 6],
            "max_samples": [0.1, 0.2, 0.3, 0.5, 0.7],
            "bootstrap": [True],
        }
    elif model == "rf":
        return {
            "bootstrap": [True],
            "max_depth": [5, 10, 15],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4],
            "min_samples_split": [2, 3, 4],
            "n_estimators": [50, 200, 300],
            "random_state": [RANDOM_STATE],
        }
    elif model == "et":
        return {
            "bootstrap": [True],
            "max_depth": [5, 10, 15],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4],
            "min_samples_split": [2, 3, 4],
            "n_estimators": [50, 200, 300],
            "random_state": [RANDOM_STATE],
        }
    elif model == "gbm":
        return {
            "learning_rate": [0.1, 0.3, 0.6, 1],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [2, 3, 4],
            "max_depth": [3, 5, 8],
        }
    else:
        return {}


def grid_search_rs(model_name, models, X_train, y_train, X_test, y_test):
    pipe_params = params(model_name)
    model = models[model_name]
    if len(pipe_params) == 0:
        raise ValueError(f"No params defined for model '{model_name}'")
    gs = RandomizedSearchCV(
        model,
        param_distributions=pipe_params,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=True,
        n_jobs=8,
        random_state=RANDOM_STATE,
    )
    gs.fit(X_train, y_train)
    train_score = gs.score(X_train, y_train)
    test_score = gs.score(X_test, y_test)

    print(f"Results from: {model_name}")
    print(f"-----------------------------------")
    print(f"Best Hyperparameters: {gs.best_params_}")
    print(f"Mean MSE: {-round(gs.best_score_, 4)}")
    print(f"Train MSE: {-round(train_score, 4)}")
    print(f"Test MSE: {-round(test_score, 4)}")
    print(" ")
    return gs




## === cell 47
models = get_models()
evaluate_models(models, X_train_ss, y_train, X_valid_ss, y_valid)



## === cell 48
pass



## === cell 49
pass



## === cell 50
pass



## === cell 51
pass



## === cell 52
pass



## === cell 53
ss_full = StandardScaler()
ss_full.fit(X)
X_ss = ss_full.transform(X)

test_features = test_df.drop(columns=["key", "pickup_datetime"])
test_features = test_features[X.columns]
test_df_ss = ss_full.transform(test_features)



## === cell 54
print(f"Shape of X: {X_ss.shape}")
print(f"Shape of y: {y.shape}")
print("Test features shape:", test_df_ss.shape)



## === cell 55
ridge = Ridge(alpha=10)
ridge.fit(X_ss, y)
ridge_preds = ridge.predict(test_df_ss)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": ridge_preds})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())



## === cell 56
check = pd.read_csv("/kaggle/working/submission.csv")
print("Submission loaded back:", check.shape)
print(check.columns.tolist())
print(check.head())



## === cell 57
pass



## === cell 58
pass
