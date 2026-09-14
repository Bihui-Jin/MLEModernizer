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

3.10

# 2. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (321 lines)
            metadata.zip (1.2 kB)
            sample_submission.csv (37088 lines)
            sample_submission.csv.zip (108.0 kB)
            test.zip (557.7 MB)
            train.zip (3.7 GB)
            metadata/
                accumulated_delta_range_state_bit_map.json (1 lines)
                constellation_type_mapping.csv (9 lines)
                ... and 1 other files
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
            test/
                2020-06-04-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (56087 lines)
                        device_imu.csv (340189 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (58761 lines)
                        device_imu.csv (342285 lines)
                        supplemental/
                            ... (max depth reached)
                2020-06-04-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (68061 lines)
                        device_imu.csv (338641 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68855 lines)
                        device_imu.csv (339610 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (73508 lines)
                        device_imu.csv (456999 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (77061 lines)
                        device_imu.csv (454150 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (64478 lines)
                        device_imu.csv (456044 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68307 lines)
                        device_imu.csv (449696 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (19537 lines)
                        device_imu.csv (221095 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (34594 lines)
                        device_imu.csv (222954 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (40323 lines)
                        device_imu.csv (216914 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-1/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (60277 lines)
                        device_imu.csv (344013 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (61077 lines)
                        device_imu.csv (235288 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-2/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (66015 lines)
                        device_imu.csv (371204 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (65501 lines)
                        device_imu.csv (257874 lines)
                        supplemental/
                            ... (max depth reached)
                2021-08-24-US-SVL-1/
                    GooglePixel4/
                        device_gnss.csv (101566 lines)
                        device_imu.csv (711980 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (112728 lines)
                        device_imu.csv (721330 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (122140 lines)
                        device_imu.csv (700392 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (133142 lines)
                        device_imu.csv (478300 lines)
                        supplemental/
                            ... (max depth reached)
                test/
            train/
                2020-05-15-US-MTV-1/
                    GooglePixel4XL/
                        device_gnss.csv (90154 lines)
                        device_imu.csv (734857 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                2020-05-21-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (61368 lines)
                        device_imu.csv (415251 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (64498 lines)
                        device_imu.csv (415486 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                ... and 53 other folders
        input/
            description.md (321 lines)
            metadata.zip (1.2 kB)
            sample_submission.csv (37088 lines)
            sample_submission.csv.zip (108.0 kB)
            test.zip (557.7 MB)
            train.zip (3.7 GB)
            metadata/
                accumulated_delta_range_state_bit_map.json (1 lines)
                constellation_type_mapping.csv (9 lines)
                ... and 1 other files
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
            test/
                2020-06-04-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (56087 lines)
                        device_imu.csv (340189 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (58761 lines)
                        device_imu.csv (342285 lines)
                        supplemental/
                            ... (max depth reached)
                2020-06-04-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (68061 lines)
                        device_imu.csv (338641 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68855 lines)
                        device_imu.csv (339610 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (73508 lines)
                        device_imu.csv (456999 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (77061 lines)
                        device_imu.csv (454150 lines)
                        supplemental/
                            ... (max depth reached)
                2020-07-08-US-MTV-2/
                    GooglePixel4/
                        device_gnss.csv (64478 lines)
                        device_imu.csv (456044 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (68307 lines)
                        device_imu.csv (449696 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-08-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (19537 lines)
                        device_imu.csv (221095 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (34594 lines)
                        device_imu.csv (222954 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (40323 lines)
                        device_imu.csv (216914 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-1/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (60277 lines)
                        device_imu.csv (344013 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (61077 lines)
                        device_imu.csv (235288 lines)
                        supplemental/
                            ... (max depth reached)
                2021-04-29-US-MTV-2/
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (66015 lines)
                        device_imu.csv (371204 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (65501 lines)
                        device_imu.csv (257874 lines)
                        supplemental/
                            ... (max depth reached)
                2021-08-24-US-SVL-1/
                    GooglePixel4/
                        device_gnss.csv (101566 lines)
                        device_imu.csv (711980 lines)
                        supplemental/
                            ... (max depth reached)
                    GooglePixel5/
                        device_gnss.csv (112728 lines)
                        device_imu.csv (721330 lines)
                        supplemental/
                            ... (max depth reached)
                    SamsungGalaxyS20Ultra/
                        device_gnss.csv (122140 lines)
                        device_imu.csv (700392 lines)
                        supplemental/
                            ... (max depth reached)
                    XiaomiMi8/
                        device_gnss.csv (133142 lines)
                        device_imu.csv (478300 lines)
                        supplemental/
                            ... (max depth reached)
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
            train/
                2020-05-15-US-MTV-1/
                    GooglePixel4XL/
                        device_gnss.csv (90154 lines)
                        device_imu.csv (734857 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                2020-05-21-US-MTV-1/
                    GooglePixel4/
                        device_gnss.csv (61368 lines)
                        device_imu.csv (415251 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                    GooglePixel4XL/
                        device_gnss.csv (64498 lines)
                        device_imu.csv (415486 lines)
                        ... and 1 other files
                        supplemental/
                            ... (max depth reached)
                ... and 53 other folders
        working/
            smartphone-decimeter-2022/
                description.md (321 lines)
                metadata.zip (1.2 kB)
                ... and 4 other files
                metadata/
                    accumulated_delta_range_state_bit_map.json (1 lines)
                    constellation_type_mapping.csv (9 lines)
                    ... and 1 other files
                smartphone-decimeter-2022/
                test/
                    2020-06-04-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-06-04-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-07-08-US-MTV-2/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    2021-04-08-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                    2021-04-29-US-MTV-1/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-04-29-US-MTV-2/
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    2021-08-24-US-SVL-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel5/
                            ... (max depth reached)
                        SamsungGalaxyS20Ultra/
                            ... (max depth reached)
                        XiaomiMi8/
                            ... (max depth reached)
                    test/
                train/
                    2020-05-15-US-MTV-1/
                        GooglePixel4XL/
                            ... (max depth reached)
                    2020-05-21-US-MTV-1/
                        GooglePixel4/
                            ... (max depth reached)
                        GooglePixel4XL/
                            ... (max depth reached)
                    ... and 53 other folders
```

-> data/metadata/accumulated_delta_range_state_bit_map.json has auto-generated json schema:
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

-> data/metadata/constellation_type_mapping.csv has 8 rows and 2 columns.
The columns are: constellationType, constellationName

-> data/metadata/raw_state_bit_map.json has auto-generated json schema:
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
    },
    "5": {
      "type": "string"
    },
    "6": {
      "type": "string"
    },
    "7": {
      "type": "string"
    },
    "8": {
      "type": "string"
    },
    "9": {
      "type": "string"
    },
    "10": {
      "type": "string"
    },
    "11": {
      "type": "string"
    },
    "12": {
      "type": "string"
    },
    "13": {
      "type": "string"
    },
    "14": {
      "type": "string"
    },
    "15": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ]
}

-> data/sample_submission.csv has 37087 rows and 4 columns.
The columns are: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

-> data/smartphone-decimeter-2022/metadata/accumulated_delta_range_state_bit_map.json has auto-generated json schema:
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

-> data/smartphone-decimeter-2022/metadata/constellation_type_mapping.csv has 8 rows and 2 columns.
The columns are: constellationType, constellationName

-> data/smartphone-decimeter-2022/metadata/raw_state_bit_map.json has auto-generated json schema:
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
    },
    "5": {
      "type": "string"
    },
    "6": {
      "type": "string"
    },
    "7": {
      "type": "string"
    },
    "8": {
      "type": "string"
    },
    "9": {
      "type": "string"
    },
    "10": {
      "type": "string"
    },
    "11": {
      "type": "string"
    },
    "12": {
      "type": "string"
    },
    "13": {
      "type": "string"
    },
    "14": {
      "type": "string"
    },
    "15": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ]
}

-> data/smartphone-decimeter-2022/sample_submission.csv has 37087 rows and 4 columns.
The columns are: tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold


## === cell 1
from contextlib import contextmanager
from time import time

class Timer:
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


## === cell 2
BASE_DIR = Path('../input/smartphone-decimeter-2022')
df_sample_submission = pd.read_csv(BASE_DIR / "sample_submission.csv")


## === cell 3
train_folders = glob(str(BASE_DIR / "train/*/*"))

X = []
y_LatitudeDegrees = []
y_LongitudeDegrees = []

for train_folder in tqdm(train_folders):
    df_device_gnss = pd.read_csv(f"{train_folder}/device_gnss.csv").rename(
        columns={"utcTimeMillis": "UnixTimeMillis"}
    )
    df_device_gnss = (
        df_device_gnss.groupby("UnixTimeMillis").mean(numeric_only=True).reset_index()
    )

    df_ground_truth = pd.read_csv(
        f"{train_folder}/ground_truth.csv",
        usecols=["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"],
    )

    df_merged = pd.merge(
        df_device_gnss, df_ground_truth, on="UnixTimeMillis", how="left"
    )

    X.append(df_merged.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]))
    y_LatitudeDegrees.append(df_merged["LatitudeDegrees"])
    y_LongitudeDegrees.append(df_merged["LongitudeDegrees"])

X = pd.concat(X).values
y_LatitudeDegrees = pd.concat(y_LatitudeDegrees).values
y_LongitudeDegrees = pd.concat(y_LongitudeDegrees).values


## === cell 4
test_folders = glob(str(BASE_DIR / "test/*/*"))

test_index = []
test_X = []
test_y_LatitudeDegrees = []
test_y_LongitudeDegrees = []

for test_folder in tqdm(test_folders):
    df_device_gnss = pd.read_csv( f"{test_folder}/device_gnss.csv").rename(columns={'utcTimeMillis': 'UnixTimeMillis'})
    df_device_gnss = df_device_gnss.groupby("UnixTimeMillis").mean()
    
    dir_name, device_name = os.path.split(test_folder)
    _, id = os.path.split(dir_name)
    
    df_sample_by_tripId = df_sample_submission[df_sample_submission["tripId"] == f"{id}/{device_name}"]
    df_merged = pd.merge(df_device_gnss, df_sample_by_tripId.drop(columns=["tripId"]), on="UnixTimeMillis", how="left")
    
    df_index = df_merged[["UnixTimeMillis"]]
    df_index["tripId"] = f"{id}/{device_name}"
    
    test_index.append(df_index)
    test_X.append(df_merged.drop(columns=['LatitudeDegrees', 'LongitudeDegrees']))
    test_y_LatitudeDegrees.append(df_merged['LatitudeDegrees'])
    test_y_LongitudeDegrees.append(df_merged['LongitudeDegrees'])
    
test_index = pd.concat(test_index).reset_index(drop=True)
test_X = pd.concat(test_X).values
test_y_LatitudeDegrees = pd.concat(test_y_LatitudeDegrees).values
test_y_LongitudeDegrees = pd.concat(test_y_LongitudeDegrees).values


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1941[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1942[0;31m             [0mres_values[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_grouper[0m[0;34m.[0m[0magg_series[0m[0;34m([0m[0mser[0m[0;34m,[0m [0malt[0m[0;34m,[0m [0mpreserve_dtype[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1943[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36magg_series[0;34m(self, obj, func, preserve_dtype)[0m
[1;32m    863[0m [0;34m[0m[0m
[0;32m--> 864[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_aggregate_series_pure_python[0m[0;34m([0m[0mobj[0m[0;34m,[0m [0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    865[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_aggregate_series_pure_python[0;34m(self, obj, func)[0m
[1;32m    884[0m         [0;32mfor[0m [0mi[0m[0;34m,[0m [0mgroup[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0msplitter[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 885[0;31m             [0mres[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mgroup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    886[0m             [0mres[0m [0;34m=[0m [0mextract_result[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2454[0;31m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2455[0m                 [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m   6548[0m     ):
[0;32m-> 6549[0;31m         [0;32mreturn[0m [0mNDFrame[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mself[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6550[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12419[0m     ) -> Series | float:
[0;32m> 12420[0;31m         return self._stat_function(
[0m[1;32m  12421[0m             [0;34m"mean"[0m[0;34m,[0m [0mnanops[0m[0;34m.[0m[0mnanmean[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_stat_function[0;34m(self, name, func, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12376[0m [0;34m[0m[0m
[0;32m> 12377[0;31m         return self._reduce(
[0m[1;32m  12378[0m             [0mfunc[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36m_reduce[0;34m(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)[0m
[1;32m   6456[0m                 )
[0;32m-> 6457[0;31m             [0;32mreturn[0m [0mop[0m[0;34m([0m[0mdelegate[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6458[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mf[0;34m(values, axis, skipna, **kwds)[0m
[1;32m    146[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 147[0;31m                 [0mresult[0m [0;34m=[0m [0malt[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    148[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnew_func[0;34m(values, axis, skipna, mask, **kwargs)[0m
[1;32m    403[0m [0;34m[0m[0m
[0;32m--> 404[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mmask[0m[0;34m=[0m[0mmask[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    405[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnanmean[0;34m(values, axis, skipna, mask)[0m
[1;32m    719[0m     [0mthe_sum[0m [0;34m=[0m [0mvalues[0m[0;34m.[0m[0msum[0m[0;34m([0m[0maxis[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 720[0;31m     [0mthe_sum[0m [0;34m=[0m [0m_ensure_numeric[0m[0;34m([0m[0mthe_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    721[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36m_ensure_numeric[0;34m(x)[0m
[1;32m   1700[0m             [0;31m# GH#44008, GH#36703 avoid casting e.g. strings to numeric[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1701[0;31m             [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34mf"Could not convert string '{x}' to numeric"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1702[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Could not convert string 'RawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRawRaw' to numeric

The above exception was the direct cause of the following exception:

[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/564959178.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m [0;32mfor[0m [0mtest_folder[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mtest_folders[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0mdf_device_gnss[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m [0;34mf"{test_folder}/device_gnss.csv"[0m[0;34m)[0m[0;34m.[0m[0mrename[0m[0;34m([0m[0mcolumns[0m[0;34m=[0m[0;34m{[0m[0;34m'utcTimeMillis'[0m[0;34m:[0m [0;34m'UnixTimeMillis'[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0mdf_device_gnss[0m [0;34m=[0m [0mdf_device_gnss[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m"UnixTimeMillis"[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;34m[0m[0m
[1;32m     12[0m     [0mdir_name[0m[0;34m,[0m [0mdevice_name[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mtest_folder[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36mmean[0;34m(self, numeric_only, engine, engine_kwargs)[0m
[1;32m   2450[0m             )
[1;32m   2451[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2452[0;31m             result = self._cython_agg_general(
[0m[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2454[0m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_cython_agg_general[0;34m(self, how, alt, numeric_only, min_count, **kwargs)[0m
[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m
[0;32m-> 1998[0;31m         [0mnew_mgr[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mgrouped_reduce[0m[0;34m([0m[0marray_func[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1999[0m         [0mres[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_wrap_agged_manager[0m[0;34m([0m[0mnew_mgr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2000[0m         [0;32mif[0m [0mhow[0m [0;32min[0m [0;34m[[0m[0;34m"idxmin"[0m[0;34m,[0m [0;34m"idxmax"[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mgrouped_reduce[0;34m(self, func)[0m
[1;32m   1467[0m                 [0;31m#  while others do not.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1468[0m                 [0;32mfor[0m [0msb[0m [0;32min[0m [0mblk[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1469[0;31m                     [0mapplied[0m [0;34m=[0m [0msb[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1470[0m                     [0mresult_blocks[0m [0;34m=[0m [0mextend_blocks[0m[0;34m([0m[0mapplied[0m[0;34m,[0m [0mresult_blocks[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1471[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py[0m in [0;36mapply[0;34m(self, func, **kwargs)[0m
[1;32m    391[0m         [0mone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    392[0m         """
[0;32m--> 393[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    394[0m [0;34m[0m[0m
[1;32m    395[0m         [0mresult[0m [0;34m=[0m [0mmaybe_coerce_values[0m[0;34m([0m[0mresult[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36marray_func[0;34m(values)[0m
[1;32m   1993[0m [0;34m[0m[0m
[1;32m   1994[0m             [0;32massert[0m [0malt[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1995[0;31m             [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_agg_py_fallback[0m[0;34m([0m[0mhow[0m[0;34m,[0m [0mvalues[0m[0;34m,[0m [0mndim[0m[0;34m=[0m[0mdata[0m[0;34m.[0m[0mndim[0m[0;34m,[0m [0malt[0m[0;34m=[0m[0malt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1944[0m             [0mmsg[0m [0;34m=[0m [0;34mf"agg function failed [how->{how},dtype->{ser.dtype}]"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1945[0m             [0;31m# preserve the kind of exception that raised[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1946[0;31m             [0;32mraise[0m [0mtype[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1947[0m [0;34m[0m[0m
[1;32m   1948[0m         [0;32mif[0m [0mser[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mobject[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: agg function failed [how->mean,dtype->object]

## === cell 5
fold = KFold(n_splits=5, shuffle=True, random_state=510)
cv = fold.split(X, y_LatitudeDegrees)
cv = list(cv)
