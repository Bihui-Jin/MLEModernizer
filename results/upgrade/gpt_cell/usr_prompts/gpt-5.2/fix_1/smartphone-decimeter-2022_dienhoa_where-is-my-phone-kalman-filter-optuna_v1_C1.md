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
joblib==1.5.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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
!pip install simdkalman 


## === cell 1
from tqdm.notebook import tqdm
from dataclasses import dataclass
from scipy.interpolate import InterpolatedUnivariateSpline
import glob
from joblib import Parallel, delayed
import random
import simdkalman
import optuna
from functools import partial
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)


## === cell 2
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY  = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


## === cell 3
@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)

@dataclass
class BLH:
    lat : np.array
    lng : np.array
    hgt : np.array


## === cell 4
def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2  = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a/b), r)
    B = np.arctan2(z + (e2_*b)*np.sin(t)**3, r - (e2*a)*np.cos(t)**3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2*np.sin(B)**2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)


## === cell 5
def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = np.sin(dlat/2)**2 + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng/2)**2
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist

def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df1['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df2['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


## === cell 6
def ecef_to_lat_lng(gnss_df, UnixTimeMillis):
    ecef_columns = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    columns = ['utcTimeMillis'] + ecef_columns
    ecef_df = (gnss_df.drop_duplicates(subset='utcTimeMillis')[columns]
               .dropna().reset_index(drop=True))
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh  = ECEF_to_BLH(ecef)

    TIME = ecef_df['utcTimeMillis'].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame({
        'UnixTimeMillis'   : UnixTimeMillis,
        'LatitudeDegrees'  : np.degrees(lat),
        'LongitudeDegrees' : np.degrees(lng),
    })


## === cell 7
def make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    
    state_transition =  np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], 
                             [0, 1, 0, T, 0, 0.5 * T ** 2], 
                             [0, 0, 1, 0, T, 0],
                             [0, 0, 0, 1, 0, T], 
                             [0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 1]])
    
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    
    observation_noise = obs_cov_mat
    
    process_noise = process_cov_mat

    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf


## === cell 8
def calc_score(pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])    
    return score


## === cell 9
INPUT_PATH = '../input/smartphone-decimeter-2022/'


## === cell 10
def apply_kf_smoothing(df, _kf):
    df_filter = df.copy()
    data = df_filter[['LatitudeDegrees','LongitudeDegrees']].to_numpy()
    data = data.reshape(1, len(data), 2)
    smoothed = _kf.smooth(data)
    df_filter['LatitudeDegrees'] = smoothed.states.mean[0, :, 0]
    df_filter['LongitudeDegrees'] = smoothed.states.mean[0, :, 1]
    return df_filter


## === cell 11
dirnames = sorted(glob.glob(f'{INPUT_PATH}/train/*/*'))


## === cell 12
process_cov_00 = 10**-6
process_cov_11 = 10**-6
process_cov_22 = 10**-6
process_cov_33 = 10**-6
process_cov_44 = 10**-6
process_cov_55 = 10**-6
process_cov_66 = 10**-6
process_cov_off_diag = 10**-9


## === cell 13
process_diag = [process_cov_00, process_cov_11, process_cov_22, process_cov_33, process_cov_44, process_cov_55]


## === cell 14
def make_cov_mat(process_diag, process_cov_off_diag):
    rank = len(process_diag)
    process_cov_mat = np.zeros((rank,rank))
    np.fill_diagonal(process_cov_mat, process_diag)
    process_cov_mat = process_cov_mat + process_cov_off_diag
    return process_cov_mat


## === cell 15
process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)


## === cell 16
obs_cov_00 = 10**-6
obs_cov_11 = 10**-6
obs_cov_off_diag = 10**-9


## === cell 17
obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)


## === cell 18
process_cov_mat


## === cell 19
obs_cov_mat


## === cell 20
T = 1


## === cell 21
kf0 = make_kalman_filter(T, process_cov_mat, obs_cov_mat)


## === cell 22
dirname = dirnames[0]
pred_dfs = []
drive, phone = dirname.split('/')[-2:]
tripID  = f'{drive}/{phone}'
gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
gt_df   = pd.read_csv(f'{dirname}/ground_truth.csv')
pred_df = ecef_to_lat_lng(gnss_df, gt_df['UnixTimeMillis'].to_numpy())
gnss_df = pd.read_csv(f'{dirnames[0]}/device_gnss.csv')
pred_dfs.append(pred_df)


## === cell 23
score = calc_score(pred_df, gt_df)


## === cell 24
score


## === cell 25
pred_df_filter = apply_kf_smoothing(pred_df, kf0)


## === cell 26
score = calc_score(pred_df_filter, gt_df)


## === cell 27
score


## === cell 28
nb_file = len(dirnames)


## === cell 29
train_dfs = [pd.read_csv(f'{dirname}/device_gnss.csv') for dirname in dirnames[:nb_file]]


## === cell 30
train_gts = [pd.read_csv(f'{dirname}/ground_truth.csv') for dirname in dirnames[:nb_file]]


## === cell 31
filter_fn = partial(apply_kf_smoothing, _kf=kf0)


## === cell 32
def all_score(train_dfs, train_gts, filter_fn=None):
    """ Calculate the score for list of df"""
    scores = []
    for gnss_df, gt_df in zip(train_dfs, train_gts):
        pred_df = ecef_to_lat_lng(gnss_df, gt_df['UnixTimeMillis'].to_numpy())
        if filter_fn:
            pred_df = filter_fn(pred_df)
        score = calc_score(pred_df, gt_df)
        scores.append(score)
    return scores


## === cell 33
def _make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    state_transition =  np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], 
                             [0, 1, 0, T, 0, 0.5 * T ** 2], 
                             [0, 0, 1, 0, T, 0],
                             [0, 0, 0, 1, 0, T], 
                             [0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 1]])
    
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    
    observation_noise = obs_cov_mat
    
    process_noise = process_cov_mat

    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf


## === cell 34
def make_kalman_filter(param):
    process_diag = [param["process_cov_00"], 
                    param["process_cov_11"], 
                    param["process_cov_22"], 
                    param["process_cov_33"], 
                    param["process_cov_44"], 
                    param["process_cov_55"]]
    process_cov_off_diag = param["process_cov_off_diag"]

    obs_diag = [param["obs_cov_00"],
                param["obs_cov_11"]]
    obs_cov_off_diag = [param["obs_cov_off_diag"]]

    T = param['T']

    process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)
    obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)

    _kf = _make_kalman_filter(T, process_cov_mat, obs_cov_mat)
    return _kf


## === cell 35
def objective(trial):
    
    param = {
        'process_cov_00': trial.suggest_float("process_cov_00", 1e-8, 1e-4, log=True),
        'process_cov_11': trial.suggest_float("process_cov_11", 1e-8, 1e-4, log=True),
        'process_cov_22': trial.suggest_float("process_cov_22", 1e-8, 1e-4, log=True),
        'process_cov_33': trial.suggest_float("process_cov_33", 1e-8, 1e-4, log=True),
        'process_cov_44': trial.suggest_float("process_cov_44", 1e-8, 1e-4, log=True),
        'process_cov_55': trial.suggest_float("process_cov_55", 1e-8, 1e-4, log=True),
        'process_cov_off_diag': trial.suggest_float("process_cov_off_diag", 1e-12, 1e-8, log=True),
        
        'obs_cov_00': trial.suggest_float("obs_cov_00", 1e-8, 1e-4, log=True),
        'obs_cov_11': trial.suggest_float("obs_cov_11", 1e-8, 1e-4, log=True),
        'obs_cov_off_diag': trial.suggest_float("obs_cov_off_diag", 1e-12, 1e-8, log=True),
        
        'T': trial.suggest_float("T", 0.6, 1.4),
    }
    
    process_diag = [param["process_cov_00"], 
                    param["process_cov_11"], 
                    param["process_cov_22"], 
                    param["process_cov_33"], 
                    param["process_cov_44"], 
                    param["process_cov_55"]]
    process_cov_off_diag = param["process_cov_off_diag"]
    
    obs_diag = [param["obs_cov_00"],
                param["obs_cov_11"]]
    obs_cov_off_diag = [param["obs_cov_off_diag"]]
    
    T = param['T']
    
    process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)
    obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)
    
    _kf = _make_kalman_filter(T, process_cov_mat, obs_cov_mat)
    filter_fn = partial(apply_kf_smoothing,  _kf=_kf)
    scores = all_score(train_dfs, train_gts, filter_fn=filter_fn)
    
    return np.mean(scores)


## === cell 36
study = optuna.create_study(direction='minimize')


## === cell 37
study.optimize(objective, n_trials=3)


## === cell 38
param = {'process_cov_00': 1.2020309620263925e-08,
 'process_cov_11': 2.5818076069527384e-08,
 'process_cov_22': 1.7418556333561943e-05,
 'process_cov_33': 8.783630087500364e-06,
 'process_cov_44': 3.190877235713106e-07,
 'process_cov_55': 1.2137227780694694e-07,
 'process_cov_off_diag': 2.1258126417177658e-09,
 'obs_cov_00': 2.5228974904239324e-07,
 'obs_cov_11': 1.2085757874535475e-05,
 'obs_cov_off_diag': 7.79887179271382e-09,
 'T': 0.7190536201746304}


## === cell 39
sample_df = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv')
pred_dfs  = []
for dirname in sorted(glob.glob(f'{INPUT_PATH}/test/*/*')):
    drive, phone = dirname.split('/')[-2:]
    tripID  = f'{drive}/{phone}'
    gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
    UnixTimeMillis = sample_df[sample_df['tripId'] == tripID]['UnixTimeMillis'].to_numpy()
    pred_df = ecef_to_lat_lng(gnss_df, UnixTimeMillis)
    _kf = make_kalman_filter(param)
    pred_df = apply_kf_smoothing(pred_df, _kf)
    pred_df.insert(0, 'tripId', tripID)
    pred_dfs.append(pred_df)
baseline_test_df = pd.concat(pred_dfs)
baseline_test_df.to_csv('baseline_test.csv', index=False)
baseline_test_df.to_csv('submission.csv', index=False)


## --- ERROR in cell 39, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1084891121.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m     [0mpred_df[0m [0;34m=[0m [0mecef_to_lat_lng[0m[0;34m([0m[0mgnss_df[0m[0;34m,[0m [0mUnixTimeMillis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0m_kf[0m [0;34m=[0m [0mmake_kalman_filter[0m[0;34m([0m[0mparam[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0mpred_df[0m [0;34m=[0m [0mapply_kf_smoothing[0m[0;34m([0m[0mpred_df[0m[0;34m,[0m [0m_kf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m     [0mpred_df[0m[0;34m.[0m[0minsert[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;34m'tripId'[0m[0;34m,[0m [0mtripID[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m     [0mpred_dfs[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mpred_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1880661167.py[0m in [0;36mapply_kf_smoothing[0;34m(df, _kf)[0m
[1;32m      3[0m     [0mdata[0m [0;34m=[0m [0mdf_filter[0m[0;34m[[0m[0;34m[[0m[0;34m'LatitudeDegrees'[0m[0;34m,[0m[0;34m'LongitudeDegrees'[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mdata[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m,[0m [0;36m2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0msmoothed[0m [0;34m=[0m [0m_kf[0m[0;34m.[0m[0msmooth[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m     [0mdf_filter[0m[0;34m[[0m[0;34m'LatitudeDegrees'[0m[0;34m][0m [0;34m=[0m [0msmoothed[0m[0;34m.[0m[0mstates[0m[0;34m.[0m[0mmean[0m[0;34m[[0m[0;36m0[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mdf_filter[0m[0;34m[[0m[0;34m'LongitudeDegrees'[0m[0;34m][0m [0;34m=[0m [0msmoothed[0m[0;34m.[0m[0mstates[0m[0;34m.[0m[0mmean[0m[0;34m[[0m[0;36m0[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/simdkalman/kalmanfilter.py[0m in [0;36msmooth[0;34m(self, data, initial_value, initial_covariance, observations, states, covariances, verbose)[0m
[1;32m    397[0m         """
[1;32m    398[0m [0;34m[0m[0m
[0;32m--> 399[0;31m         return self.compute(
[0m[1;32m    400[0m             [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    401[0m             [0;36m0[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/simdkalman/kalmanfilter.py[0m in [0;36mcompute[0;34m(self, data, n_test, initial_value, initial_covariance, smoothed, filtered, states, covariances, observations, likelihoods, gains, log_likelihood, verbose)[0m
[1;32m    590[0m                 [0mresult[0m[0;34m.[0m[0mpairwise_covariances[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mn_vars[0m[0;34m,[0m [0mn_measurements[0m[0;34m,[0m [0mn_states[0m[0;34m,[0m [0mn_states[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    591[0m [0;34m[0m[0m
[0;32m--> 592[0;31m             [0mms[0m [0;34m=[0m [0mfiltered_states[0m[0;34m.[0m[0mmean[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m-[0m[0;36m1[0m[0;34m,[0m[0;34m:[0m[0;34m][0m[0;34m[[0m[0;34m...[0m[0;34m,[0m[0mnp[0m[0;34m.[0m[0mnewaxis[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    593[0m             [0mPs[0m [0;34m=[0m [0mfiltered_states[0m[0;34m.[0m[0mcov[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m[0;34m-[0m[0;36m1[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    594[0m [0;34m[0m[0m

[0;31mIndexError[0m: index -1 is out of bounds for axis 1 with size 0
