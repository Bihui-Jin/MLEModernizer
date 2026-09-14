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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

5.68499

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1052.78877) has done: 'Implemented fixes for image loading, map plotting, pandas grouping syntax, airport distance calculations, seaborn jointplot call, and ensured output directory exists. These changes resolve runtime errors, allow all cells to execute, and keep the linear model pipeline unchanged, which should bring the RMSE closer to the target score.'

# 9. Code solution

## === cell 0
df_train = pd.read_csv("../input/train.csv", nrows=500000)

df_train.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2479523729.py in <cell line: 0>()
----> 1 df_train = pd.read_csv("../input/train.csv", nrows=500000)
      2 
      3 df_train.head()
      4 

NameError: name 'pd' is not defined

## === cell 1
df_train.dtypes



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723727466.py in <cell line: 0>()
----> 1 df_train.dtypes
      2 

NameError: name 'df_train' is not defined

## === cell 2
df_train.describe()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1262383953.py in <cell line: 0>()
----> 1 df_train.describe()
      2 

NameError: name 'df_train' is not defined

## === cell 3
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >= 0]
print("New size: %d" % len(df_train))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2666475382.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(df_train))
      2 df_train = df_train[df_train.fare_amount >= 0]
      3 print("New size: %d" % len(df_train))
      4 

NameError: name 'df_train' is not defined

## === cell 4
df_train[df_train.fare_amount < 100].fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/720782975.py in <cell line: 0>()
----> 1 df_train[df_train.fare_amount < 100].fare_amount.hist(bins=100, figsize=(14, 3))
      2 plt.xlabel("fare $USD")
      3 plt.title("Histogram")
      4 

NameError: name 'df_train' is not defined

## === cell 5
print(df_train.isnull().sum())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4117061199.py in <cell line: 0>()
----> 1 print(df_train.isnull().sum())
      2 

NameError: name 'df_train' is not defined

## === cell 6
print("Old size: %d" % len(df_train))
df_train = df_train.dropna(how="any", axis="rows")
print("New size: %d" % len(df_train))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/546819525.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(df_train))
      2 df_train = df_train.dropna(how="any", axis="rows")
      3 print("New size: %d" % len(df_train))
      4 

NameError: name 'df_train' is not defined

## === cell 7
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2814542586.py in <cell line: 0>()
     15 
     16 
---> 17 print("Old size: %d" % len(df_train))
     18 df_train = df_train[select_within_boundingbox(df_train, BB)]
     19 print("New size: %d" % len(df_train))

NameError: name 'df_train' is not defined

## === cell 8
try:
    import PIL.Image as Image

    with urllib.request.urlopen(
        "https://aiblog.nl/download/nyc_-75_40_-73_41.5.png"
    ) as url:
        nyc_map = np.array(Image.open(BytesIO(url.read())))
except Exception:
    nyc_map = None


def plot_on_map(df, BB, nyc_map):
    """Scatter plot of pickup and dropoff points; background map shown if available."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, s=1, alpha=0.2, c="r")
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title("Pickup locations")
    if nyc_map is not None:
        axs[0].imshow(nyc_map, zorder=0, extent=[BB[0], BB[1], BB[2], BB[3]])
    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, s=1, alpha=0.2, c="r")
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title("Dropoff locations")
    if nyc_map is not None:
        axs[1].imshow(nyc_map, zorder=0, extent=[BB[0], BB[1], BB[2], BB[3]])




## === cell 9
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df_train["distance_km"] = distance(
    df_train.pickup_latitude,
    df_train.pickup_longitude,
    df_train.dropoff_latitude,
    df_train.dropoff_longitude,
)

df_train.distance_km.hist(bins=50, figsize=(12, 4))
plt.xlabel("distance km")
plt.title("Histogram")
df_train.distance_km.describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3801325581.py in <cell line: 0>()
     10 
     11 df_train["distance_km"] = distance(
---> 12     df_train.pickup_latitude,
     13     df_train.pickup_longitude,
     14     df_train.dropoff_latitude,

NameError: name 'df_train' is not defined

## === cell 10
df_train.groupby("passenger_count")[["distance_km", "fare_amount"]].mean()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1470609530.py in <cell line: 0>()
----> 1 df_train.groupby("passenger_count")[["distance_km", "fare_amount"]].mean()
      2 

NameError: name 'df_train' is not defined

## === cell 11
print(
    "Average $USD/KM : {:0.2f}".format(
        df_train.fare_amount.sum() / df_train.distance_km.sum()
    )
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2169890113.py in <cell line: 0>()
      1 print(
      2     "Average $USD/KM : {:0.2f}".format(
----> 3         df_train.fare_amount.sum() / df_train.distance_km.sum()
      4     )
      5 )

NameError: name 'df_train' is not defined

## === cell 12
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].scatter(df_train.distance_km, df_train.fare_amount, alpha=0.2)
axs[0].set_xlabel("distance km")
axs[0].set_ylabel("fare $USD")
axs[0].set_title("All data")

idx = (df_train.distance_km < 21) & (df_train.fare_amount < 100)
axs[1].scatter(df_train[idx].distance_km, df_train[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel("distance km")
axs[1].set_ylabel("fare $USD")
axs[1].set_title("Zoom in on distance < 20km, fare < $100")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1871186963.py in <cell line: 0>()
----> 1 fig, axs = plt.subplots(1, 2, figsize=(16, 6))
      2 axs[0].scatter(df_train.distance_km, df_train.fare_amount, alpha=0.2)
      3 axs[0].set_xlabel("distance km")
      4 axs[0].set_ylabel("fare $USD")
      5 axs[0].set_title("All data")

NameError: name 'plt' is not defined

## === cell 13
idx = df_train.distance_km >= 0.1
print("Old size: %d" % len(df_train))
df_train = df_train[idx]
print("New size: %d" % len(df_train))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3323121490.py in <cell line: 0>()
----> 1 idx = df_train.distance_km >= 0.1
      2 print("Old size: %d" % len(df_train))
      3 df_train = df_train[idx]
      4 print("New size: %d" % len(df_train))
      5 

NameError: name 'df_train' is not defined

## === cell 14
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)

print(
    "Distance JFK airport - NYC center = {} km".format(
        distance(jfk[1], jfk[0], nyc[1], nyc[0])
    )
)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
idx = distance(df_train.pickup_latitude, df_train.pickup_longitude, jfk[1], jfk[0]) < 3
df_train[idx].fare_amount.hist(bins=100, ax=axs[0])
axs[0].set_xlabel("fare")
axs[0].set_title("Histogram pickup location within 3km of JFK Airport")

idx = (
    distance(df_train.dropoff_latitude, df_train.dropoff_longitude, jfk[1], jfk[0]) < 3
)
df_train[idx].fare_amount.hist(bins=100, ax=axs[1])
axs[1].set_xlabel("fare")
axs[1].set_title("Histogram dropoff location within 3km of JFK Airport")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/925544410.py in <cell line: 0>()
      4 print(
      5     "Distance JFK airport - NYC center = {} km".format(
----> 6         distance(jfk[1], jfk[0], nyc[1], nyc[0])
      7     )
      8 )

/tmp/ipykernel_11/3801325581.py in distance(lat1, lon1, lat2, lon2)
      3     a = (
      4         0.5
----> 5         - np.cos((lat2 - lat1) * p) / 2
      6         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
      7     )

NameError: name 'np' is not defined

## === cell 15
df_train["hour"] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
df_train["year"] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
df_train["fare_per_km"] = df_train.fare_amount / df_train.distance_km



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3339683629.py in <cell line: 0>()
----> 1 df_train["hour"] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
      2 df_train["year"] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
      3 df_train["fare_per_km"] = df_train.fare_amount / df_train.distance_km
      4 

NameError: name 'df_train' is not defined

## === cell 16
df_train.fare_per_km.describe()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3954128189.py in <cell line: 0>()
----> 1 df_train.fare_per_km.describe()
      2 

NameError: name 'df_train' is not defined

## === cell 17
idx = (df_train.distance_km < 5) & (df_train.fare_amount < 100)
plt.scatter(df_train[idx].distance_km, df_train[idx].fare_per_km)
plt.xlabel("distance km")
plt.ylabel("fare per distance km")

theta = (12.0, 4.0)
x = np.linspace(0.1, 5, 100)
plt.plot(x, theta[0] / x + theta[1], "--", c="r", lw=2)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2085083908.py in <cell line: 0>()
----> 1 idx = (df_train.distance_km < 5) & (df_train.fare_amount < 100)
      2 plt.scatter(df_train[idx].distance_km, df_train[idx].fare_per_km)
      3 plt.xlabel("distance km")
      4 plt.ylabel("fare per distance km")
      5 

NameError: name 'df_train' is not defined

## === cell 18
df_train.pivot_table("fare_per_km", index="hour", columns="year").plot(figsize=(14, 6))
plt.ylabel("Fare $USD / KM")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1321495301.py in <cell line: 0>()
----> 1 df_train.pivot_table("fare_per_km", index="hour", columns="year").plot(figsize=(14, 6))
      2 plt.ylabel("Fare $USD / KM")
      3 

NameError: name 'df_train' is not defined

## === cell 19
df_train["distance_to_center"] = distance(
    nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2190510630.py in <cell line: 0>()
      1 df_train["distance_to_center"] = distance(
----> 2     nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude
      3 )
      4 

NameError: name 'df_train' is not defined

## === cell 20
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train.distance_to_center,
    df_train.distance_km,
    c=np.clip(df_train.fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx = (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824293536.py in <cell line: 0>()
----> 1 fig, axs = plt.subplots(1, 2, figsize=(16, 6))
      2 im = axs[0].scatter(
      3     df_train.distance_to_center,
      4     df_train.distance_km,
      5     c=np.clip(df_train.fare_amount, 0, 100),

NameError: name 'plt' is not defined

## === cell 21
df_train["pickup_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1446852524.py in <cell line: 0>()
      1 df_train["pickup_distance_to_jfk"] = distance(
----> 2     jfk[1], jfk[0], df_train.pickup_latitude, df_train.pickup_longitude
      3 )
      4 df_train["dropoff_distance_to_jfk"] = distance(
      5     jfk[1], jfk[0], df_train.dropoff_latitude, df_train.dropoff_longitude

NameError: name 'df_train' is not defined

## === cell 22
idx = ~((df_train.pickup_distance_to_jfk < 3) | (df_train.dropoff_distance_to_jfk < 3))

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx1].distance_to_center,
    df_train[idx1].distance_km,
    c=np.clip(df_train[idx1].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/874001197.py in <cell line: 0>()
----> 1 idx = ~((df_train.pickup_distance_to_jfk < 3) | (df_train.dropoff_distance_to_jfk < 3))
      2 
      3 fig, axs = plt.subplots(1, 2, figsize=(16, 6))
      4 im = axs[0].scatter(
      5     df_train[idx].distance_to_center,

NameError: name 'df_train' is not defined

## === cell 23
idx = (df_train.fare_amount > 80) & (df_train.distance_km < 40)
plot_on_map(df_train[idx], BB, nyc_map)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3827932667.py in <cell line: 0>()
----> 1 idx = (df_train.fare_amount > 80) & (df_train.distance_km < 40)
      2 plot_on_map(df_train[idx], BB, nyc_map)
      3 

NameError: name 'df_train' is not defined

## === cell 24
ewr = (-74.175, 40.69)  # EWR airport
df_train["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)

lgr = (-73.87, 40.77)  # LGA airport
df_train["pickup_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4217159977.py in <cell line: 0>()
      1 ewr = (-74.175, 40.69)  # EWR airport
      2 df_train["pickup_distance_to_ewr"] = distance(
----> 3     ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude
      4 )
      5 df_train["dropoff_distance_to_ewr"] = distance(

NameError: name 'df_train' is not defined

## === cell 25
idx = ~(
    (df_train.pickup_distance_to_jfk < 3)
    | (df_train.dropoff_distance_to_jfk < 3)
    | (df_train.pickup_distance_to_ewr < 3)
    | (df_train.dropoff_distance_to_ewr < 3)
    | (df_train.pickup_distance_to_lgr < 3)
    | (df_train.dropoff_distance_to_lgr < 3)
)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx1].distance_to_center,
    df_train[idx1].distance_km,
    c=np.clip(df_train[idx1].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3163008729.py in <cell line: 0>()
      1 idx = ~(
----> 2     (df_train.pickup_distance_to_jfk < 3)
      3     | (df_train.dropoff_distance_to_jfk < 3)
      4     | (df_train.pickup_distance_to_ewr < 3)
      5     | (df_train.dropoff_distance_to_ewr < 3)

NameError: name 'df_train' is not defined

## === cell 26
df_test = pd.read_csv("../input/test.csv")



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1596602274.py in <cell line: 0>()
----> 1 df_test = pd.read_csv("../input/test.csv")
      2 

NameError: name 'pd' is not defined

## === cell 27
plot_on_map(df_test, BB, nyc_map)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3845424442.py in <cell line: 0>()
----> 1 plot_on_map(df_test, BB, nyc_map)
      2 

NameError: name 'df_test' is not defined

## === cell 28
df_test.passenger_count.hist()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/848993872.py in <cell line: 0>()
----> 1 df_test.passenger_count.hist()
      2 

NameError: name 'df_test' is not defined

## === cell 29
df_test["distance_km"] = distance(
    df_test.pickup_latitude,
    df_test.pickup_longitude,
    df_test.dropoff_latitude,
    df_test.dropoff_longitude,
)
df_test["distance_to_center"] = distance(
    nyc[1], nyc[0], df_test.dropoff_latitude, df_test.dropoff_longitude
)
df_test["hour"] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
df_test["year"] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1082372138.py in <cell line: 0>()
      1 df_test["distance_km"] = distance(
----> 2     df_test.pickup_latitude,
      3     df_test.pickup_longitude,
      4     df_test.dropoff_latitude,
      5     df_test.dropoff_longitude,

NameError: name 'df_test' is not defined

## === cell 30
df_test[~select_within_boundingbox(df_test, BB)]



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2353242207.py in <cell line: 0>()
----> 1 df_test[~select_within_boundingbox(df_test, BB)]
      2 

NameError: name 'df_test' is not defined

## === cell 31
idx = (df_train.distance_to_center < 40) & (df_train.passenger_count != 0)
features = ["year", "hour", "distance_km", "passenger_count"]
X = df_train[idx][features].values
y = df_train[idx]["fare_amount"].values



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1375507211.py in <cell line: 0>()
----> 1 idx = (df_train.distance_to_center < 40) & (df_train.passenger_count != 0)
      2 features = ["year", "hour", "distance_km", "passenger_count"]
      3 X = df_train[idx][features].values
      4 y = df_train[idx]["fare_amount"].values
      5 

NameError: name 'df_train' is not defined

## === cell 32
X.shape, y.shape



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1256159962.py in <cell line: 0>()
----> 1 X.shape, y.shape
      2 

NameError: name 'X' is not defined

## === cell 33
from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_prediction_analysis(y, y_pred, figsize=(10, 4), title=""):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c="red")
    axs[0].set_xlabel("$y$")
    axs[0].set_ylabel("$\\hat{y}$")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title("rmse = {:.2f}, evs = {:.2f}".format(rmse, evs))

    axs[1].hist(y - y_pred, bins=50)
    avg = np.mean(y - y_pred)
    std = np.std(y - y_pred)
    axs[1].set_xlabel("$y - \\hat{y}$")
    axs[1].set_title(
        "Histogram prediction error, $\\mu$ = {:.2f}, $\\sigma$ = {:.2f}".format(
            avg, std
        )
    )

    if title != "":
        fig.suptitle(title)




## === cell 34
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824522028.py in <cell line: 0>()
      2 
      3 X_train, X_test, y_train, y_test = train_test_split(
----> 4     X, y, test_size=0.25, random_state=42
      5 )
      6 

NameError: name 'X' is not defined

## === cell 35
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline(
    (
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    )
)
model_lin.fit(X_train, y_train)

train_fare_max = y_train.max()

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title="Linear Model - Training set")

y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title="Linear Model - Test set")




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2248640565.py in <cell line: 0>()
      9     )
     10 )
---> 11 model_lin.fit(X_train, y_train)
     12 
     13 # Store the maximum fare observed in the training split for later clipping

NameError: name 'X_train' is not defined

## === cell 36
def plot_rmse_analysis(model, X, y, N=400, test_size=0.25, figsize=(10, 4), title=""):
    rmse_train, rmse_test = [], []
    for i in range(N):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        rmse_train.append(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_te, y_te_pred)))

    g = sns.jointplot(x=rmse_train, y=rmse_test, kind="scatter", height=5)
    g.set_axis_labels(
        "RMSE training ($\\mu$={:.2f})".format(np.mean(rmse_train)),
        "RMSE test ($\\mu$={:.2f})".format(np.mean(rmse_test)),
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{title} (N={N}, test_size={test_size:.2f})")




## === cell 37
plot_rmse_analysis(model_lin, X, y, title="Linear model")



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4057394266.py in <cell line: 0>()
----> 1 plot_rmse_analysis(model_lin, X, y, title="Linear model")
      2 

NameError: name 'X' is not defined

## === cell 38
XTEST = df_test[features].values



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2131295638.py in <cell line: 0>()
----> 1 XTEST = df_test[features].values
      2 

NameError: name 'df_test' is not defined

## === cell 39
os.makedirs("output", exist_ok=True)

y_pred_final = model_lin.predict(XTEST)

y_pred_final = np.clip(y_pred_final, 0, train_fare_max)

submission = pd.DataFrame(
    {"key": df_test.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2977557574.py in <cell line: 0>()
----> 1 os.makedirs("output", exist_ok=True)
      2 
      3 y_pred_final = model_lin.predict(XTEST)
      4 
      5 # Clip predictions to a realistic range using the max fare seen in training

NameError: name 'os' is not defined

## === cell 40
submission

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
