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

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')


## === cell 1
df_train =  pd.read_csv('../input/train.csv', nrows = 500000)

df_train.head()


## === cell 2
df_train.dtypes


## === cell 3
df_train.describe()


## === cell 4
print('Old size: %d' % len(df_train))
df_train = df_train[df_train.fare_amount>=0]
print('New size: %d' % len(df_train))


## === cell 5
df_train[df_train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');


## === cell 6
print(df_train.isnull().sum())


## === cell 7
print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df_train))


## === cell 8
BB = (-75, -73, 40, 41.5)

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

print('Old size: %d' % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print('New size: %d' % len(df_train))


## === cell 9
import urllib.request
from PIL import Image

url = "https://aiblog.nl/download/nyc_-75_40_-73_41.5.png"

try:
    with urllib.request.urlopen(url) as resp:
        nyc_map = np.array(Image.open(resp))
except Exception:
    nyc_map = None


def plot_on_map(df, BB, nyc_map):
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=0.2, c="r")
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title("Pickup locations")
    if nyc_map is not None:
        axs[0].imshow(nyc_map, zorder=0, extent=[-75, -73, 40, 41.5])

    axs[1].scatter(
        df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=0.2, c="r"
    )
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title("Dropoff locations")
    if nyc_map is not None:
        axs[1].imshow(nyc_map, zorder=0, extent=[-75, -73, 40, 41.5])


plot_on_map(df_train, BB, nyc_map)


## === cell 10
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

df_train['distance_km'] = distance(df_train.pickup_latitude, df_train.pickup_longitude, \
                                   df_train.dropoff_latitude, df_train.dropoff_longitude)

df_train.distance_km.hist(bins=50, figsize=(12,4))
plt.xlabel('distance km')
plt.title('Histogram')
df_train.distance_km.describe()


## === cell 11
df_train.groupby("passenger_count")[["distance_km", "fare_amount"]].mean()


## === cell 12
print("Average $USD/KM : {:0.2f}".format(df_train.fare_amount.sum()/df_train.distance_km.sum()))


## === cell 13
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(df_train.distance_km, df_train.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance km')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')

idx = (df_train.distance_km < 21) & (df_train.fare_amount < 100)
axs[1].scatter(df_train[idx].distance_km, df_train[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance km')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 20km, fare < $100');


## === cell 14
idx = (df_train.distance_km >= 0.1)
print('Old size: %d' % len(df_train))
df_train = df_train[idx]
print('New size: %d' % len(df_train))


## === cell 15
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)

print("Distance JFK airport - NYC center = {} km".format(distance(jfk[1], jfk[0], nyc[1], nyc[0])))

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
idx = (distance(df_train.pickup_latitude, df_train.pickup_longitude, jfk[1], jfk[0]) < 3)
df_train[idx].fare_amount.hist(bins=100, ax=axs[0])
axs[0].set_xlabel('fare')
axs[0].set_title('Histogram pickup location within 3km of JKF Airport')

idx = (distance(df_train.dropoff_latitude, df_train.dropoff_longitude, jfk[1], jfk[0]) < 3)
df_train[idx].fare_amount.hist(bins=100, ax=axs[1])
axs[1].set_xlabel('fare')
axs[1].set_title('Histogram dropoff location within 3km of JKF Airport');


## === cell 16
df_train['hour'] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
df_train['year'] = df_train.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
df_train['fare_per_km'] = df_train.fare_amount / df_train.distance_km


## === cell 17
df_train.fare_per_km.describe()


## === cell 18
idx = (df_train.distance_km < 5) & (df_train.fare_amount < 100)
plt.scatter(df_train[idx].distance_km, df_train[idx].fare_per_km)
plt.xlabel('distance km')
plt.ylabel('fare per distance km')

theta = (12.0, 4.0)
x = np.linspace(0.1, 5, 100)
plt.plot(x, theta[0]/x + theta[1], '--', c='r', lw=2);


## === cell 19
df_train.pivot_table('fare_per_km', index='hour', columns='year').plot(figsize=(14,6))
plt.ylabel('Fare $USD / KM');


## === cell 20
df_train['distance_to_center'] = distance(nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude)


## === cell 21
fig, axs = plt.subplots(1, 2, figsize=(16,6))
im = axs[0].scatter(df_train.distance_to_center, df_train.distance_km, c=np.clip(df_train.fare_amount, 0, 100), 
                     cmap='viridis', alpha=1.0, s=1)
axs[0].set_xlabel('pickup distance from NYC center')
axs[0].set_ylabel('distance km')
axs[0].set_title('All data')
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel('fare_amount', rotation=270)

idx = (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(df_train[idx].distance_to_center, df_train[idx].distance_km, 
                     c=np.clip(df_train[idx].fare_amount, 0, 100), cmap='viridis', alpha=1.0, s=1)
axs[1].set_xlabel('pickup distance from NYC center')
axs[1].set_ylabel('distance km')
axs[1].set_title('Zoom in')
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel('fare_amount', rotation=270);


## === cell 22
df_train['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0], df_train.dropoff_latitude, df_train.dropoff_longitude)


## === cell 23
idx = ~((df_train.pickup_distance_to_jfk < 3) | (df_train.dropoff_distance_to_jfk < 3))

fig, axs = plt.subplots(1, 2, figsize=(16,6))
im = axs[0].scatter(df_train[idx].distance_to_center, df_train[idx].distance_km, 
                    c=np.clip(df_train[idx].fare_amount, 0, 100), 
                     cmap='viridis', alpha=1.0, s=1)
axs[0].set_xlabel('pickup distance from NYC center')
axs[0].set_ylabel('distance km')
axs[0].set_title('All data')
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel('fare_amount', rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(df_train[idx1].distance_to_center, df_train[idx1].distance_km, 
                     c=np.clip(df_train[idx1].fare_amount, 0, 100), cmap='viridis', alpha=1.0, s=1)
axs[1].set_xlabel('pickup distance from NYC center')
axs[1].set_ylabel('distance km')
axs[1].set_title('Zoom in')
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel('fare_amount', rotation=270);


## === cell 24
idx = (df_train.fare_amount>80) & (df_train.distance_km<40) 
plot_on_map(df_train[idx], BB, nyc_map)


## === cell 25
ewr = (-74.175, 40.69) # see https://www.travelmath.com/airport/EWR
df_train['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0], df_train.dropoff_latitude, df_train.dropoff_longitude)

lgr = (-73.87, 40.77) # see https://www.travelmath.com/airport/LGA
df_train['pickup_distance_to_lgr'] = distance(ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude)
df_train['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0], df_train.dropoff_latitude, df_train.dropoff_longitude)


## === cell 26
idx = ~((df_train.pickup_distance_to_jfk < 3) | (df_train.dropoff_distance_to_jfk < 3) |
        (df_train.pickup_distance_to_ewr < 3) | (df_train.dropoff_distance_to_ewr < 3) |
        (df_train.pickup_distance_to_lgr < 3) | (df_train.dropoff_distance_to_lgr < 3))

fig, axs = plt.subplots(1, 2, figsize=(16,6))
im = axs[0].scatter(df_train[idx].distance_to_center, df_train[idx].distance_km, 
                    c=np.clip(df_train[idx].fare_amount, 0, 100), 
                     cmap='viridis', alpha=1.0, s=1)
axs[0].set_xlabel('pickup distance from NYC center')
axs[0].set_ylabel('distance km')
axs[0].set_title('All data')
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel('fare_amount', rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(df_train[idx1].distance_to_center, df_train[idx1].distance_km, 
                     c=np.clip(df_train[idx1].fare_amount, 0, 100), cmap='viridis', alpha=1.0, s=1)
axs[1].set_xlabel('pickup distance from NYC center')
axs[1].set_ylabel('distance km')
axs[1].set_title('Zoom in')
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel('fare_amount', rotation=270);


## === cell 27
df_test =  pd.read_csv('../input/test.csv')


## === cell 28
plot_on_map(df_test, BB, nyc_map)


## === cell 29
df_test.passenger_count.hist();


## === cell 30
df_test['distance_km'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                  df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['distance_to_center'] = distance(nyc[1], nyc[0], \
                                          df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
df_test['year'] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)


## === cell 31
df_test[~select_within_boundingbox(df_test, BB)]


## === cell 32
idx = (df_train.distance_to_center<40) & (df_train.passenger_count!=0)
features = ['year', 'hour', 'distance_km', 'passenger_count']
X = df_train[idx][features].values
y = df_train[idx]['fare_amount'].values


## === cell 33
X.shape, y.shape


## === cell 34
from sklearn.metrics import mean_squared_error, explained_variance_score

def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))
    
    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))
    
    if title!='':
        fig.suptitle(title)


## === cell 35
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


## === cell 36
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title='Linear Model - Trainingset')

y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title='Linear Model - Testset')


## === cell 37
def plot_rmse_analysis(model, X, y, N=400, test_size=0.25, figsize=(10,4), title=''):
    rmse_train, rmse_test = [], []
    for i in range(N):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    g = sns.jointplot(np.array(rmse_train), np.array(rmse_test), kind='scatter', stat_func=None, size=5)
    g.set_axis_labels("RMSE training ($\mu$={:.2f})".format(np.mean(rmse_train)), 
                      "RMSE test ($\mu$={:.2f})".format(np.mean(rmse_test)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('{} (N={}, test_size={:0.2f})'.format(title, N, test_size))


## === cell 38
plot_rmse_analysis(model_lin, X, y, title="Linear model")


## --- ERROR in cell 38, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2482526278.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Fix: seaborn.jointplot API no longer accepts x/y as positional args in newer versions.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;31m# Using keyword arguments keeps identical plotting semantics and avoids the TypeError.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mplot_rmse_analysis[0m[0;34m([0m[0mmodel_lin[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mtitle[0m[0;34m=[0m[0;34m"Linear model"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2823372513.py[0m in [0;36mplot_rmse_analysis[0;34m(model, X, y, N, test_size, figsize, title)[0m
[1;32m     13[0m         [0mrmse_test[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0msqrt[0m[0;34m([0m[0mmean_squared_error[0m[0;34m([0m[0my_test[0m[0;34m,[0m [0my_test_pred[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m
[0;32m---> 15[0;31m     [0mg[0m [0;34m=[0m [0msns[0m[0;34m.[0m[0mjointplot[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mrmse_train[0m[0;34m)[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mrmse_test[0m[0;34m)[0m[0;34m,[0m [0mkind[0m[0;34m=[0m[0;34m'scatter'[0m[0;34m,[0m [0mstat_func[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0msize[0m[0;34m=[0m[0;36m5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m     g.set_axis_labels("RMSE training ($\mu$={:.2f})".format(np.mean(rmse_train)), 
[1;32m     17[0m                       "RMSE test ($\mu$={:.2f})".format(np.mean(rmse_test)))

[0;31mTypeError[0m: jointplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 1 keyword-only argument) were given

## === cell 39
XTEST = df_test[features].values
