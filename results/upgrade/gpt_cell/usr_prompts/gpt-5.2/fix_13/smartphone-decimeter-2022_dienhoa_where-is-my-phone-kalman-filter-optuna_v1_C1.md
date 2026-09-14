# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Compute smartphones location based on raw location measurements from Android smartphones collected in opensky and light urban roads.

## Metric
Submissions are scored on the mean of the 50th and 95th percentile distance errors. For every `phone` and once per second, the horizontal distance (in meters) is computed between the predicted latitude/longitude and the ground truth latitude/longitude. These distance errors form a distribution from which the 50th and 95th percentile errors are calculated (i.e. the 95th percentile error is the value, in meters, for which 95% of the distance errors are smaller). The 50th and 95th percentile errors are then averaged for each phone. Lastly, the mean of these averaged values is calculated across all phones in the test set.

## Submission Format
For each `phone` and `UnixTimeMillis` in the sample submission, you must predict the latitude and longitude. The sample submission typically requires a prediction once per second but may include larger gaps if there were too few valid GNSS signals. The submission file should contain a header and have the following format:

```
phone,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
2020-05-15-US-MTV-1_Pixel4,1273608785432,37.904611315634504,-86.48107806249548
2020-05-15-US-MTV-1_Pixel4,1273608786432,37.904611315634504,-86.48107806249548
2020-05-15-US-MTV-1_Pixel4,1273608787432,37.904611315634504,-86.48107806249548
```

## Dataset
**[train/test]/[drive_id]/[phone_name]/supplemental/[phone_name][.20o/.21o/.22o/.nmea]** - Equivalent data to the gnss logs in other formats used by the GPS community.

**train/[drive_id]/[phone_name]/ground_truth.csv** - Reference locations at expected timestamps.

- `MessageType` - "Fix", the prefix of sentence.

- `Provider` - "GT", short for ground truth.

- `[Latitude/Longitude]Degrees` - The [WGS84](https://en.wikipedia.org/w/index.php?title=World_Geodetic_System&oldid=1013033380) latitude, longitude (in decimal degrees) estimated by the reference GNSS receiver (NovAtel SPAN). When extracting from the NMEA file, linear interpolation has been applied to align the location to the expected non-integer timestamps.

- `AltitudeMeters` - The height above the WGS84 ellipsoid (in meters) estimated by the reference GNSS receiver.

- `SpeedMps`* - The speed over ground in meters per second.

- `AccuracyMeters` - The estimated horizontal accuracy radius in meters of this location at the 68th percentile confidence level. This means that there is a 68% chance that the true location of the device is within a distance of this uncertainty of the reported location.

- `BearingDegrees` - Bearing is measured in degrees clockwise from north. It ranges from 0 to 359.999 degrees.

- `UnixTimeMillis` - An integer number of milliseconds since the GPS epoch (1970/1/1 midnight UTC). Converted from [GnssClock](https://developer.android.com/reference/android/location/GnssClock).

**[train/test]/[drive_id]/[phone_name]/device_gnss.csv** - Each row contains raw GNSS measurements, derived values, and a baseline estimated location.. This baseline was computed using correctedPrM and the satellite positions, using a standard Weighted Least Squares (WLS) solver, with the phone's position (x, y, z), clock bias (t), and isrbM for each unique signal type as states for each epoch. Some of the raw measurement fields are not included in this file because they are deprecated or are not populated in the original gnss_log.txt.

- `MessageType` - "Raw", the prefix of sentence.

- `utcTimeMillis` - Milliseconds since UTC epoch (1970/1/1), converted from GnssClock.

- `TimeNanos` - The GNSS receiver internal hardware clock value in nanoseconds.

- `LeapSecond` - The leap second associated with the clock's time.

- `FullBiasNanos` - The difference between hardware clock (getTimeNanos()) inside GPS receiver and the true GPS time since 0000Z, January 6, 1980, in nanoseconds.

- `BiasNanos` - The clock's sub-nanosecond bias.

- `BiasUncertaintyNanos` - The clock's bias uncertainty (1-sigma) in nanoseconds.

- `DriftNanosPerSecond` - The clock's drift in nanoseconds per second.

- `DriftUncertaintyNanosPerSecond` - The clock's drift uncertainty (1-sigma) in nanoseconds per second.

- `HardwareClockDiscontinuityCount` - Count of hardware clock discontinuities.

- `Svid` - The satellite ID.

- `TimeOffsetNanos` - The time offset at which the measurement was taken in nanoseconds.

- `State` - Integer signifying sync state of the satellite. Each bit in the integer attributes to a particular state information of the measurement. See the **metadata/raw_state_bit_map.json** file for the mapping between bits and states.

- `ReceivedSvTimeNanos` - The received GNSS satellite time, at the measurement time, in nanoseconds.

- `ReceivedSvTimeUncertaintyNanos` - The error estimate (1-sigma) for the received GNSS time, in nanoseconds.

- `Cn0DbHz` - The carrier-to-noise density in dB-Hz.

- `PseudorangeRateMetersPerSecond` - The pseudorange rate at the timestamp in m/s.

- `PseudorangeRateUncertaintyMetersPerSecond` - The pseudorange's rate uncertainty (1-sigma) in m/s.

- `AccumulatedDeltaRangeState` - This indicates the state of the 'Accumulated Delta Range' measurement. Each bit in the integer attributes to state of the measurement. See the **metadata/accumulated_delta_range_state_bit_map.json** file for the mapping between bits and states.

- `AccumulatedDeltaRangeMeters` - The accumulated delta range since the last channel reset, in meters.

- `AccumulatedDeltaRangeUncertaintyMeters` - The accumulated delta range's uncertainty (1-sigma) in meters.

- `CarrierFrequencyHz` - The carrier frequency of the tracked signal.

- `MultipathIndicator` - A value indicating the 'multipath' state of the event.

- `ConstellationType` - GNSS constellation type. The mapping to human readable values is provided in the **metadata/constellation_type_mapping.csv** file.

- `CodeType` - The GNSS measurement's code type. Only available in recent logs.

- `ChipsetElapsedRealtimeNanos` - The elapsed real-time of this clock since system boot, in nanoseconds. Only available in recent logs.

- `ArrivalTimeNanosSinceGpsEpoch` - An integer number of nanoseconds since the GPS epoch (1980/1/6 midnight UTC). Its value equals round((Raw::TimeNanos - Raw::FullBiasNanos), for each unique epoch described in the Raw sentences.

- `RawPseudorangeMeters` - Raw pseudorange in meters. It is the product between the speed of light and the time difference from the signal transmission time (receivedSvTimeInGpsNanos) to the signal arrival time (Raw::TimeNanos - Raw::FullBiasNanos - Raw;;BiasNanos). Its uncertainty can be approximated by the product between the speed of light and the ReceivedSvTimeUncertaintyNanos.

- `SignalType` - The GNSS signal type is a combination of the constellation name and the frequency band. Common signal types measured by smartphones include GPS_L1, GPS_L5, GAL_E1, GAL_E5A, GLO_G1, BDS_B1I, BDS_B1C, BDS_B2A, QZS_J1, and QZS_J5.

- `ReceivedSvTimeNanosSinceGpsEpoch` - The signal transmission time received by the chipset, in the numbers of nanoseconds since the GPS epoch. Converted from ReceivedSvTimeNanos, this derived value is in a unified time scale for all constellations, while ReceivedSvTimeNanos refers to the time of day for GLONASS and the time of week for non-GLONASS constellations.

- `SvPosition[X/Y/Z]EcefMeters` - The satellite position (meters) in an ECEF coordinate frame at best estimate of "true signal transmission time" defined as ttx = receivedSvTimeInGpsNanos - satClkBiasNanos (defined below). They are computed with the satellite broadcast ephemeris, and have ~1-meter error with respect to the true satellite position.

- `Sv[Elevation/Azimuth]Degrees` - The elevation and azimuth in degrees of the satellite. They are computed using the WLS estimated user position.

- `SvVelocity[X/Y/Z]EcefMetersPerSecond` - The satellite velocity (meters per second) in an ECEF coordinate frame at best estimate of "true signal transmission time" ttx. They are computed with the satellite broadcast ephemeris, with this algorithm.

- `SvClockBiasMeters` - The satellite time correction combined with the satellite hardware delay in meters at the signal transmission time (receivedSvTimeInGpsNanos). Its time equivalent is termed as satClkBiasNanos. satClkBiasNanos equals the satelliteTimeCorrection minus the satelliteHardwareDelay. As defined in IS-GPS-200H Section 20.3.3.3.3.1, satelliteTimeCorrection is calculated from ∆tsv = af0 + af1(t - toc) + af2(t - toc)2 + ∆tr, while satelliteHardwareDelay is defined in Section 20.3.3.3.3.2. Parameters in the equations above are provided on the satellite broadcast ephemeris.

- `SvClockDriftMetersPerSecond` - The satellite clock drift in meters per second at the signal transmission time (receivedSvTimeInGpsNanos). It equals the difference of the satellite clock biases at t+0.5s and t-0.5s.

- `IsrbMeters` - The Inter-Signal Range Bias (ISRB) in meters from a non-GPS-L1 signal to GPS-L1 signals. For example, when the isrbM of GPS L5 is 1000m, it implies that a GPS L5 pseudorange is 1000m longer than the GPS L1 pseudorange transmitted by the same GPS satellite. It's zero for GPS-L1 signals. ISRB is introduced in the GPS chipset level and estimated as a state in the Weighted Least Squares engine.

- `IonosphericDelayMeters` - The ionospheric delay in meters, estimated with the Klobuchar model.

- `TroposphericDelayMeters` - The tropospheric delay in meters, estimated with the EGNOS model by Nigel Penna, Alan Dodson and W. Chen (2001).

- `WlsPositionXEcefMeters` - WlsPositionYEcefMeters,WlsPositionZEcefMeters: User positions in ECEF estimated by a Weighted-Least-Square (WLS) solver.

**[train/test]/[drive_id]/[phone_name]/device_imu.csv** - Readings the phone's accelerometer, gyroscope, and magnetometer.

- `MessageType` - which of the three instruments the row's data is from.

- `utcTimeMillis` - The sum of `elapsedRealtimeNanos` below and the estimated device boot time at UTC, after a recent NTP (Network Time Protocol) sync.

- `Measurement[X/Y/Z]` - [x/y/z]_uncalib without bias compensation.

- `Bias[X/Y/Z]MicroT` - Estimated [x/y/z]_bias. Null in datasets collected in earlier dates.

**[train/test]/[drive_id]/[phone_name]/supplemental/rinex.o** A text file of GNSS measurements on , collected from Android APIs (same as the "Raw" messages above), then converted to the [RINEX v3.03 format](http://rtcm.info/RINEX_3.04.IGS.RTCM_Final.pdf). refers to the last two digits of the year. During the conversion, the following treatments are taken to comply with the RINEX format. Essentially, this file contains a subset of information in the _GnssLog.txt.

1. The epoch time (in GPS time scale as per RINEX standard) is computed from [TimeNanos - (FullBiasNanos + BiasNanos)](https://developer.android.com/reference/android/location/GnssClock#getFullBiasNanos()), and then floored to the 100-nanosecond level to meet the precision requirement of the RINEX epoch time. The sub-100-nanosecond part has been wiped out by subtracting the raw pseudorange by the distance equivalent, and by subtracting the carrier phase range by the product of pseudorange rate and the sub-100-nanosecond part.
2. An epoch of measurements will not be converted to RINEX, if any of the following conditions occurs: a) the BiasUncertaintyNanos is larger or equal than 1E6, b) GnssClock values are invalid, e.g. FullBiasNanos is not a meaningful number.
3. Pseudorange value will not be converted if any of the following conditions occurs:

- [STATE_CODE_LOCK](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_CODE_LOCK) (for non-GAL E1) or [STATE_GAL_E1BC_CODE_LOCK](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GAL_E1BC_CODE_LOCK) (for GAL E1) is not set,
- [STATE_TOW_DECODED](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_TOW_DECODED) and [STATE_TOW_KNOWN](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_TOW_KNOWN) are not set for non-GLO signals,
- [STATE_GLO_TOD_DECODED](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GLO_TOD_DECODED) and [STATE_GLO_TOD_KNOWN](https://developer.android.com/reference/android/location/GnssMeasurement#STATE_GLO_TOD_KNOWN) are not set for GLO signals,
- CN0 is less than 20 dB-Hz,

1. ReceivedSvTimeUncertaintyNanos is larger than 500 nanoseconds, Carrier frequency is out of nominal range of each band. Loss of lock indicator (LLI) is set to 1 if [ADR_STATE_CYCLE_SLIP](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_CYCLE_SLIP) is set, or 2 if [ADR_STATE_HALF_CYCLE_REPORTED](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_HALF_CYCLE_REPORTED) is set and [ADR_STATE_HALF_CYCLE_RESOLVED](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_HALF_CYCLE_RESOLVED) is not set, or blank if [ADR_STATE_VALID](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_VALID) is not set or [ADR_STATE_RESET](https://developer.android.com/reference/android/location/GnssMeasurement#ADR_STATE_RESET) is set, or 0 otherwise.

**[train/test]/[drive_id]/[phone_name]/supplemental/gnss_log.txt** - The phone's logs as generated by the [GnssLogger App](https://play.google.com/store/apps/details?id=com.google.android.apps.location.gps.gnsslogger&hl=en_US&gl=US). [This notebook](https://www.kaggle.com/sohier/loading-gnss-logs/) demonstrates how to parse the logs. Each gnss file contains several sub-datasets, each of which is detailed below:

Raw - The raw GNSS measurements of one GNSS signal (each satellite may have 1-2 signals for L5-enabled smartphones), collected from the Android API [GnssMeasurement](https://developer.android.com/reference/android/location/GnssMeasurement).

- `utcTimeMillis` - Milliseconds since UTC epoch (1970/1/1), converted from [GnssClock](https://developer.android.com/reference/android/location/GnssClock)

- [`TimeNanos`](https://developer.android.com/reference/android/location/GnssClock#getTimeNanos()) - The GNSS receiver internal hardware clock value in nanoseconds.

- [`LeapSecond`](https://developer.android.com/reference/android/location/GnssClock#getLeapSecond()) - The leap second associated with the clock's time.

- [`TimeUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssClock#getTimeUncertaintyNanos()) - The clock's time uncertainty (1-sigma) in nanoseconds.

- [`FullBiasNanos`](https://developer.android.com/reference/android/location/GnssClock#getFullBiasNanos()) - The difference between hardware clock [getTimeNanos()](https://developer.android.com/reference/android/location/GnssClock#getTimeNanos()) inside GPS receiver and the true GPS time since 0000Z, January 6, 1980, in nanoseconds.

- [`BiasNanos`](https://developer.android.com/reference/android/location/GnssClock#getBiasNanos()) - The clock's sub-nanosecond bias.

- [`BiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssClock#getBiasUncertaintyNanos()) - The clock's bias uncertainty (1-sigma) in nanoseconds.

- [`DriftNanosPerSecond`](https://developer.android.com/reference/android/location/GnssClock#getDriftNanosPerSecond()) - The clock's drift in nanoseconds per second.

- [`DriftUncertaintyNanosPerSecond`](https://developer.android.com/reference/android/location/GnssClock#getDriftUncertaintyNanosPerSecond()) - The clock's drift uncertainty (1-sigma) in nanoseconds per second.

- [`HardwareClockDiscontinuityCount`](https://developer.android.com/reference/android/location/GnssClock#getHardwareClockDiscontinuityCount()) - Count of hardware clock discontinuities.

- [`Svid`](https://developer.android.com/reference/android/location/GnssMeasurement#getSvid()) - The satellite ID. More info can be found [here](https://developer.android.com/reference/android/location/GnssMeasurement#getSvid()).

- [`TimeOffsetNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getTimeOffsetNanos()) - The time offset at which the measurement was taken in nanoseconds.

- [`State`](https://developer.android.com/reference/android/location/GnssMeasurement#getState()) - Integer signifying sync state of the satellite. Each bit in the integer attributes to a particular state information of the measurement. See the **metadata/raw_state_bit_map.json** file for the mapping between bits and states.

- [`ReceivedSvTimeNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getReceivedSvTimeNanos()) - The received GNSS satellite time, at the measurement time, in nanoseconds.

- [`ReceivedSvTimeUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getReceivedSvTimeUncertaintyNanos()) - The error estimate (1-sigma) for the received GNSS time, in nanoseconds.

- [`Cn0DbHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getCn0DbHz()) - The carrier-to-noise density in dB-Hz.

- [`PseudorangeRateMetersPerSecond`](https://developer.android.com/reference/android/location/GnssMeasurement#getPseudorangeRateMetersPerSecond()) - The pseudorange rate at the timestamp in m/s.

- [`PseudorangeRateUncertaintyMetersPerSecond`](https://developer.android.com/reference/android/location/GnssMeasurement#getPseudorangeRateUncertaintyMetersPerSecond()) - The pseudorange's rate uncertainty (1-sigma) in m/s.

- [`AccumulatedDeltaRangeState`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeState()) - This indicates the state of the 'Accumulated Delta Range' measurement. Each bit in the integer attributes to state of the measurement. See the **metadata/accumulated_delta_range_state_bit_map.json** file for the mapping between bits and states.

- [`AccumulatedDeltaRangeMeters`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeMeters()) - The accumulated delta range since the last channel reset, in meters.

- [`AccumulatedDeltaRangeUncertaintyMeters`](https://developer.android.com/reference/android/location/GnssMeasurement#getAccumulatedDeltaRangeUncertaintyMeters()) - The accumulated delta range's uncertainty (1-sigma) in meters.

- [`CarrierFrequencyHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierFrequencyHz()) - The carrier frequency of the tracked signal.

- [`CarrierCycles`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierCycles()) - The number of full carrier cycles between the satellite and the receiver. Null in these datasets.

- [`CarrierPhase`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierPhase()) - The RF phase detected by the receiver. Null in these datasets.

- [`CarrierPhaseUncertainty`](https://developer.android.com/reference/android/location/GnssMeasurement#getCarrierPhaseUncertainty()) - The carrier-phase's uncertainty (1-sigma). Null in these datasets.

- [`MultipathIndicator`](https://developer.android.com/reference/android/location/GnssMeasurement#getMultipathIndicator()) - A value indicating the 'multipath' state of the event.

- [`SnrInDb`](https://developer.android.com/reference/android/location/GnssMeasurement#getSnrInDb()) - The (post-correlation & integration) Signal-to-Noise ratio (SNR) in dB.

- [`ConstellationType`](https://developer.android.com/reference/android/location/GnssMeasurement#getConstellationType()) - GNSS constellation type. It's an integer number, whose mapping to string value is provided in the constellation_type_mapping.csv file.

- [`AgcDb`](https://developer.android.com/reference/android/location/GnssMeasurement#getAutomaticGainControlLevelDb()) - The Automatic Gain Control level in dB.

- [`BasebandCn0DbHz`](https://developer.android.com/reference/android/location/GnssMeasurement#getBasebandCn0DbHz()) - The baseband carrier-to-noise density in dB-Hz. Only available in Android 11.

- [`FullInterSignalBiasNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getFullInterSignalBiasNanos()) - The GNSS measurement's inter-signal bias in nanoseconds with sub-nanosecond accuracy. Only available in Pixel 5 logs in 2021. Only available in Android 11.

- [`FullInterSignalBiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getFullInterSignalBiasUncertaintyNanos()) - The GNSS measurement's inter-signal bias uncertainty (1 sigma) in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`SatelliteInterSignalBiasNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getSatelliteInterSignalBiasNanos()) - The GNSS measurement's satellite inter-signal bias in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`SatelliteInterSignalBiasUncertaintyNanos`](https://developer.android.com/reference/android/location/GnssMeasurement#getSatelliteInterSignalBiasUncertaintyNanos()) - The GNSS measurement's satellite inter-signal bias uncertainty (1 sigma) in nanoseconds with sub-nanosecond accuracy. Only available in Android 11.

- [`CodeType`](https://developer.android.com/reference/android/location/GnssMeasurement#getCodeType()) - The GNSS measurement's code type. Only available in recent logs.

- [`ChipsetElapsedRealtimeNanos`](https://developer.android.com/reference/android/location/GnssClock#getElapsedRealtimeNanos()) - The elapsed real-time of this clock since system boot, in nanoseconds. Only available in recent logs.

Status - The status of a GNSS signal, as collected from the Android API [GnssStatus](https://developer.android.com/reference/android/location/GnssStatus.Callback).

- `UnixTimeMillis` - Milliseconds since UTC epoch (1970/1/1), reported from the last location changed by [GPS](https://developer.android.com/reference/android/location/LocationManager#GPS_PROVIDER) provider.

- [`SignalCount`](https://developer.android.com/reference/android/location/GnssStatus#getSatelliteCount()) - The total number of satellites in the satellite list.

- `SignalIndex` - The index of current signal.

- [`ConstellationType`](https://developer.android.com/reference/android/location/GnssStatus#getConstellationType(int)): The constellation type of the satellite at the specified index.

- [`Svid`](https://developer.android.com/reference/android/location/GnssStatus#getSvid(int)): The satellite ID.

- [`CarrierFrequencyHz`](https://developer.android.com/reference/android/location/GnssStatus#getCarrierFrequencyHz(int)): The carrier frequency of the signal tracked.

- [`Cn0DbHz`](https://developer.android.com/reference/android/location/GnssStatus#getCn0DbHz(int)): The carrier-to-noise density at the antenna of the satellite at the specified index in dB-Hz.

- [`AzimuthDegrees`](https://developer.android.com/reference/android/location/GnssStatus#getAzimuthDegrees(int)): The azimuth the satellite at the specified index.

- [`ElevationDegrees`](https://developer.android.com/reference/android/location/GnssStatus#getElevationDegrees(int)): The elevation of the satellite at the specified index.

- [`UsedInFix`](https://developer.android.com/reference/android/location/GnssStatus#usedInFix(int)): Whether the satellite at the specified index was used in the calculation of the most recent position fix.

- [`HasAlmanacData`](https://developer.android.com/reference/android/location/GnssStatus#hasAlmanacData(int)): Whether the satellite at the specified index has almanac data.

- [`HasEphemerisData`](https://developer.android.com/reference/android/location/GnssStatus#hasEphemerisData(int)): Whether the satellite at the specified index has ephemeris data.

- [`BasebandCn0DbHz`](https://developer.android.com/reference/android/location/GnssStatus#getBasebandCn0DbHz(int)): The baseband carrier-to-noise density of the satellite at the specified index in dB-Hz.

OrientationDeg - Each row represents an estimated device orientation, collected from Android API [SensorManager#getOrientation](https://developer.android.com/reference/android/hardware/SensorManager#getOrientation(float%5B%5D,%20float%5B%5D)).This message is only available in logs collected since March 2021.

- `utcTimeMillis` - The sum of `elapsedRealtimeNanos` below and the estimated device boot time at UTC, after a recent NTP (Network Time Protocol) sync.

- [`elapsedRealtimeNanos`](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp) - The time in nanoseconds at which the event happened.

- `yawDeg` - If the screen is in portrait mode, this value equals the Azimuth degree (modulus to 0°~360°). If the screen is in landscape mode, it equals the sum (modulus to 0°~360°) of the screen rotation angle (either 90° or 270°) and the Azimuth degree. *Azimuth*, refers to the angle of rotation about the -z axis. This value represents the angle between the device's y axis and the magnetic north pole.

- `rollDeg` - *Roll*, angle of rotation about the y axis. This value represents the angle between a plane perpendicular to the device's screen and a plane perpendicular to the ground.

- `pitchDeg` - *Pitch*, angle of rotation about the x axis. This value represents the angle between a plane parallel to the device's screen and a plane parallel to the ground.

# 2. Python version

3.10

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

5.03

# 6. Current score

3122773.6564

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3122773.6564) has done: 'Diagnosis: The crash happens because `pred_parts` is empty, so `pd.concat(pred_parts, ...)` raises `ValueError: No objects to concatenate`. This occurs when the loop skips all test directories, most likely because `INPUT_PATH` points to `../input/smartphone-decimeter-2022/` but in this environment the data lives under `/kaggle/data/...` (or `/kaggle/input/...`), so `glob` finds no test trips and/or `os.path.exists(gnss_path)` fails for all. The fix is to make cell 41 robust by resolving `INPUT_PATH` to an existing base directory among known candidates and to safely handle the case where `pred_parts` remains empty by creating an empty DataFrame with the expected columns (so cell 42 can still run).

Patch summary: In cell 41 only, add a small path-resolution step that chooses the first existing base path among `INPUT_PATH` and common Kaggle-style locations, then use that resolved path for reading `sample_submission.csv` and enumerating `test` trips. Also guard `pd.concat` with an empty-list fallback that returns an empty DataFrame with the required schema (`tripId`, `UnixTimeMillis`, `LatitudeDegrees`, `LongitudeDegrees`) to keep downstream merge logic unchanged.

Updated cells: Only cell 41 is modified below.

Compatibility notes for cell k+1: Cell 42 expects `pred_all` to exist and contain columns `tripId`, `UnixTimeMillis`, `LatitudeDegrees`, `LongitudeDegrees`. The patch guarantees `pred_all` always exists with those columns (even if no predictions were generated), so the merge/fill logic in cell 42 remains valid and deterministic.

Assumptions: The dataset is available under one of `/kaggle/input/smartphone-decimeter-2022`, `/kaggle/data/smartphone-decimeter-2022`, `../input/smartphone-decimeter-2022/`, or `../kaggle/data/smartphone-decimeter-2022/`, and `sample_submission.csv` exists under the resolved base.'
- What this solution (achieved 3122773.6564) has done: 'Your current score is catastrophically worse than the target (lower-is-better), which strongly suggests a submission schema mismatch rather than a modeling issue. The provided task description says the required key column is `phone`, but your `sample_submission.csv` (and your code) uses `tripId`, so Kaggle likely cannot align predictions to ground truth, producing an enormous penalty. I make the smallest possible change to output both schemas correctly: keep your existing `tripId` pipeline intact, but additionally write a `submission.csv` with `phone` as an alias of `tripId` (and also write `submission_tripId.csv` for debugging). This preserves your core prediction logic (ECEF→lat/lng + Kalman smoothing) and only fixes the output format so the evaluator can score your predictions properly.'
- What this solution (achieved 3122773.6564) has done: 'I make the smallest change that is directly tied to scoring: fix the submission key column to exactly match what this competition’s sample submission expects in your environment (`tripId`, not `phone`). Right now you write `submission.csv` with `phone`, which likely causes Kaggle’s evaluator to mis-join rows and yield a catastrophic score. I keep your prediction pipeline (ECEF→lat/lng interpolation + Kalman smoothing + merge/fill) identical, but write `submission.csv` in the exact same schema/order as `sample_submission.csv`, and keep the `phone`-aliased file only as an extra debug artifact.'
- What this solution (achieved 3122773.6564) has done: 'Your huge (worse) score strongly indicates the evaluator is not matching your rows to the ground truth keys, so your valid lat/lng predictions are effectively ignored/penalized. The smallest score-relevant fix is to ensure the submission schema exactly matches the competition’s expected key column name: `phone` (not `tripId`), while keeping your existing trip folder parsing and prediction logic identical. I also keep a debug `submission_tripId.csv` so you can verify alignment locally without affecting Kaggle scoring. No modeling/smoothing changes are made—only the output columns and the final merge key are adjusted to reduce the catastrophic mismatch gap toward the target.'
- What this solution (achieved 3122773.6564) has done: 'Your current score is so far from the target (lower-is-better) that it almost certainly comes from the evaluator not matching your predictions to the correct keys, not from the Kalman smoothing itself. I make the smallest score-relevant change: build the trip key in the exact format used by `sample_submission.csv` (it uses underscores between drive and phone, e.g. `2020-06-04-US-MTV-1_GooglePixel4`, not `drive/phone`). Then I use that corrected key consistently for filtering the sample rows, merging predictions back, and writing the final `submission.csv` with the exact same columns/order as `sample_submission.csv`. No model/smoothing/feature logic changes are made.'
- What this solution (achieved 3122773.6564) has done: 'Your score indicates the evaluator is effectively not using your predictions, which is most commonly caused by an ID/key mismatch: your code constructs `trip_key` as `drive_phone`, but the sample submission’s `tripId` values may not match that exact pattern in this environment. I add a minimal “key normalization” step: for each test folder, try both `drive_phone` and the legacy `drive/phone` key, and use whichever actually exists in `sample_submission.csv` for that trip. This keeps your ECEF→lat/lng interpolation + Kalman smoothing identical, but greatly increases the chance that predictions merge onto the correct rows instead of falling back to the (bad) sample lat/lng. I also keep the output schema exactly equal to the sample submission columns/order to avoid any silent scoring mismatch.'

# 9. Code solution

## === cell 0
import sys, subprocess, textwrap, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "simdkalman"])



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

pd.set_option("display.max_columns", 50)



## === cell 2
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
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
    lat: np.array
    lng: np.array
    hgt: np.array




## === cell 4
def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2 = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (e2_ * b) * np.sin(t) ** 3, r - (e2 * a) * np.cos(t) ** 3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B) ** 2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)




## === cell 5
def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist


def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df1["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2["LatitudeDegrees"].to_numpy()),
        lng=np.deg2rad(df2["LongitudeDegrees"].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)




## === cell 6
def ecef_to_lat_lng(gnss_df, UnixTimeMillis):
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns
    ecef_df = (
        gnss_df.drop_duplicates(subset="utcTimeMillis")[columns]
        .dropna()
        .reset_index(drop=True)
    )

    if len(ecef_df) < 2:
        if len(ecef_df) == 0:
            return pd.DataFrame(
                {
                    "UnixTimeMillis": UnixTimeMillis,
                    "LatitudeDegrees": np.full_like(
                        UnixTimeMillis, np.nan, dtype=float
                    ),
                    "LongitudeDegrees": np.full_like(
                        UnixTimeMillis, np.nan, dtype=float
                    ),
                }
            )
        ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
        blh = ECEF_to_BLH(ecef)
        return pd.DataFrame(
            {
                "UnixTimeMillis": UnixTimeMillis,
                "LatitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, blh.lat[0], dtype=float)
                ),
                "LongitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, blh.lng[0], dtype=float)
                ),
            }
        )

    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame(
        {
            "UnixTimeMillis": UnixTimeMillis,
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )




## === cell 7
def make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    state_transition = np.array(
        [
            [1, 0, T, 0, 0.5 * T**2, 0],
            [0, 1, 0, T, 0, 0.5 * T**2],
            [0, 0, 1, 0, T, 0],
            [0, 0, 0, 1, 0, T],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_cov_mat,
        observation_model=observation_model,
        observation_noise=obs_cov_mat,
    )
    return kf




## === cell 8
def calc_score(pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score




## === cell 9
INPUT_PATH = "../input/smartphone-decimeter-2022/"




## === cell 10
def apply_kf_smoothing(df, _kf):
    if df is None or len(df) == 0:
        return df

    df_filter = df.copy()
    data = df_filter[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
    if data.shape[0] == 0:
        return df_filter

    data = data.reshape(1, len(data), 2)
    smoothed = _kf.smooth(data)
    df_filter["LatitudeDegrees"] = smoothed.states.mean[0, :, 0]
    df_filter["LongitudeDegrees"] = smoothed.states.mean[0, :, 1]
    return df_filter




## === cell 11
dirnames = sorted(glob.glob(f"{INPUT_PATH}/train/*/*"))



## === cell 12
process_cov_00 = 10**-6
process_cov_11 = 10**-6
process_cov_22 = 10**-6
process_cov_33 = 10**-6
process_cov_44 = 10**-6
process_cov_55 = 10**-6
process_cov_off_diag = 10**-9



## === cell 13
process_diag = [
    process_cov_00,
    process_cov_11,
    process_cov_22,
    process_cov_33,
    process_cov_44,
    process_cov_55,
]




## === cell 14
def make_cov_mat(process_diag, process_cov_off_diag):
    rank = len(process_diag)
    process_cov_mat = np.zeros((rank, rank))
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
drive, phone = dirname.split("/")[-2:]
tripID = f"{drive}/{phone}"
gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
gt_df = pd.read_csv(f"{dirname}/ground_truth.csv")
pred_df = ecef_to_lat_lng(gnss_df, gt_df["UnixTimeMillis"].to_numpy())
gnss_df = pd.read_csv(f"{dirnames[0]}/device_gnss.csv")
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
train_dfs = [
    pd.read_csv(f"{dirname}/device_gnss.csv") for dirname in dirnames[:nb_file]
]



## === cell 30
train_gts = [
    pd.read_csv(f"{dirname}/ground_truth.csv") for dirname in dirnames[:nb_file]
]



## === cell 31
filter_fn = partial(apply_kf_smoothing, _kf=kf0)




## === cell 32
def all_score(train_dfs, train_gts, filter_fn=None):
    """Calculate the score for list of df"""
    scores = []
    for gnss_df, gt_df in zip(train_dfs, train_gts):
        pred_df = ecef_to_lat_lng(gnss_df, gt_df["UnixTimeMillis"].to_numpy())
        if filter_fn:
            pred_df = filter_fn(pred_df)
        score = calc_score(pred_df, gt_df)
        scores.append(score)
    return scores




## === cell 33
def _make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    state_transition = np.array(
        [
            [1, 0, T, 0, 0.5 * T**2, 0],
            [0, 1, 0, T, 0, 0.5 * T**2],
            [0, 0, 1, 0, T, 0],
            [0, 0, 0, 1, 0, T],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_cov_mat,
        observation_model=observation_model,
        observation_noise=obs_cov_mat,
    )
    return kf




## === cell 34
def make_kalman_filter_from_param(param):
    process_diag = [
        param["process_cov_00"],
        param["process_cov_11"],
        param["process_cov_22"],
        param["process_cov_33"],
        param["process_cov_44"],
        param["process_cov_55"],
    ]
    process_cov_off_diag = param["process_cov_off_diag"]

    obs_diag = [param["obs_cov_00"], param["obs_cov_11"]]
    obs_cov_off_diag = param["obs_cov_off_diag"]

    T = param["T"]

    process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)
    obs_cov_mat = make_cov_mat(obs_diag, obs_cov_off_diag)

    _kf = _make_kalman_filter(T, process_cov_mat, obs_cov_mat)
    return _kf




## === cell 35
def objective(trial):
    param = {
        "process_cov_00": trial.suggest_float("process_cov_00", 1e-8, 1e-4, log=True),
        "process_cov_11": trial.suggest_float("process_cov_11", 1e-8, 1e-4, log=True),
        "process_cov_22": trial.suggest_float("process_cov_22", 1e-8, 1e-4, log=True),
        "process_cov_33": trial.suggest_float("process_cov_33", 1e-8, 1e-4, log=True),
        "process_cov_44": trial.suggest_float("process_cov_44", 1e-8, 1e-4, log=True),
        "process_cov_55": trial.suggest_float("process_cov_55", 1e-8, 1e-4, log=True),
        "process_cov_off_diag": trial.suggest_float(
            "process_cov_off_diag", 1e-12, 1e-8, log=True
        ),
        "obs_cov_00": trial.suggest_float("obs_cov_00", 1e-8, 1e-4, log=True),
        "obs_cov_11": trial.suggest_float("obs_cov_11", 1e-8, 1e-4, log=True),
        "obs_cov_off_diag": trial.suggest_float(
            "obs_cov_off_diag", 1e-12, 1e-8, log=True
        ),
        "T": trial.suggest_float("T", 0.6, 1.4),
    }

    _kf = make_kalman_filter_from_param(param)
    filter_fn = partial(apply_kf_smoothing, _kf=_kf)
    scores = all_score(train_dfs, train_gts, filter_fn=filter_fn)
    return np.mean(scores)




## === cell 36
study = optuna.create_study(direction="minimize")



## === cell 37
RUN_OPTUNA = False
if RUN_OPTUNA:
    study.optimize(objective, n_trials=3)



## === cell 38
param = {
    "process_cov_00": 1.2020309620263925e-08,
    "process_cov_11": 2.5818076069527384e-08,
    "process_cov_22": 1.7418556333561943e-05,
    "process_cov_33": 8.783630087500364e-06,
    "process_cov_44": 3.190877235713106e-07,
    "process_cov_55": 1.2137227780694694e-07,
    "process_cov_off_diag": 2.1258126417177658e-09,
    "obs_cov_00": 2.5228974904239324e-07,
    "obs_cov_11": 1.2085757874535475e-05,
    "obs_cov_off_diag": 7.79887179271382e-09,
    "T": 0.7190536201746304,
}




## === cell 39
def apply_kf_smoothing(df, _kf):
    if df is None or len(df) == 0:
        return df

    df_filter = df.copy()
    data = df_filter[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
    if data.shape[0] == 0:
        return df_filter

    data = data.reshape(1, len(data), 2)
    smoothed = _kf.smooth(data)
    df_filter["LatitudeDegrees"] = smoothed.states.mean[0, :, 0]
    df_filter["LongitudeDegrees"] = smoothed.states.mean[0, :, 1]
    return df_filter




## === cell 40
kf_final = make_kalman_filter_from_param(param)



## === cell 41
candidate_bases = [
    INPUT_PATH,
    "/kaggle/input/smartphone-decimeter-2022",
    "/kaggle/data/smartphone-decimeter-2022",
    "/kaggle/input/smartphone-decimeter-2022/smartphone-decimeter-2022",
    "/kaggle/data/smartphone-decimeter-2022/smartphone-decimeter-2022",
    "../kaggle/input/smartphone-decimeter-2022",
    "../kaggle/data/smartphone-decimeter-2022",
]
base_path = None
for p in candidate_bases:
    if p and os.path.exists(p):
        base_path = p.rstrip("/") + "/"
        break
if base_path is None:
    base_path = INPUT_PATH.rstrip("/") + "/"

sample_path = f"{base_path}/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

key_col = (
    "phone"
    if "phone" in sample_sub.columns
    else ("tripId" if "tripId" in sample_sub.columns else None)
)
if key_col is None:
    raise ValueError(
        f"Unknown sample submission schema: columns={list(sample_sub.columns)}"
    )

available_keys = set(sample_sub[key_col].astype(str).unique())

test_base = f"{base_path}/test"
test_trip_dirs = sorted(glob.glob(f"{test_base}/*/*"))  # .../drive/phone

pred_parts = []
used_key_style_counts = {"drive_phone": 0, "drive/phone": 0, "skipped_no_key": 0}
for d in tqdm(test_trip_dirs, desc="Predict test trips"):
    drive, phone_name = d.split("/")[-2:]

    trip_key_candidates = [f"{drive}_{phone_name}", f"{drive}/{phone_name}"]
    trip_key = None
    if trip_key_candidates[0] in available_keys:
        trip_key = trip_key_candidates[0]
        used_key_style_counts["drive_phone"] += 1
    elif trip_key_candidates[1] in available_keys:
        trip_key = trip_key_candidates[1]
        used_key_style_counts["drive/phone"] += 1
    else:
        used_key_style_counts["skipped_no_key"] += 1
        continue

    gnss_path = f"{d}/device_gnss.csv"
    if not os.path.exists(gnss_path):
        continue
    gnss_df = pd.read_csv(gnss_path)

    sub_trip = sample_sub.loc[
        sample_sub[key_col] == trip_key, [key_col, "UnixTimeMillis"]
    ].copy()
    if len(sub_trip) == 0:
        continue

    pred_df = ecef_to_lat_lng(gnss_df, sub_trip["UnixTimeMillis"].to_numpy())
    pred_df[key_col] = trip_key
    pred_df = pred_df[
        [key_col, "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    pred_df = pred_df.sort_values("UnixTimeMillis").reset_index(drop=True)
    pred_df = apply_kf_smoothing(pred_df, kf_final)

    pred_parts.append(pred_df)

if len(pred_parts) == 0:
    pred_all = pd.DataFrame(
        columns=[key_col, "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    )
else:
    pred_all = pd.concat(pred_parts, axis=0, ignore_index=True)

subm = sample_sub[[key_col, "UnixTimeMillis"]].merge(
    pred_all,
    on=[key_col, "UnixTimeMillis"],
    how="left",
)

subm["LatitudeDegrees"] = subm["LatitudeDegrees"].fillna(sample_sub["LatitudeDegrees"])
subm["LongitudeDegrees"] = subm["LongitudeDegrees"].fillna(
    sample_sub["LongitudeDegrees"]
)
subm = subm[[key_col, "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

assert list(subm.columns) == list(sample_sub.columns), (
    subm.columns,
    sample_sub.columns,
)
assert len(subm) == len(sample_sub), (len(subm), len(sample_sub))
subm = (
    subm.merge(
        sample_sub[[key_col, "UnixTimeMillis"]].assign(_row=np.arange(len(sample_sub))),
        on=[key_col, "UnixTimeMillis"],
        how="left",
    )
    .sort_values("_row")
    .drop(columns="_row")
    .reset_index(drop=True)
)

out_path = "submission.csv"
subm.to_csv(out_path, index=False)

if key_col == "phone":
    debug_trip_path = "submission_tripId.csv"
    subm.rename(columns={"phone": "tripId"}).to_csv(debug_trip_path, index=False)
else:
    debug_phone_path = "submission_phone.csv"
    subm.rename(columns={"tripId": "phone"}).to_csv(debug_phone_path, index=False)

print("Resolved base_path:", base_path)
print("Sample key column:", key_col)
print("Used key styles:", used_key_style_counts)
print("Wrote", out_path, "with shape", subm.shape, "and columns", list(subm.columns))
print(subm.head())
print(
    "Predicted rows:",
    0 if pred_all is None else len(pred_all),
    "across trips:",
    len(pred_parts),
)
