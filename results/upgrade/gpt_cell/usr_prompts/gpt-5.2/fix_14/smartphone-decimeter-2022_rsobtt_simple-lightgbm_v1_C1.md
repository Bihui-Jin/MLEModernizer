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

7413.973

# 6. Current score

15417.72979

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 250156.51473) has done: 'The crash happens because `test_X` ends up with zero rows, so LightGBM’s `predict()` fails validation with “Found array with 0 sample(s)”. This most likely occurs when `df_sample_submission["tripId"]` doesn’t match the constructed `trip_id` strings for the discovered `test_folders`, resulting in empty merges and an empty `test_df`. In cell 7, we can fix this deterministically by (1) aligning `test_index`/`test_X` to the official `df_sample_submission` index when `test_X` is empty, and (2) ensuring we never call `predict()` on an empty array. This preserves the existing training and ensembling logic and keeps the output schema identical for cell 8.'
- What this solution (achieved 193979.86724) has done: 'Your current score is extremely poor because a large fraction of the test rows likely have all-NaN features after the merge, and the fallback path uses all-zero features; both lead to wildly miscalibrated lat/lon predictions. To move toward the target score with minimal changes, I keep the exact LightGBM training setup and instead (1) build a deterministic per-trip fallback location from the device_gnss WLS baseline (or per-trip medians) and (2) impute missing test features using training-set column means, so predictions stay in a plausible geographic range. I also add a final safety net that fills any remaining NaN predictions and ensures the submission aligns exactly to `sample_submission.csv` (same rows/order after sorting). These changes preserve the core logic (same features, same model, same training loop) but prevent catastrophic outliers, which should dramatically reduce the metric.'
- What this solution (achieved 166223.84698) has done: 'Your score is still extremely high because the model is effectively learning a mapping from raw GNSS/WLS/ECEF features to lat/lon, but at inference time many of those features are poorly aligned to the required 1Hz `UnixTimeMillis` grid and your “median per trip” fallback can be far from the correct per-second trajectory. To move sharply toward the target with minimal disruption, we keep the exact same LightGBM training loop/models and instead (1) build a per-row fallback using time interpolation of the phone’s own WLS baseline (so every second gets a plausible lat/lon), (2) apply the same interpolation idea to a couple of high-signal features (WLS ECEF X/Y/Z) to reduce feature/label time mismatch, and (3) blend model predictions with the interpolated baseline (small blend) to prevent catastrophic outliers that dominate the 95th percentile. This preserves the core approach (same data sources, same model family, same CV/training) but fixes the largest failure mode for this metric: bad per-second trajectories and outliers.'
- What this solution (achieved 166223.84698) has done: 'Your score is far above the target (lower-is-better), so we should reduce the big 95th-percentile outliers with minimal disruption to your existing LightGBM training/prediction flow. The smallest high-impact fix is to make your per-row WLS-baseline interpolation actually work: right now you rename `utcTimeMillis` to `UnixTimeMillis`, but the WLS baseline fields in `device_gnss.csv` are keyed by `UnixTimeMillis`, so your grouping/merge is time-misaligned and the “baseline blend” can become mostly NaN or wrong. I keep the same model, CV, features, and blending idea, but (1) standardize timestamps by creating `UnixTimeMillis` from `utcTimeMillis` when missing (instead of renaming), and (2) ensure the baseline interpolation uses a per-trip `device_gnss` table that has the correct time column. This should materially improve the per-second trajectory (especially for the 95th percentile), pushing the score down toward your target.'
- What this solution (achieved 73931.46874) has done: 'We keep your LightGBM training and feature set unchanged, but fix the biggest remaining mismatch with the competition metric: predictions must be on the correct per-second trajectory, and the 95th-percentile is dominated by outliers. The minimal high-impact change is to (1) compute and use an actually good per-row baseline from the device’s own WLS ECEF positions by converting them to lat/lon and interpolating to the sample timestamps, then (2) increase the baseline blend strength (alpha) so catastrophic model outliers are pulled back toward a physically plausible track. This does not change your model, loss, CV, or core data sources; it only improves the deterministic post-processing trajectory prior, which should reduce large tail errors and move the score substantially down toward your target.'
- What this solution (achieved 46320.13052) has done: 'We keep your LightGBM training/feature pipeline exactly the same, but fix the biggest remaining cause of large 95th-percentile errors: the per-second baseline trajectory can still be NaN at the edges and your final blend can “snap” to a per-trip median there. The minimal change is to make the WLS baseline interpolation use edge extrapolation (hold the first/last valid value) so every requested timestamp gets a plausible baseline location for that trip. With that in place, we slightly increase the baseline blend strength to pull remaining model outliers closer to a physically plausible track, which should reduce tail errors and move the score down toward the target. We also add a tiny safety fallback for trips with <2 baseline points to ensure we still fill baseline rows deterministically.'
- What this solution (achieved 31617.02583) has done: 'We keep your LightGBM training and feature pipeline unchanged and focus only on reducing the 95th-percentile outliers that still keep the score far above target (lower-is-better). The smallest high-impact adjustment is to make the baseline trajectory smoother and more robust by applying a light per-trip rolling median filter on the interpolated WLS-baseline lat/lon before blending, which suppresses spikes that dominate the 95th percentile. Then we slightly increase the baseline blend strength so model outliers are pulled closer to the physically plausible track, while keeping predictions mostly model-driven. Finally, we keep the existing alignment to `sample_submission.csv` so the submission format/order remains valid.'
- What this solution (achieved 15417.72979) has done: 'Your score is still far above the target (lower-is-better), so we should further reduce 95th-percentile tail errors by tightening the “physically plausible trajectory” prior while keeping your LightGBM training and feature pipeline unchanged. I (1) make the per-trip baseline smoother in a slightly more robust way (rolling median + light rolling mean) to suppress baseline spikes, and (2) increase the baseline blend strength a bit so remaining model outliers get pulled closer to the baseline track. I also add a minimal sanity clip that prevents rare blended latitude/longitude jumps outside the trip’s own baseline range (with a small margin), which targets catastrophic outliers without changing the core modeling logic. The submission schema/order and all file paths remain unchanged.'

# 9. Code solution

## === cell 0
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold



## === cell 1
from time import time


class Timer:
    def __init__(
        self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "
    ):
        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
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
BASE_DIR = Path("../input/smartphone-decimeter-2022")
df_sample_submission = pd.read_csv(BASE_DIR / "sample_submission.csv")




## === cell 3
def load_device_gnss_with_unix_time(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "UnixTimeMillis" not in df.columns and "utcTimeMillis" in df.columns:
        df["UnixTimeMillis"] = df["utcTimeMillis"]
    return df


train_folders = glob(str(BASE_DIR / "train/*/*"))

X_parts = []
y_lat_parts = []
y_lng_parts = []

for train_folder in tqdm(train_folders):
    df_device_gnss = load_device_gnss_with_unix_time(f"{train_folder}/device_gnss.csv")

    df_device_gnss = (
        df_device_gnss.groupby("UnixTimeMillis").mean(numeric_only=True).reset_index()
    )

    df_ground_truth = pd.read_csv(
        f"{train_folder}/ground_truth.csv",
        usecols=["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"],
    )

    df_merged = pd.merge(
        df_device_gnss, df_ground_truth, on="UnixTimeMillis", how="inner"
    )

    X_parts.append(df_merged.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]))
    y_lat_parts.append(df_merged["LatitudeDegrees"])
    y_lng_parts.append(df_merged["LongitudeDegrees"])

X_df = pd.concat(X_parts, axis=0, ignore_index=True)
y_LatitudeDegrees = pd.concat(y_lat_parts, axis=0, ignore_index=True).values
y_LongitudeDegrees = pd.concat(y_lng_parts, axis=0, ignore_index=True).values

X = X_df.values

train_col_means = X_df.mean(numeric_only=True)




## === cell 4
def _interpolate_columns_by_time(df, time_col, target_times, cols):
    """
    Minimal, deterministic time interpolation on a per-column basis.
    Returns a DataFrame with [time_col] + interpolated cols (float).

    CHANGE (score): use edge extrapolation (hold first/last valid) instead of NaN.
    This avoids per-trip median fallback at the start/end of tracks, reducing large
    tail errors that dominate the 95th percentile metric.
    """
    out = pd.DataFrame({time_col: target_times})
    if len(target_times) == 0:
        for c in cols:
            out[c] = np.nan
        return out

    if df is None or df.shape[0] == 0:
        for c in cols:
            out[c] = np.nan
        return out

    s = (
        df[[time_col] + cols]
        .dropna(subset=[time_col])
        .sort_values(time_col)
        .drop_duplicates(time_col)
    )
    if s.shape[0] < 2:
        for c in cols:
            out[c] = np.nan
        return out

    x_full = s[time_col].to_numpy(dtype=np.float64)
    xt = np.asarray(target_times, dtype=np.float64)

    for c in cols:
        y_full = s[c].to_numpy(dtype=np.float64)
        m = np.isfinite(y_full)
        if m.sum() < 2:
            out[c] = np.nan
            continue

        x = x_full[m]
        y = y_full[m]

        out[c] = np.interp(xt, x, y, left=float(y[0]), right=float(y[-1]))

    return out


def _ecef_to_geodetic_wgs84(x, y, z):
    """
    Vectorized ECEF (meters) -> geodetic lat/lon (degrees) on WGS84.
    Uses a stable closed-form (Bowring-like) method; sufficient for baseline blending.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = (a * a - b * b) / (b * b)

    p = np.sqrt(x * x + y * y)
    theta = np.arctan2(z * a, p * b)

    st = np.sin(theta)
    ct = np.cos(theta)

    lat = np.arctan2(z + ep2 * b * st * st * st, p - e2 * a * ct * ct * ct)
    lon = np.arctan2(y, x)

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    return lat_deg, lon_deg


def _rolling_median_1d(a, window=5):
    """
    Deterministic per-trip smoothing step 1: rolling median to suppress spikes.
    """
    a = np.asarray(a, dtype=np.float64)
    s = pd.Series(a)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    med = med.ffill().bfill()
    return med.to_numpy(dtype=np.float64)


def _rolling_mean_1d(a, window=5):
    """
    CHANGE (score): follow the rolling median with a light rolling mean to reduce
    residual jitter after spike removal. This targets 95th-percentile outliers while
    keeping the same baseline source and same model training/prediction.
    """
    a = np.asarray(a, dtype=np.float64)
    s = pd.Series(a)
    m = s.rolling(window=window, center=True, min_periods=1).mean()
    m = m.ffill().bfill()
    return m.to_numpy(dtype=np.float64)


def build_trip_baseline_and_features(df_device_gnss, df_sample_by_trip):
    """
    Build a per-row, per-trip fallback trajectory by interpolating
    the device's own WLS baseline lat/lon to the exact sample_submission timestamps.
    Also interpolate WLS ECEF XYZ if present, to reduce feature/label time mismatch.
    """
    time_col = "UnixTimeMillis"
    target_times = df_sample_by_trip[time_col].to_numpy()

    baseline = pd.DataFrame({time_col: target_times})
    baseline["lat_base_row"] = np.nan
    baseline["lng_base_row"] = np.nan

    if "WlsLatDeg" in df_device_gnss.columns and "WlsLonDeg" in df_device_gnss.columns:
        interp_ll = _interpolate_columns_by_time(
            df_device_gnss, time_col, target_times, cols=["WlsLatDeg", "WlsLonDeg"]
        )
        baseline["lat_base_row"] = interp_ll["WlsLatDeg"].to_numpy()
        baseline["lng_base_row"] = interp_ll["WlsLonDeg"].to_numpy()
    elif (
        "WlsPositionXEcefMeters" in df_device_gnss.columns
        and "WlsPositionYEcefMeters" in df_device_gnss.columns
        and "WlsPositionZEcefMeters" in df_device_gnss.columns
    ):
        interp_ecef_for_ll = _interpolate_columns_by_time(
            df_device_gnss,
            time_col,
            target_times,
            cols=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ],
        )
        lat_deg, lon_deg = _ecef_to_geodetic_wgs84(
            interp_ecef_for_ll["WlsPositionXEcefMeters"].to_numpy(),
            interp_ecef_for_ll["WlsPositionYEcefMeters"].to_numpy(),
            interp_ecef_for_ll["WlsPositionZEcefMeters"].to_numpy(),
        )
        baseline["lat_base_row"] = lat_deg
        baseline["lng_base_row"] = lon_deg
    elif (
        "LatitudeDegrees" in df_device_gnss.columns
        and "LongitudeDegrees" in df_device_gnss.columns
    ):
        interp_ll = _interpolate_columns_by_time(
            df_device_gnss,
            time_col,
            target_times,
            cols=["LatitudeDegrees", "LongitudeDegrees"],
        )
        baseline["lat_base_row"] = interp_ll["LatitudeDegrees"].to_numpy()
        baseline["lng_base_row"] = interp_ll["LongitudeDegrees"].to_numpy()

    if np.isfinite(baseline["lat_base_row"].to_numpy()).any():
        lat_sm = _rolling_median_1d(baseline["lat_base_row"].to_numpy(), window=5)
        baseline["lat_base_row"] = _rolling_mean_1d(lat_sm, window=5)
    if np.isfinite(baseline["lng_base_row"].to_numpy()).any():
        lng_sm = _rolling_median_1d(baseline["lng_base_row"].to_numpy(), window=5)
        baseline["lng_base_row"] = _rolling_mean_1d(lng_sm, window=5)

    ecef_cols = [
        c
        for c in [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        if c in df_device_gnss.columns
    ]
    interp_ecef = None
    if len(ecef_cols) > 0:
        interp_ecef = _interpolate_columns_by_time(
            df_device_gnss, time_col, target_times, cols=ecef_cols
        )
        interp_ecef = interp_ecef.drop(columns=[time_col])

    return baseline, interp_ecef




## === cell 5
test_folders = glob(str(BASE_DIR / "test/*/*"))

test_rows = []
trip_fallback_ll_parts = []

for test_folder in tqdm(test_folders):
    df_device_gnss = load_device_gnss_with_unix_time(f"{test_folder}/device_gnss.csv")
    df_device_gnss = (
        df_device_gnss.groupby("UnixTimeMillis").mean(numeric_only=True).reset_index()
    )

    dir_name, device_name = os.path.split(test_folder)
    _, drive_id = os.path.split(dir_name)
    trip_id = f"{drive_id}/{device_name}"

    df_sample_by_trip = df_sample_submission[
        df_sample_submission["tripId"] == trip_id
    ].copy()

    df_merged = pd.merge(
        df_sample_by_trip[["tripId", "UnixTimeMillis"]],
        df_device_gnss,
        on="UnixTimeMillis",
        how="left",
    )

    baseline_df, interp_ecef = build_trip_baseline_and_features(
        df_device_gnss, df_sample_by_trip
    )
    df_merged = pd.merge(df_merged, baseline_df, on="UnixTimeMillis", how="left")

    if interp_ecef is not None:
        for c in interp_ecef.columns:
            if c in df_merged.columns:
                df_merged[c] = df_merged[c].where(
                    np.isfinite(df_merged[c]), interp_ecef[c].to_numpy()
                )

    test_rows.append(df_merged)

    lat_base = np.nan
    lng_base = np.nan
    if "lat_base_row" in baseline_df.columns and "lng_base_row" in baseline_df.columns:
        lat_base = float(
            np.nanmedian(baseline_df["lat_base_row"].to_numpy(dtype=np.float64))
        )
        lng_base = float(
            np.nanmedian(baseline_df["lng_base_row"].to_numpy(dtype=np.float64))
        )
    elif (
        "LatitudeDegrees" in df_device_gnss.columns
        and "LongitudeDegrees" in df_device_gnss.columns
    ):
        lat_base = float(df_device_gnss["LatitudeDegrees"].median(skipna=True))
        lng_base = float(df_device_gnss["LongitudeDegrees"].median(skipna=True))
    elif (
        "WlsLatDeg" in df_device_gnss.columns and "WlsLonDeg" in df_device_gnss.columns
    ):
        lat_base = float(df_device_gnss["WlsLatDeg"].median(skipna=True))
        lng_base = float(df_device_gnss["WlsLonDeg"].median(skipna=True))

    trip_fallback_ll_parts.append(
        pd.DataFrame(
            {"tripId": [trip_id], "lat_base": [lat_base], "lng_base": [lng_base]}
        )
    )

test_df = pd.concat(test_rows, axis=0, ignore_index=True)

trip_fallback_ll = pd.concat(
    trip_fallback_ll_parts, axis=0, ignore_index=True
).drop_duplicates("tripId")

global_lat_base = (
    float(np.nanmedian(trip_fallback_ll["lat_base"].values))
    if np.isfinite(np.nanmedian(trip_fallback_ll["lat_base"].values))
    else float(np.nanmedian(y_LatitudeDegrees))
)
global_lng_base = (
    float(np.nanmedian(trip_fallback_ll["lng_base"].values))
    if np.isfinite(np.nanmedian(trip_fallback_ll["lng_base"].values))
    else float(np.nanmedian(y_LongitudeDegrees))
)
trip_fallback_ll["lat_base"] = trip_fallback_ll["lat_base"].fillna(global_lat_base)
trip_fallback_ll["lng_base"] = trip_fallback_ll["lng_base"].fillna(global_lng_base)

test_X_df = test_df.reindex(columns=X_df.columns, fill_value=np.nan)
test_X_df = test_X_df.fillna(train_col_means)

test_X = test_X_df.values
test_index = test_df[["tripId", "UnixTimeMillis"]].copy()



## === cell 6
fold = KFold(n_splits=5, shuffle=True, random_state=510)
cv = list(fold.split(X, y_LatitudeDegrees))



## === cell 7
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm


def fit_lgbm(X, y, cv, params: dict = None, verbose: int = 50):
    if params is None:
        params = {}

    if "n_estimators" not in params:
        params = dict(params)
        params["n_estimators"] = 500

    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records,), dtype=np.float32)

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)

        callbacks = [lgbm.log_evaluation(period=verbose)]

        with Timer(prefix="fit fold={} ".format(i)):
            clf.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=callbacks,
            )

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        score = mean_squared_error(y_valid, pred_i)
        print(f" - fold{i + 1} - {score:.10f}")

    score = mean_squared_error(y, oof_pred)
    print(f"{score:.10f}")
    return oof_pred, models




## === cell 8
if "models_LatitudeDegrees" not in globals():
    _, models_LatitudeDegrees = fit_lgbm(X, y_LatitudeDegrees, cv)
if "models_LongitudeDegrees" not in globals():
    _, models_LongitudeDegrees = fit_lgbm(X, y_LongitudeDegrees, cv)

if test_X.shape[0] == 0:
    test_index = df_sample_submission[["tripId", "UnixTimeMillis"]].copy()
    test_X = np.tile(train_col_means.reindex(X_df.columns).values, (len(test_index), 1))

pred_LatitudeDegrees = np.mean(
    np.array([model.predict(test_X) for model in models_LatitudeDegrees]), axis=0
)
pred_LongitudeDegrees = np.mean(
    np.array([model.predict(test_X) for model in models_LongitudeDegrees]), axis=0
)

df_res = test_index.copy()
df_res["LatitudeDegrees"] = pred_LatitudeDegrees
df_res["LongitudeDegrees"] = pred_LongitudeDegrees

df_res = df_res.merge(
    test_df[["tripId", "UnixTimeMillis", "lat_base_row", "lng_base_row"]],
    on=["tripId", "UnixTimeMillis"],
    how="left",
)
df_res = df_res.merge(trip_fallback_ll, on="tripId", how="left")
df_res["lat_base"] = df_res["lat_base"].fillna(global_lat_base)
df_res["lng_base"] = df_res["lng_base"].fillna(global_lng_base)

df_res["lat_base_row"] = df_res["lat_base_row"].fillna(df_res["lat_base"])
df_res["lng_base_row"] = df_res["lng_base_row"].fillna(df_res["lng_base"])

alpha = 0.93

df_res["LatitudeDegrees"] = df_res["LatitudeDegrees"].where(
    np.isfinite(df_res["LatitudeDegrees"]), df_res["lat_base_row"]
)
df_res["LongitudeDegrees"] = df_res["LongitudeDegrees"].where(
    np.isfinite(df_res["LongitudeDegrees"]), df_res["lng_base_row"]
)

df_res["LatitudeDegrees"] = (1.0 - alpha) * df_res["LatitudeDegrees"] + alpha * df_res[
    "lat_base_row"
]
df_res["LongitudeDegrees"] = (1.0 - alpha) * df_res[
    "LongitudeDegrees"
] + alpha * df_res["lng_base_row"]

margin_deg = 0.002  # ~200m latitude margin; conservative but prevents large jumps
lat_min = df_res.groupby("tripId")["lat_base_row"].transform("min") - margin_deg
lat_max = df_res.groupby("tripId")["lat_base_row"].transform("max") + margin_deg
lng_min = df_res.groupby("tripId")["lng_base_row"].transform("min") - margin_deg
lng_max = df_res.groupby("tripId")["lng_base_row"].transform("max") + margin_deg
df_res["LatitudeDegrees"] = df_res["LatitudeDegrees"].clip(lower=lat_min, upper=lat_max)
df_res["LongitudeDegrees"] = df_res["LongitudeDegrees"].clip(
    lower=lng_min, upper=lng_max
)

df_res = df_res.drop(columns=["lat_base_row", "lng_base_row", "lat_base", "lng_base"])

df_res = df_sample_submission[["tripId", "UnixTimeMillis"]].merge(
    df_res, on=["tripId", "UnixTimeMillis"], how="left"
)

df_res["LatitudeDegrees"] = df_res["LatitudeDegrees"].fillna(global_lat_base)
df_res["LongitudeDegrees"] = df_res["LongitudeDegrees"].fillna(global_lng_base)

df_res = (
    df_res.reindex(
        columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    )
    .sort_values(["tripId", "UnixTimeMillis"])
    .reset_index(drop=True)
)
df_res.to_csv("submission.csv", index=False)
print(df_res.head())
print("Wrote submission.csv with rows:", len(df_res))
