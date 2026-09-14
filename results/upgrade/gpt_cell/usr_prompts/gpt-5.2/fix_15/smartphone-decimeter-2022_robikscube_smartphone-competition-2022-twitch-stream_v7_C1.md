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
plotly==5.24.1
plotly-express==0.4.1
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

5.19

# 6. Current score

nan

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I make the pipeline reliably produce a valid `submission.csv` by fixing the required submission schema (the competition uses `tripId`, not `phone`) and ensuring predictions are aligned and ordered exactly like `sample_submission.csv`. I also fix a couple of small but score-relevant bugs: places where `trip_id` (a global) is mistakenly used instead of the function argument `tripId`, and a stop-detection bug where timestamps weren’t filtered by trip (which can wrongly “freeze” coordinates across different trips). Finally, I keep the same core approach (WLS ECEF → BLH + spline interpolation + optional stop smoothing), but make it robust to edge cases (very short sequences) so it always runs end-to-end within constraints.'
- What this solution (achieved nan) has done: 'I keep your baseline WLS(ECEF→BLH) + spline interpolation + stop smoothing intact, but fix two score-critical issues that can silently produce a `nan`/invalid submission: (1) the stop-detection distance currently uses a shifted lat/lon without guarding trip boundaries/NaNs and can mark long “stops” incorrectly, and (2) the output lat/lon NaN filling is too late and can still leave NaNs after the merge. I also ensure the submission is *exactly* aligned to `sample_submission.csv` order and contains no missing coordinates, which is required for a valid Kaggle score. Changes are minimal and localized to stop-detection distance computation and final assembly/filling.'
- What this solution (achieved nan) has done: 'I make the smallest changes needed to eliminate the likely cause of your `nan` Kaggle score: invalid (NaN/inf) coordinates created during stop detection from the custom haversine (`arcsin` without clipping), which can propagate and invalidate scoring. Specifically, I clip the haversine “a” term to `[0,1]` and use `arcsin(sqrt(a))` to keep distances finite and stable, while preserving your exact stop-smoothing logic and WLS→BLH→spline core approach. I also fix a units bug in `speed_calc` (ms vs seconds) that can distort diagnostics/thresholding decisions (without changing your stop detection which uses `dist_prev` directly), and I add a final hard guard that replaces any remaining non-finite lat/lon with safe per-trip filled values before writing the CSV. These changes are localized and aimed at producing a valid, scorable submission whose score should move toward your target by avoiding catastrophic invalid rows.'
- What this solution (achieved nan) has done: 'I keep your core pipeline (WLS ECEF→BLH + spline interpolation + stop smoothing) intact, but fix two places that can produce an unscorable/`nan` submission: (1) spline interpolation currently operates on raw radians and can “wrap” at ±π for longitude (and near poles for latitude), creating huge jumps and non-finite values; I unwrap radians per trip before fitting the spline and then wrap back to [-π, π]. (2) the final global `ffill/bfill` across all trips can leak values between trips when a merge yields NaNs, so I make the final fill strictly per-`tripId` and only then fall back to safe medians, ensuring every row is valid and aligned exactly to `sample_submission.csv`.'
- What this solution (achieved nan) has done: 'I first fix the `nan` score by eliminating an evaluation-breaking schema issue: the competition expects `tripId` (not `phone`), and your current stop-smoothing uses `ss` from a different sample submission file (`../input/.../sample_submission.csv`) while later you read `../input/smartphone-decimeter-2022/sample_submission.csv`, which can silently misalign merge keys and create invalid/empty joins. Then I make two minimal, score-relevant robustness tweaks that keep your same core pipeline (WLS ECEF→BLH + spline + stop smoothing): (1) ensure `InterpolateUnivariateSpline` always receives unique, strictly increasing times (drop duplicates and guard m<2), and (2) make stop-detection grouping use per-trip time diffs computed after sorting by time to prevent accidental long windows. Finally, I ensure the final submission is *exactly* aligned to the chosen sample submission (same source, same order) and contains no NaN/inf lat/lon, so Kaggle always returns a numeric score that can move toward your 5.19 target.'
- What this solution (achieved nan) has done: 'I keep your core approach (WLS ECEF→BLH + spline interpolation + stop smoothing) exactly the same, but fix two remaining “nan score / invalid submission” failure modes that can still happen in edge cases: (1) `InterpolatedUnivariateSpline` can silently yield NaNs if `utcTimeMillis` becomes non-strictly-increasing after `drop_duplicates` (float casting) or contains tiny non-monotonicities; I enforce strictly increasing `TIME` by sorting + taking `np.unique` indices and guarding again. (2) The stop-smoothing currently computes `time_group` using `.diff()` on a filtered subset; if the subset keeps original indices, diff is still correct, but to avoid any weirdness and ensure per-trip grouping is stable, I reset index within the filtered stopped dataframe before diff/cumsum. These are minimal, localized robustness fixes intended to ensure the pipeline always produces a fully finite, correctly aligned `submission.csv` so Kaggle returns a numeric score that can move toward your 5.19 target.'
- What this solution (achieved nan) has done: 'I make two minimal, score-relevant fixes that keep your exact core pipeline (WLS ECEF→BLH + spline interpolation + stop smoothing) unchanged but prevent silent data-loss and misalignment that can yield `nan`/invalid scores. First, I ensure every predicted row survives the merge by deduplicating `sub_pred` to one row per (`tripId`,`UnixTimeMillis`) and switching the merge validation to tolerate duplicate-left issues (while still enforcing sample submission uniqueness). Second, I add a final “exact order + complete fill” guard that fills any remaining missing timestamps per trip using the nearest available prediction before per-trip ffill/bfill, ensuring the submission is fully populated and scorable. These changes are localized to the submission assembly stage and are intended to move you from `nan` to a numeric score, toward your 5.19 target.'
- What this solution (achieved nan) has done: 'Your current “nan” score is almost certainly from an invalid submission (NaN/inf or missing predictions after the merge), not from the WLS+spline core itself. I keep your exact modeling approach and stop-smoothing intact, but add two minimal guards that prevent silent merge loss: (1) force `sub_df` to cover *all* (`tripId`,`UnixTimeMillis`) pairs by left-merging your predictions onto `sample_submission` *before* stop detection, and (2) if a trip has no valid WLS points, fall back to a safe per-trip constant (the trip’s first available prediction, else global median). This should reliably yield a numeric Kaggle score and move you toward the 5.19 target by avoiding catastrophic invalid rows while preserving your semantics.'
- What this solution (achieved nan) has done: 'Your current “nan” score is most likely still caused by an invalid submission (some rows ending up missing/NaN after the merge) rather than model quality, so I add one small, score-relevant guard to guarantee full coverage: ensure `sub_df_raw` is deduplicated to exactly one row per (`tripId`,`UnixTimeMillis`) before merging to `sample_submission`. I also make the stop-smoothing window computation robust to tiny timestamp gaps by explicitly requiring the stop-group to be contiguous in the *full* trip timeline (not just within the filtered stopped rows), which prevents accidental long “stop freezes” that can hurt the 95th percentile. Finally, I keep your core WLS(ECEF→BLH)+spline and the same stop-smoothing approach, but enforce a final “no-missing, per-trip nearest fill” *after* aligning exactly to `sample_submission` to ensure Kaggle always returns a numeric score.'
- What this solution (achieved nan) has done: 'I keep your WLS(ECEF→BLH)+spline interpolation and the same stop-smoothing idea, but make two minimal, score-relevant fixes that should move you from an invalid/`nan` score to a numeric score closer to the 5.19 target. First, I prevent stop-smoothing from “freezing” across long gaps by requiring the stop window to also be contiguous in the full trip timeline (not just within the filtered stopped rows), which reduces catastrophic 95th-percentile errors. Second, I ensure the submission assembly never loses/misorders rows by using the sample submission as the single source of truth early, and by enforcing one prediction per (`tripId`,`UnixTimeMillis`) before any smoothing/filling. These changes are localized to stop detection windowing and final submission alignment/finiteness guards, and the script still write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I keep your core pipeline (WLS ECEF→BLH + spline interpolation + your existing stop-smoothing) unchanged, and only apply two minimal fixes that can directly turn a `nan` score into a valid numeric score and improve it toward 5.19. First, your stop detection currently uses `dist_prev` computed against the *pre-stop-smoothed* coordinates; after you overwrite coordinates inside a stop window, `dist_prev` is stale, which can create incorrect stop grouping and large 95th-percentile errors—so we recompute `dist_prev`/`dist_post` once after stop smoothing before any further logic. Second, I remove a submission-breaking mismatch in the visualization helper default (`color_col="phone"`) by aligning it to `tripId` (so the notebook can run end-to-end in non-interactive runs without failing when called). Everything else is preserved, and the script still writes a fully finite `submission.csv` aligned exactly to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, warnings

warnings.filterwarnings("ignore")



## === cell 1
try:
    ip = get_ipython()
    if ip is not None:
        try:
            ip.run_line_magic("load_ext", "lab_black")
        except ModuleNotFoundError:
            try:
                ip.run_line_magic("load_ext", "nb_black")
            except ModuleNotFoundError:
                pass
except NameError:
    pass



## === cell 2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px

pd.set_option("display.max_columns", 500)



## === cell 3
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"



## === cell 4
gt = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/ground_truth.csv"
)
gnss = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_gnss.csv"
)
imu = pd.read_csv(
    "../input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/device_imu.csv"
)



## === cell 5
import glob
from dataclasses import dataclass
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

INPUT_PATH = "../input/smartphone-decimeter-2022"

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


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


def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
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


def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis):
    """
    WLS ECEF->BLH + spline interpolation.
    Small robustness guards only (avoid spline NaNs / wrap jumps), preserving core logic.
    """
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns

    ecef_df = gnss_df[columns].copy()
    ecef_df = ecef_df.dropna().sort_values("utcTimeMillis").reset_index(drop=True)

    if len(ecef_df) > 0:
        t = ecef_df["utcTimeMillis"].to_numpy()
        _, unique_idx = np.unique(t, return_index=True)
        ecef_df = ecef_df.iloc[np.sort(unique_idx)].reset_index(drop=True)

    if len(ecef_df) == 0:
        return pd.DataFrame(
            {
                "tripId": tripID,
                "UnixTimeMillis": np.asarray(UnixTimeMillis, dtype=np.int64),
                "LatitudeDegrees": np.nan,
                "LongitudeDegrees": np.nan,
            }
        )

    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh = ECEF_to_BLH(ecef)

    TIME = ecef_df["utcTimeMillis"].to_numpy(dtype=np.float64)
    m = len(TIME)

    if m < 2:
        lat = np.clip(blh.lat.astype(np.float64)[-1], -np.pi / 2, np.pi / 2)
        lng = _wrap_to_pi(blh.lng.astype(np.float64)[-1])
        UnixTimeMillis = np.asarray(UnixTimeMillis, dtype=np.int64)
        return pd.DataFrame(
            {
                "tripId": tripID,
                "UnixTimeMillis": UnixTimeMillis,
                "LatitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, lat, dtype=float)
                ),
                "LongitudeDegrees": np.degrees(
                    np.full_like(UnixTimeMillis, lng, dtype=float)
                ),
            }
        )

    k = int(min(3, max(1, m - 1)))

    lat_u = np.unwrap(blh.lat.astype(np.float64))
    lng_u = np.unwrap(blh.lng.astype(np.float64))

    lat_spl = InterpolatedUnivariateSpline(TIME, lat_u, k=k, ext=3)
    lng_spl = InterpolatedUnivariateSpline(TIME, lng_u, k=k, ext=3)

    UnixTimeMillis = np.asarray(UnixTimeMillis, dtype=np.float64)
    lat = lat_spl(UnixTimeMillis)
    lng = lng_spl(UnixTimeMillis)

    lat = np.clip(lat, -np.pi / 2, np.pi / 2)
    lng = _wrap_to_pi(lng)

    return pd.DataFrame(
        {
            "tripId": tripID,
            "UnixTimeMillis": UnixTimeMillis.astype(np.int64),
            "LatitudeDegrees": np.degrees(lat),
            "LongitudeDegrees": np.degrees(lng),
        }
    )


def calc_score(tripID, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score




## === cell 6
ss = pd.read_csv(f"{INPUT_PATH}/sample_submission.csv")



## === cell 7
trip_id = "2020-05-15-US-MTV-1/GooglePixel4XL"
baseline = ecef_to_lat_lng(trip_id, gnss, gt["UnixTimeMillis"].values)




## === cell 8
def visualize_traffic(
    df,
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    center=None,
    color_col="tripId",
    label_col="tripId",
    zoom=9,
    opacity=1,
):
    if center is None:
        center = {
            "lat": df[lat_col].mean(),
            "lon": df[lon_col].mean(),
        }
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        labels=label_col,
        zoom=zoom,
        center=center,
        height=600,
        width=800,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()


def plot_gt_vs_baseline(tripId):
    gt = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/ground_truth.csv")
    gnss = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_gnss.csv")
    _ = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_imu.csv")

    baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
    baseline["isGT"] = False
    gt["isGT"] = True
    gt["tripId"] = tripId

    combined = (
        pd.concat([baseline, gt[baseline.columns]], axis=0)
        .reset_index(drop=True)
        .copy()
    )

    visualize_traffic(
        combined,
        lat_col="LatitudeDegrees",
        lon_col="LongitudeDegrees",
        color_col="isGT",
        zoom=10,
    )




## === cell 9
plot_gt_vs_baseline(trip_id)



## === cell 10
from glob import glob

train_gts = glob(f"{INPUT_PATH}/train/*/*/ground_truth.csv")
trip_ids = ["/".join(p.split("/")[-3:-1]) for p in train_gts]



## === cell 11
tripId = trip_ids[10]
gt = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/ground_truth.csv")
gnss = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_gnss.csv")
imu = pd.read_csv(f"{INPUT_PATH}/train/{tripId}/device_imu.csv")

baseline = ecef_to_lat_lng(tripId, gnss, gt["UnixTimeMillis"].values)
baseline["isGT"] = False
gt["isGT"] = True
gt["tripId"] = tripId

combined = (
    pd.concat([baseline, gt[baseline.columns]], axis=0).reset_index(drop=True).copy()
)



## === cell 12
offset = 150_000
start = 1607640760432 + offset
combined.query("UnixTimeMillis < @start")
visualize_traffic(combined.query("UnixTimeMillis < @start"), color_col="isGT", zoom=16)



## === cell 13
import glob

sample_df = ss.copy()

pred_dfs = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/test/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)

    UnixTimeMillis = sample_df.loc[
        sample_df["tripId"] == tripID, "UnixTimeMillis"
    ].to_numpy()
    pred_dfs.append(ecef_to_lat_lng(tripID, gnss_df, UnixTimeMillis))

sub_df_raw = pd.concat(pred_dfs, ignore_index=True)

sub_df_raw = (
    sub_df_raw.sort_values(["tripId", "UnixTimeMillis"])
    .groupby(["tripId", "UnixTimeMillis"], as_index=False)
    .agg({"LatitudeDegrees": "median", "LongitudeDegrees": "median"})
)

sub_df = ss[["tripId", "UnixTimeMillis"]].merge(
    sub_df_raw, on=["tripId", "UnixTimeMillis"], how="left", validate="one_to_one"
)

baselines = []
gts = []
for dirname in tqdm(sorted(glob.glob(f"{INPUT_PATH}/train/*/*"))):
    drive, phone = dirname.split("/")[-2:]
    tripID = f"{drive}/{phone}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv", low_memory=False)
    gt_df = pd.read_csv(f"{dirname}/ground_truth.csv", low_memory=False)
    baseline_df = ecef_to_lat_lng(tripID, gnss_df, gt_df["UnixTimeMillis"].to_numpy())
    baselines.append(baseline_df)
    gts.append(gt_df)
baselines = pd.concat(baselines, ignore_index=True)
gts = pd.concat(gts, ignore_index=True)



## === cell 14
baselines["group"] = "train_baseline"
sub_df["group"] = "submission_baseline"
gts["group"] = "train_ground_truth"
combined = pd.concat([baselines, sub_df, gts]).reset_index(drop=True).copy()



## === cell 15
sf_paths = combined.query("LatitudeDegrees > 36").copy()
la_paths = combined.query("LatitudeDegrees < 36").copy()



## === cell 16
visualize_traffic(
    sf_paths.sample(frac=0.2, random_state=0),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)



## === cell 17
visualize_traffic(
    la_paths.sample(frac=0.2, random_state=0),
    lat_col="LatitudeDegrees",
    lon_col="LongitudeDegrees",
    color_col="group",
    zoom=9,
)




## === cell 18
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points on the earth."""
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    dist = 2 * RADIUS * np.arcsin(np.sqrt(a))
    return dist


def add_prev_post_shift(
    df,
    lat_col="LatitudeDegrees",
    lng_col="LongitudeDegrees",
    dist_suffix="",
    sortby=["tripId", "UnixTimeMillis"],
):
    """
    Shifts within tripId to avoid boundary leakage that can cause false stop detection.
    """
    df = df.sort_values(sortby).reset_index(drop=True)
    df[f"{lat_col}_shift1"] = df.groupby(["tripId"])[lat_col].shift(1)
    df[f"{lng_col}_shift1"] = df.groupby(["tripId"])[lng_col].shift(1)
    df[f"{lat_col}_shift-1"] = df.groupby(["tripId"])[lat_col].shift(-1)
    df[f"{lng_col}_shift-1"] = df.groupby(["tripId"])[lng_col].shift(-1)

    df["UnixTimeMillis_shift1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(1)
    df["UnixTimeMillis_shift-1"] = df.groupby(["tripId"])["UnixTimeMillis"].shift(-1)

    df[f"dist_prev{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift1"], df[f"{lng_col}_shift1"]
    )
    df[f"dist_post{dist_suffix}"] = calc_haversine(
        df[lat_col], df[lng_col], df[f"{lat_col}_shift-1"], df[f"{lng_col}_shift-1"]
    )

    df.loc[
        df[f"{lat_col}_shift1"].isna() | df[f"{lng_col}_shift1"].isna(),
        f"dist_prev{dist_suffix}",
    ] = np.nan
    df.loc[
        df[f"{lat_col}_shift-1"].isna() | df[f"{lng_col}_shift-1"].isna(),
        f"dist_post{dist_suffix}",
    ] = np.nan

    return df




## === cell 19
baselines2 = add_prev_post_shift(baselines)
baselines2["UnixTimeMillis_prev_diff"] = (
    baselines2["UnixTimeMillis"] - baselines2["UnixTimeMillis_shift1"]
)
baselines2["speed_calc"] = baselines2["dist_prev"] / (
    baselines2["UnixTimeMillis_prev_diff"] / 1000.0
)

sub_df = add_prev_post_shift(sub_df)
sub_df["UnixTimeMillis_prev_diff"] = (
    sub_df["UnixTimeMillis"] - sub_df["UnixTimeMillis_shift1"]
)
sub_df["speed_calc"] = sub_df["dist_prev"] / (
    sub_df["UnixTimeMillis_prev_diff"] / 1000.0
)



## === cell 20
baselines2.query("dist_prev < 5")["dist_prev"].plot(
    kind="hist", bins=50, title="Distribution of Speeds < 5"
)
plt.show()



## === cell 21
sub = sub_df.copy()
sub["stopped"] = False

sub = sub.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

for trip, sub_trip in sub.groupby("tripId", sort=False):
    sub_trip = sub_trip.sort_values("UnixTimeMillis").copy()

    sub_stopped = sub_trip.loc[
        sub_trip["dist_prev"].notna() & (sub_trip["dist_prev"] < 2)
    ].copy()
    sub_stopped = sub_stopped.sort_values("UnixTimeMillis").reset_index(drop=True)

    if len(sub_stopped) == 0:
        continue

    full_dt = sub_trip["UnixTimeMillis"].diff().to_numpy()
    full_dt_by_time = pd.Series(full_dt, index=sub_trip["UnixTimeMillis"].to_numpy())
    sub_stopped["full_dt"] = sub_stopped["UnixTimeMillis"].map(full_dt_by_time)

    sub_stopped["UnixTimeMillis_diff"] = sub_stopped["UnixTimeMillis"].diff()
    sub_stopped["big_timeshift"] = (sub_stopped["UnixTimeMillis_diff"] > 2_000) | (
        sub_stopped["full_dt"].fillna(0) > 2_000
    )
    sub_stopped["time_group"] = sub_stopped["big_timeshift"].astype("int").cumsum()

    for stop_group, d in sub_stopped.groupby("time_group"):
        tstart, tstop = d["UnixTimeMillis"].min(), d["UnixTimeMillis"].max()

        mask_trip_window = (
            (sub["tripId"] == trip)
            & (sub["UnixTimeMillis"] >= tstart)
            & (sub["UnixTimeMillis"] <= tstop)
        )
        stopped_len = int(mask_trip_window.sum())

        if stopped_len >= 20:
            buffer = 800
            mask_trip_buffer = (
                (sub["tripId"] == trip)
                & (sub["UnixTimeMillis"] >= (tstart - buffer))
                & (sub["UnixTimeMillis"] <= (tstop + buffer))
            )

            latDegmean = np.nanmean(
                sub.loc[mask_trip_buffer, "LatitudeDegrees"].to_numpy()
            )
            lngDegmean = np.nanmean(
                sub.loc[mask_trip_buffer, "LongitudeDegrees"].to_numpy()
            )

            if np.isfinite(latDegmean) and np.isfinite(lngDegmean):
                sub.loc[mask_trip_window, "LatitudeDegrees"] = latDegmean
                sub.loc[mask_trip_window, "LongitudeDegrees"] = lngDegmean
                sub.loc[mask_trip_window, "stopped"] = True

sub["stopped"] = sub["stopped"].fillna(False)

sub = add_prev_post_shift(sub)



## === cell 22
pred_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

sub_pred = sub[pred_cols].copy()
sub_pred = sub_pred.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)

for c in ["LatitudeDegrees", "LongitudeDegrees"]:
    sub_pred.loc[~np.isfinite(sub_pred[c].to_numpy()), c] = np.nan

global_lat_med = np.nanmedian(sub_pred["LatitudeDegrees"].to_numpy())
global_lng_med = np.nanmedian(sub_pred["LongitudeDegrees"].to_numpy())


def _fill_trip_nearest(trip_df):
    trip_df = trip_df.sort_values("UnixTimeMillis").copy()
    for col, global_med in [
        ("LatitudeDegrees", global_lat_med),
        ("LongitudeDegrees", global_lng_med),
    ]:
        s = trip_df[col]
        n_valid = int(s.notna().sum())
        if n_valid >= 2:
            x = trip_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
            y = s.to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            trip_df[col] = np.interp(x, x[mask], y[mask])
        elif n_valid == 1:
            trip_df[col] = s.ffill().bfill()
        else:
            trip_df[col] = global_med
    return trip_df


sub_pred = sub_pred.groupby("tripId", group_keys=False).apply(_fill_trip_nearest)

sub_pred["LatitudeDegrees"] = sub_pred.groupby("tripId")["LatitudeDegrees"].transform(
    lambda s: s.ffill().bfill()
)
sub_pred["LongitudeDegrees"] = sub_pred.groupby("tripId")["LongitudeDegrees"].transform(
    lambda s: s.ffill().bfill()
)

sub_pred["LatitudeDegrees"] = sub_pred.groupby("tripId")["LatitudeDegrees"].transform(
    lambda s: s.fillna(s.median())
)
sub_pred["LongitudeDegrees"] = sub_pred.groupby("tripId")["LongitudeDegrees"].transform(
    lambda s: s.fillna(s.median())
)

sub_pred = (
    sub_pred.sort_values(["tripId", "UnixTimeMillis"])
    .groupby(["tripId", "UnixTimeMillis"], as_index=False)
    .agg({"LatitudeDegrees": "median", "LongitudeDegrees": "median"})
)

submission = ss[["tripId", "UnixTimeMillis"]].merge(
    sub_pred, on=["tripId", "UnixTimeMillis"], how="left", validate="one_to_one"
)

for c in ["LatitudeDegrees", "LongitudeDegrees"]:
    submission.loc[~np.isfinite(submission[c].to_numpy()), c] = np.nan

submission = submission.groupby("tripId", group_keys=False).apply(_fill_trip_nearest)

submission["LatitudeDegrees"] = submission.groupby("tripId")[
    "LatitudeDegrees"
].transform(lambda s: s.ffill().bfill())
submission["LongitudeDegrees"] = submission.groupby("tripId")[
    "LongitudeDegrees"
].transform(lambda s: s.ffill().bfill())

submission["LatitudeDegrees"] = submission.groupby("tripId")[
    "LatitudeDegrees"
].transform(lambda s: s.fillna(s.median()))
submission["LongitudeDegrees"] = submission.groupby("tripId")[
    "LongitudeDegrees"
].transform(lambda s: s.fillna(s.median()))

submission["LatitudeDegrees"] = submission["LatitudeDegrees"].fillna(
    submission["LatitudeDegrees"].median()
)
submission["LongitudeDegrees"] = submission["LongitudeDegrees"].fillna(
    submission["LongitudeDegrees"].median()
)

submission = (
    submission.set_index(["tripId", "UnixTimeMillis"])
    .loc[ss.set_index(["tripId", "UnixTimeMillis"]).index]
    .reset_index()
)

submission["LatitudeDegrees"] = submission["LatitudeDegrees"].clip(-90.0, 90.0)
submission["LongitudeDegrees"] = (
    (submission["LongitudeDegrees"] + 180.0) % 360.0
) - 180.0

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
print(
    "Any NaNs:",
    submission[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().to_dict(),
)
print(
    "Any non-finite:",
    (
        ~np.isfinite(submission[["LatitudeDegrees", "LongitudeDegrees"]].to_numpy())
    ).any(),
)
