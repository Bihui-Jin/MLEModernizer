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
numpy==1.26.4
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

5.215

# 6. Current score

15374.96028

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12788455.72106) has done: 'Your code didn’t yield a Kaggle score mainly because the dataset path and `tripId` formatting are inconsistent with the competition’s expected `tripId` (it should look like `drive_phone`, not `drive/phone`), which can silently produce mostly-empty matches against `sample_submission.csv`. I make the smallest possible changes: (1) fix `INPUT_PATH` to the actual provided location, (2) construct `tripId` using an underscore so it matches the sample submission, and (3) ensure `sub_df` is aligned exactly to `sample_df` ordering (so every required row is present and correctly ordered), which is critical for correct evaluation. The core logic (ECEF->BLH conversion + spline interpolation) is unchanged.'
- What this solution (achieved nan) has done: 'I make the smallest changes needed to (1) use the correct input directory for this environment (so you actually read the train/test files you think you’re reading), and (2) match the competition’s submission key column exactly (`tripId`, not `phone`) so Kaggle can score it correctly. I keep your ECEF→BLH conversion and spline interpolation unchanged, but I also ensure predictions are merged back in the exact same row order as `sample_submission.csv` to avoid any silent misalignment. Finally, I add a tiny safety fix for longitude interpolation around the dateline by unwrapping radians before spline fitting (still the same spline approach, just avoids rare 360° jumps that can explode distance errors).'
- What this solution (achieved nan) has done: 'Your current approach already generates a valid `submission.csv`, but the score is still effectively “nan/invalid” likely because predictions can become NaN/inf (or wildly wrong) for some phones due to ECEF→BLH numerical edge cases and/or spline behavior when the input time series has repeated/ill-conditioned timestamps. I keep your exact core pipeline (WLS ECEF → BLH → spline interpolation → ffill/bfill → fallback) and make only safety/robustness fixes: (1) drop non-finite ECEF rows and enforce float64 to avoid NaNs propagating, (2) make the spline explicitly linear (`k=1`) to reduce overshoot spikes that can explode distance percentiles, and (3) wrap longitudes back into [-180, 180] after interpolation to prevent rare unwrap-induced drift. These changes typically improve stability and should move the score down toward your target without changing the overall method. The output format and ordering remain aligned exactly to `sample_submission.csv`.'
- What this solution (achieved nan) has done: 'I make the smallest changes that address why you’re still getting `nan` (unscored/invalid) while keeping your core pipeline (WLS ECEF → BLH → linear spline interpolation) identical. The main fix is to ensure every `tripId` present in `sample_submission.csv` gets predictions, even if the corresponding folder isn’t found under `test/*/*` (this can otherwise leave rows missing/NaN and lead to invalid scoring). I also harden the per-phone interpolation against remaining degenerate cases by explicitly coercing outputs to finite values and applying the same fallback when a whole phone is missing. Finally, I keep the submission aligned exactly to `sample_submission.csv` order.'
- What this solution (achieved nan) has done: 'I make the smallest changes that address why Kaggle is still returning an unscored/`nan` result: your submission key column is correct (`tripId`), but the per-phone interpolation can still yield non-finite or wildly out-of-range lat/lon for a few timestamps, which can invalidate scoring. I keep your exact pipeline (WLS ECEF → BLH → linear spline interpolation + ffill/bfill + fallback), but add two safety clamps: (1) robustly compute BLH and drop any numerically-bad ECEF points before spline fitting, and (2) force final predictions into valid geographic ranges (lat ∈ [-90, 90], lon wrapped to [-180, 180]) before writing. Finally, I ensure the merge back to `sample_submission.csv` preserves exact row order and remains one-to-one, so every required row is present and aligned.'
- What this solution (achieved nan) has done: 'Your pipeline is already close to producing a scorable file, so the most likely reason for `nan` scoring is a subtle key mismatch (sample uses `tripId`, but your intermediate uses `phone`) and/or duplicated keys after merging that can silently violate Kaggle’s expected one-row-per-(tripId, time). I make minimal, score-relevant hardening changes: (1) enforce exact `(tripId, UnixTimeMillis)` uniqueness and ordering to match `sample_submission.csv`, (2) fix the train-side scoring helper to evaluate per-phone then mean (matching the competition metric more closely, which helps you detect issues locally), and (3) add a tiny fallback for any remaining missing/degenerate phones by using the per-phone median WLS lat/lon computed from that phone’s own test GNSS (no model/logic change). This keeps your core method (WLS ECEF→BLH + linear spline interpolation + ffill/bfill + fallback) identical while ensuring Kaggle can always score the submission and pushing the metric down from “invalid/unscored” toward your 5.215 target.'
- What this solution (achieved nan) has done: 'Your `nan` leaderboard result is almost certainly coming from invalid numeric outputs (NaN/inf) or out-of-domain coordinates that still slip through after merging/filling. I keep your exact core method (WLS ECEF→BLH, linear spline interpolation, ffill/bfill, per-phone fallback) but harden two edge cases that can still break scoring: (1) the BLH conversion can produce non-finite values near the poles/degenerate radii, and (2) your spline is currently only evaluated “in range”, leaving NaNs at the ends that can propagate into bad fallbacks for some phones. The minimal fix is to make `ECEF_to_BLH` numerically safe (avoid division-by-zero and invalid `H`), and to let the linear spline extrapolate (`ext=3`) so every timestamp gets a finite interpolated value before the existing clamps/fallbacks. This should move you from “unscored/nan” to a valid score and typically lowers the error without changing the approach.'
- What this solution (achieved 15374.96028) has done: 'I make the smallest changes needed to eliminate the remaining causes of Kaggle “nan/unscored” while preserving your exact core method (WLS ECEF→BLH, linear spline interpolation, ffill/bfill, per-phone fallback). The main issue is that `InterpolatedUnivariateSpline` can still produce non-finite or huge values when timestamps are ill-conditioned; I add a minimal “seconds-grid pre-aggregation” (median ECEF per second) to ensure strictly well-behaved inputs without changing the modeling approach. I also make the fallback always finite by computing a global fallback from all test GNSS (instead of medians of already-missing phones), and I enforce final numeric sanity (finite + range) right before writing. These changes are directly score-relevant by preventing a few catastrophic tracks from exploding the 95th percentile and by ensuring the submission is always fully valid.'
- What this solution (achieved 15374.96028) has done: 'Your score is far worse than the target (lower is better), so we need a real but still minimal fix that reduces catastrophic per-phone errors. The biggest likely issue in your current pipeline is time-base mismatch: you spline-fit using `utcTimeMillis` from `device_gnss.csv`, but you evaluate on `UnixTimeMillis` from the sample/ground-truth; these are not guaranteed to be the same clock and can introduce huge offsets, making the interpolation essentially wrong. I keep your exact core method (WLS ECEF→BLH + linear spline interpolation + per-second aggregation + ffill/bfill + fallback), but change the spline’s x-axis to the same “rounded-to-second UnixTimeMillis grid” as the submission keys (computed from `utcTimeMillis`), and add a tiny guard to drop any duplicate timestamps before fitting to avoid ill-conditioned splines. This should drastically reduce distance errors while preserving your approach and still finishing within the time limit.'
- What this solution (achieved 15374.96028) has done: 'Your current score is massively worse than the target (lower is better), so the most likely remaining “catastrophic error” is still timestamp misalignment: you aggregate GNSS on a floored Unix-second grid (`UnixTimeMillis`), but the sample/GT timestamps are not always exactly on that grid, so the spline ends up extrapolating and/or mismatching by up to ~1s repeatedly. I keep your exact pipeline (WLS ECEF→BLH + linear spline interpolation + ffill/bfill + fallback), but change the per-second aggregation to round-to-nearest second (instead of floor) and also round the target times to the same grid only for interpolation (while merging back to the original sample timestamps unchanged). Additionally, I add a tiny, score-relevant guard: if after rounding there are still too-few unique timestamps, fall back to the phone median WLS lat/lon (already part of your logic) instead of returning NaNs, which can otherwise create large jumps after ffill/bfill. These are minimal changes that directly address distance explosions and should move the score down sharply toward your target band.'

# 9. Code solution

## === cell 0
import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

INPUT_PATH = "/kaggle/input/smartphone-decimeter-2022"

WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000




## === cell 1
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

    x = np.asarray(ecef.x, dtype=np.float64)
    y = np.asarray(ecef.y, dtype=np.float64)
    z = np.asarray(ecef.z, dtype=np.float64)

    r = np.sqrt(x**2 + y**2)
    r_safe = np.maximum(r, 1e-12)

    t = np.arctan2(z * (a / b), r_safe)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    B = np.arctan2(
        z + (e2_ * b) * (sin_t**3),
        r_safe - (e2 * a) * (cos_t**3),
    )
    L = np.arctan2(y, x)

    sinB = np.sin(B)
    n = a / np.sqrt(np.maximum(1.0 - e2 * (sinB**2), 1e-18))

    cosB = np.cos(B)
    cosB_safe = np.where(np.abs(cosB) < 1e-12, np.sign(cosB) * 1e-12 + 1e-12, cosB)

    H = (r_safe / cosB_safe) - n
    return BLH(lat=B, lng=L, hgt=H)


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




## === cell 2
def _wrap_lng_deg(lng_deg: np.ndarray) -> np.ndarray:
    lng_deg = (lng_deg + 180.0) % 360.0 - 180.0
    return lng_deg


def _clip_lat_deg(lat_deg: np.ndarray) -> np.ndarray:
    return np.clip(lat_deg, -90.0, 90.0)


def _round_unix_to_nearest_second_ms(t_ms: np.ndarray) -> np.ndarray:
    t_ms = np.asarray(t_ms, dtype=np.float64)
    return (np.round(t_ms / 1000.0) * 1000.0).astype(np.int64)


def _robust_prepare_wls_ecef(gnss_df: pd.DataFrame) -> pd.DataFrame:
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    columns = ["utcTimeMillis"] + ecef_columns
    df = gnss_df[columns].copy()
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if df.empty:
        return df

    arr = df[ecef_columns].to_numpy(dtype=np.float64)
    t = df["utcTimeMillis"].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(arr).all(axis=1) & np.isfinite(t)
    df = df.loc[finite_mask].copy()
    if df.empty:
        return df

    df["UnixTimeMillis"] = _round_unix_to_nearest_second_ms(
        df["utcTimeMillis"].to_numpy()
    )

    df = df.groupby("UnixTimeMillis", sort=True, as_index=False)[ecef_columns].median(
        numeric_only=True
    )

    df = df.sort_values("UnixTimeMillis", kind="mergesort").reset_index(drop=True)
    df = df.drop_duplicates(subset=["UnixTimeMillis"], keep="last").reset_index(
        drop=True
    )
    return df


def ecef_to_lat_lng(phone, gnss_df, UnixTimeMillis, fallback_latlng=None):
    ecef_columns = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    ecef_df = _robust_prepare_wls_ecef(gnss_df)

    TIME = (
        ecef_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
        if (not ecef_df.empty and "UnixTimeMillis" in ecef_df.columns)
        else np.array([], dtype=np.float64)
    )

    out = pd.DataFrame({"phone": phone, "UnixTimeMillis": UnixTimeMillis})

    t_eval = _round_unix_to_nearest_second_ms(np.asarray(UnixTimeMillis))

    if TIME.size < 2:
        if (
            fallback_latlng is not None
            and np.isfinite(fallback_latlng[0])
            and np.isfinite(fallback_latlng[1])
        ):
            out["LatitudeDegrees"] = float(fallback_latlng[0])
            out["LongitudeDegrees"] = float(fallback_latlng[1])
        else:
            out["LatitudeDegrees"] = np.nan
            out["LongitudeDegrees"] = np.nan
        return out

    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy(dtype=np.float64))
    blh = ECEF_to_BLH(ecef)

    blh_ok = np.isfinite(blh.lat) & np.isfinite(blh.lng) & np.isfinite(TIME)
    if blh_ok.sum() < 2:
        if (
            fallback_latlng is not None
            and np.isfinite(fallback_latlng[0])
            and np.isfinite(fallback_latlng[1])
        ):
            out["LatitudeDegrees"] = float(fallback_latlng[0])
            out["LongitudeDegrees"] = float(fallback_latlng[1])
        else:
            out["LatitudeDegrees"] = np.nan
            out["LongitudeDegrees"] = np.nan
        return out

    TIME2 = TIME[blh_ok]
    lat2 = blh.lat.astype(np.float64)[blh_ok]
    lng2 = blh.lng.astype(np.float64)[blh_ok]

    order = np.argsort(TIME2, kind="mergesort")
    TIME2 = TIME2[order]
    lat2 = lat2[order]
    lng2 = lng2[order]
    uniq_mask = np.concatenate(([True], np.diff(TIME2) != 0))
    TIME2 = TIME2[uniq_mask]
    lat2 = lat2[uniq_mask]
    lng2 = lng2[uniq_mask]
    if TIME2.size < 2:
        if (
            fallback_latlng is not None
            and np.isfinite(fallback_latlng[0])
            and np.isfinite(fallback_latlng[1])
        ):
            out["LatitudeDegrees"] = float(fallback_latlng[0])
            out["LongitudeDegrees"] = float(fallback_latlng[1])
        else:
            out["LatitudeDegrees"] = np.nan
            out["LongitudeDegrees"] = np.nan
        return out

    lat_spl = InterpolatedUnivariateSpline(TIME2, lat2, k=1, ext=3)
    lng_unwrapped = np.unwrap(lng2)
    lng_spl = InterpolatedUnivariateSpline(TIME2, lng_unwrapped, k=1, ext=3)

    lat = lat_spl(t_eval.astype(np.float64)).astype(np.float64, copy=False)
    lng = lng_spl(t_eval.astype(np.float64)).astype(np.float64, copy=False)

    lat[~np.isfinite(lat)] = np.nan
    lng[~np.isfinite(lng)] = np.nan

    lat_deg = _clip_lat_deg(np.degrees(lat))
    lng_deg = _wrap_lng_deg(np.degrees(lng))

    out["LatitudeDegrees"] = lat_deg
    out["LongitudeDegrees"] = lng_deg
    return out


def calc_score(phone, pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])
    return score


def calc_competition_style_mean_score(
    pred_all: pd.DataFrame, gt_all: pd.DataFrame
) -> float:
    merged = gt_all.merge(
        pred_all,
        on=["phone", "UnixTimeMillis"],
        how="inner",
        validate="one_to_one",
        suffixes=("_gt", "_pred"),
    )
    if merged.empty:
        return np.nan

    def _phone_score(g):
        df_pred = g[["LatitudeDegrees_pred", "LongitudeDegrees_pred"]].rename(
            columns={
                "LatitudeDegrees_pred": "LatitudeDegrees",
                "LongitudeDegrees_pred": "LongitudeDegrees",
            }
        )
        df_gt = g[["LatitudeDegrees_gt", "LongitudeDegrees_gt"]].rename(
            columns={
                "LatitudeDegrees_gt": "LatitudeDegrees",
                "LongitudeDegrees_gt": "LongitudeDegrees",
            }
        )
        d = pandas_haversine_distance(df_pred, df_gt)
        return float(np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)]))

    return float(merged.groupby("phone", sort=False).apply(_phone_score).mean())




## === cell 3
score_list = []
pred_all_train = []
gt_all_train = []

for dirname in sorted(glob.glob(f"{INPUT_PATH}/train/*/*")):
    drive, phone_name = dirname.split("/")[-2:]
    phone = f"{drive}_{phone_name}"
    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
    gt_df = pd.read_csv(f"{dirname}/ground_truth.csv")
    gt_df = (
        gt_df.rename(columns={"tripId": "phone"})
        if "tripId" in gt_df.columns
        else gt_df
    )
    gt_df["phone"] = phone

    pred_df = ecef_to_lat_lng(phone, gnss_df, gt_df["UnixTimeMillis"].to_numpy())
    score = calc_score(phone, pred_df, gt_df)
    print(f"{phone:<45}: score = {score:.3f}")
    score_list.append(score)

    pred_all_train.append(pred_df)
    gt_all_train.append(
        gt_df[["phone", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
    )



## === cell 4
mean_score = np.mean(score_list) if len(score_list) else np.nan
print(f"mean_score (per-phone simple mean of per-drive scores) = {mean_score:.3f}")

try:
    pred_all_train_df = pd.concat(pred_all_train, ignore_index=True)
    gt_all_train_df = pd.concat(gt_all_train, ignore_index=True)
    comp_style = calc_competition_style_mean_score(pred_all_train_df, gt_all_train_df)
    print(f"mean_score (competition-style mean across phones) = {comp_style:.3f}")
except Exception as e:
    print("Competition-style score check skipped due to:", repr(e))



## === cell 5
sample_df = pd.read_csv(f"{INPUT_PATH}/sample_submission.csv")

if "tripId" not in sample_df.columns:
    raise ValueError(
        f"sample_submission.csv must contain tripId. Got columns: {list(sample_df.columns)}"
    )

sample_df = sample_df.copy()
if sample_df.duplicated(subset=["tripId", "UnixTimeMillis"]).any():
    raise ValueError(
        "sample_submission has duplicated (tripId, UnixTimeMillis) keys unexpectedly."
    )

sample_df_internal = sample_df.rename(columns={"tripId": "phone"})


def phone_wls_median_latlng(gnss_df):
    ecef_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    tmp = gnss_df[ecef_cols].copy()
    for c in ecef_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna()
    if tmp.shape[0] < 1:
        return (np.nan, np.nan)
    arr = tmp.to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] < 1:
        return (np.nan, np.nan)
    blh = ECEF_to_BLH(ECEF.from_numpy(arr))
    lat = np.degrees(np.nanmedian(blh.lat))
    lng = np.degrees(np.nanmedian(blh.lng))
    lng = _wrap_lng_deg(lng)
    lat = float(_clip_lat_deg(np.array([lat], dtype=np.float64))[0])
    if not np.isfinite(lat) or not np.isfinite(lng):
        return (np.nan, np.nan)
    return (lat, lng)


test_dirs = {
    f"{d.split('/')[-2]}_{d.split('/')[-1]}": d
    for d in glob.glob(f"{INPUT_PATH}/test/*/*")
}
phones_in_sample = sample_df_internal["phone"].unique().tolist()

global_fallback_latlng = (np.nan, np.nan)
try:
    lat_list = []
    lng_list = []
    for d in list(test_dirs.values())[:2000]:
        try:
            gnss_tmp = pd.read_csv(
                f"{d}/device_gnss.csv",
                usecols=[
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ],
            )
        except Exception:
            continue
        latlng = phone_wls_median_latlng(gnss_tmp)
        if np.isfinite(latlng[0]) and np.isfinite(latlng[1]):
            lat_list.append(latlng[0])
            lng_list.append(latlng[1])
    if len(lat_list) > 0:
        global_fallback_latlng = (
            float(np.median(lat_list)),
            float(np.median(lng_list)),
        )
except Exception:
    global_fallback_latlng = (np.nan, np.nan)

if not (
    np.isfinite(global_fallback_latlng[0]) and np.isfinite(global_fallback_latlng[1])
):
    global_fallback_latlng = (37.4, -122.1)

pred_dfs = []
fallback_map = {}

for phone in tqdm(phones_in_sample):
    dirname = test_dirs.get(phone, None)

    UnixTimeMillis = sample_df_internal.loc[
        sample_df_internal["phone"] == phone, "UnixTimeMillis"
    ].to_numpy()

    if dirname is None:
        pred_dfs.append(
            pd.DataFrame(
                {
                    "phone": phone,
                    "UnixTimeMillis": UnixTimeMillis,
                    "LatitudeDegrees": np.nan,
                    "LongitudeDegrees": np.nan,
                }
            )
        )
        fallback_map[phone] = global_fallback_latlng
        continue

    gnss_df = pd.read_csv(f"{dirname}/device_gnss.csv")
    fb = phone_wls_median_latlng(gnss_df)
    fallback_map[phone] = (
        fb if (np.isfinite(fb[0]) and np.isfinite(fb[1])) else global_fallback_latlng
    )

    pred_dfs.append(
        ecef_to_lat_lng(
            phone, gnss_df, UnixTimeMillis, fallback_latlng=fallback_map[phone]
        )
    )

pred_all = pd.concat(pred_dfs, ignore_index=True)

if pred_all.duplicated(subset=["phone", "UnixTimeMillis"]).any():
    pred_all = pred_all.drop_duplicates(
        subset=["phone", "UnixTimeMillis"], keep="last"
    ).reset_index(drop=True)

sub_df = sample_df_internal[["phone", "UnixTimeMillis"]].merge(
    pred_all,
    on=["phone", "UnixTimeMillis"],
    how="left",
    validate="one_to_one",
    sort=False,
)

sub_df["LatitudeDegrees"] = sub_df.groupby("phone", sort=False)[
    "LatitudeDegrees"
].transform(lambda s: s.ffill().bfill())
sub_df["LongitudeDegrees"] = sub_df.groupby("phone", sort=False)[
    "LongitudeDegrees"
].transform(lambda s: s.ffill().bfill())

sub_df.loc[
    ~np.isfinite(sub_df["LatitudeDegrees"].to_numpy(dtype=np.float64)),
    "LatitudeDegrees",
] = np.nan
sub_df.loc[
    ~np.isfinite(sub_df["LongitudeDegrees"].to_numpy(dtype=np.float64)),
    "LongitudeDegrees",
] = np.nan

mask_lat = sub_df["LatitudeDegrees"].isna()
mask_lng = sub_df["LongitudeDegrees"].isna()
if mask_lat.any() or mask_lng.any():
    fb_lat = sub_df["phone"].map(
        lambda p: fallback_map.get(p, global_fallback_latlng)[0]
    )
    fb_lng = sub_df["phone"].map(
        lambda p: fallback_map.get(p, global_fallback_latlng)[1]
    )
    sub_df.loc[mask_lat, "LatitudeDegrees"] = fb_lat[mask_lat].to_numpy()
    sub_df.loc[mask_lng, "LongitudeDegrees"] = fb_lng[mask_lng].to_numpy()

sub_df["LatitudeDegrees"] = pd.to_numeric(sub_df["LatitudeDegrees"], errors="coerce")
sub_df["LongitudeDegrees"] = pd.to_numeric(sub_df["LongitudeDegrees"], errors="coerce")
sub_df.loc[
    ~np.isfinite(sub_df["LatitudeDegrees"].to_numpy(dtype=np.float64)),
    "LatitudeDegrees",
] = global_fallback_latlng[0]
sub_df.loc[
    ~np.isfinite(sub_df["LongitudeDegrees"].to_numpy(dtype=np.float64)),
    "LongitudeDegrees",
] = global_fallback_latlng[1]

sub_df["LatitudeDegrees"] = _clip_lat_deg(
    sub_df["LatitudeDegrees"].to_numpy(dtype=np.float64)
)
sub_df["LongitudeDegrees"] = _wrap_lng_deg(
    sub_df["LongitudeDegrees"].to_numpy(dtype=np.float64)
)

sub_df = sub_df.rename(columns={"phone": "tripId"})
sub_df = sub_df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

if len(sub_df) != len(sample_df):
    raise ValueError(
        f"Row count mismatch vs sample: sub={len(sub_df)} sample={len(sample_df)}"
    )

if sub_df.duplicated(subset=["tripId", "UnixTimeMillis"]).any():
    raise ValueError("Submission has duplicated (tripId, UnixTimeMillis) keys.")

sub_df = sample_df[["tripId", "UnixTimeMillis"]].merge(
    sub_df,
    on=["tripId", "UnixTimeMillis"],
    how="left",
    validate="one_to_one",
    sort=False,
)

if sub_df[["LatitudeDegrees", "LongitudeDegrees"]].isna().any().any():
    sub_df["LatitudeDegrees"] = sub_df["LatitudeDegrees"].fillna(
        global_fallback_latlng[0]
    )
    sub_df["LongitudeDegrees"] = sub_df["LongitudeDegrees"].fillna(
        global_fallback_latlng[1]
    )

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("Any NaNs left:", sub_df.isna().any().to_dict())
print("Row count matches sample:", len(sub_df) == len(sample_df))
print("Global fallback used (lat, lng):", global_fallback_latlng)
