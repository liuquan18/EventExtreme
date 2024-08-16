![logo](logo.png)

# Overview
Extracting extreme events from daily time series data.


Such extreme events are defined as the values exceeding a certain threshold, and lasting for a certain duration.

Extremes events are identified by attributes such as:

    - extreme_start_time
    - extreme_end_time
    - extreme_duration
    - sign_start_time
    - sign_end_time
    - sign_duration
    - max
    - min
    - mean

![plot](plot.png)

The algorithm has some robust features, such as:

(P: positive sign, N: negative sign, E: extreme value)

    - single opposite valued data are ignored. e.g, P,P,P,N,P,P,P -> P,P,P,P,P,P
    - Multiple extreme events within one sign event are considered as one event. e.g, P,P,P,E,P,E,P,P,P is considered as one event.

# Installation
```bash
pip install eventextreme
```
