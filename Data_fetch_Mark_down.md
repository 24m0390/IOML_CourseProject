# 🏎️ Formula 1 3D Dataset Generation using FastF1 API

## 1. Overview

This project builds a **3D dataset** from Formula 1 race telemetry using the FastF1 API.

The final dataset is a tensor of shape:

[
(N \times T \times n)
]

Where:

* **N** → Total number of (driver, race) combinations
* **T** → Number of laps per race (fixed to 78)
* **n** → Number of features per lap

---

## 2. Data Source

The data is fetched using the FastF1 API, which provides access to:

* Lap timings
* Sector times
* Weather data
* Tire compounds

Caching is enabled to improve performance:

```python
ff1.Cache.enable_cache('cache')
```

---

## 3. Feature Description

Each lap is represented by the following features:

| Feature   | Description                    |
| --------- | ------------------------------ |
| Driver    | Encoded driver ID              |
| Race      | Encoded race ID                |
| Lap       | Lap number                     |
| Sector1   | Sector 1 time (seconds)        |
| Sector2   | Sector 2 time (seconds)        |
| Sector3   | Sector 3 time (seconds)        |
| LapTime   | Total lap time (seconds)       |
| Rainfall  | Binary (1 = rain, 0 = no rain) |
| TrackTemp | Track temperature              |
| Compound  | Tire compound (encoded)        |

---

## 4. Encoding Strategy

Categorical variables are converted into integers:

### Drivers

```python
'VER' → 1, 'HAM' → 2, ...
```

### Races

```python
'Monaco' → 6, 'Silverstone' → 9, ...
```

### Tire Compounds

```python
SOFT → 1, MEDIUM → 2, HARD → 3
```

Unknown values are encoded as:

```python
-1
```

---

## 5. Data Fetching Pipeline

### Step 1: Get Race Schedule

```python
schedule = ff1.get_event_schedule(year)
```

* Extract race locations
* Remove pre-season testing events

---

### Step 2: Load Race Session

```python
session = ff1.get_session(year, race, 'R')
session.load()
```

* Loads full race lap data

---

### Step 3: Extract Driver Data

For each driver:

```python
session.laps.pick_drivers(driver)
```

Extract:

* Lap times
* Sector times
* Tire compounds

---

### Step 4: Add Weather Data

Weather is aligned per lap:

```python
weather = session.laps.get_weather_data()
```

Derived features:

* Rainfall → binary encoding
* Track temperature

---

### Step 5: Build DataFrame

Each driver-race pair is converted into a structured table:

[
T \times n
]

---

### Step 6: Handle Missing Data

* Missing values → `-1`
* If laps < 78 → pad with dummy rows

```python
while arr.shape[0] < MAX_LAPS:
    arr = np.vstack(...)
```

---

## 6. Constructing the 3D Dataset

Each (driver, race) → 2D matrix:

[
(78 \times n)
]

All matrices are stacked into:

[
(N \times 78 \times n)
]

```python
full_dataset = np.zeros((N, m, n))
```

---

## 7. Output

The dataset is saved as a NumPy file:

```python
np.save('ex1_data.npy', full_dataset)
```

### Example Shape

```
Dataset shape: (N, 78, 10)
```

---

## 8. Key Design Decisions

### Fixed Sequence Length

* All races padded to **78 laps**
* Ensures compatibility with ML models

### Numerical Encoding

* Converts categorical → numeric
* Required for machine learning

### Missing Data Handling

* Uses `-1` as placeholder

---

## 9. Challenges

* API rate limits → handled via retries
* Inconsistent lap counts across races
* Missing telemetry values
* Nested API structure

---

## 10. Final Pipeline

```text
FastF1 API
    ↓
Race Schedule
    ↓
Session Data
    ↓
Driver-wise Extraction
    ↓
Feature Engineering
    ↓
Padding (78 laps)
    ↓
3D Tensor (N × T × n)
    ↓
Saved as .npy
```

---

## 11. Use Cases

* Lap time prediction
* Race strategy optimization
* Reinforcement learning models
* Driver performance analysis

---

🚀 This pipeline transforms raw Formula 1 telemetry into a structured 3D dataset ready for machine learning and analysis.
