# %%
import fastf1 as ff1
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from fastf1.exceptions import RateLimitExceededError

ff1.Cache.enable_cache('cache')

# -------------------------------
# Encoding dictionaries
# -------------------------------
# These should map string names to integer values.
# Extend as needed based on your actual drivers, races, and compounds.

drivers = {
    'VER': 1, 'HAM': 2, 'LEC': 3, 'SAI': 4, 'PER': 5,
    'RUS': 6, 'NOR': 7, 'ALO': 8, 'OCO': 9, 'STR': 10,
    'GAS': 11, 'VET': 12, 'BOT': 13, 'MSC': 14, 'MAG': 15,
    'TSU': 16, 'RIC': 17, 'ZHO': 18, 'LAT': 19, 'ALB': 20,
    'HUL': 21, 'DEV': 22, 'SAR': 23, 'PIA': 24, 'LAW': 25,
    # 2025 additions
    'ANT': 26, 'BOR': 27, 'HAD': 28, 'DOO': 29, 'BEA': 30,
     # 2024 replacement driver
    'COL': 31,
}

races = {
    'Bahrain': 1, 'Jeddah': 2, 'Melbourne': 3, 'Baku': 4,
    'Miami': 5, 'Monaco': 6, 'Barcelona': 7, 'Montreal': 8,
    'Silverstone': 9, 'Spielberg': 10, 'Budapest': 11,
    'Francorchamps': 12, 'Zandvoort': 13, 'Monza': 14,
    'Singapore': 15, 'Suzuka': 16, 'Lusail': 17, 'Austin': 18,
    'Mexico City': 19, 'São Paulo': 20, 'Las Vegas': 21,
    'Abu Dhabi': 22, 'Imola': 23, 'Sakhir': 24, 'Portimao': 25,
    'Istanbul': 26, 'Mugello': 27, 'Nuerburg': 28, 'Bahrain2': 29,
    'Yas Island': 30, 'Montmelo': 31, 'Hungaroring': 32,
    # exact names as returned by FastF1
    'Montréal': 33, 'Spa-Francorchamps': 34, 'Marina Bay': 35,
    'Shanghai': 36, 'Miami Gardens': 37,
}

compound = {
    'SOFT': 1, 'MEDIUM': 2, 'HARD': 3,
    'INTERMEDIATE': 4, 'WET': 5,
}

# Monaco Grand Prix has the most laps — used as fixed sequence length
MAX_LAPS = 78


# -------------------------------
# Utility: Convert timedelta → seconds
# -------------------------------
def timedelta_to_seconds(td):
    if pd.isna(td):
        return np.nan
    return td.total_seconds()


# -------------------------------
# Get race list for a year
# -------------------------------
def get_race_list(year):
    # Retry up to 3 times if rate limit is hit while fetching the schedule
    for attempt in range(3):
        try:
            schedule = ff1.get_event_schedule(year)
            break
        except RateLimitExceededError:
            wait = 60 * (attempt + 1)   # wait 60s, then 120s, then 180s
            print(f"Rate limit hit fetching schedule for {year}. "
                  f"Waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)
    else:
        print(f"Could not fetch schedule for {year} after 3 attempts. Skipping.")
        return []

    race_list = list(schedule['Location'])

    # Remove pre-season testing sessions
    if year == 2025:
        while 'Bahrain' in race_list:
            race_list.remove('Bahrain')
        while 'Sakhir' in race_list:
            race_list.remove('Sakhir')

    elif year == 2024:
        while 'Bahrain' in race_list:
            race_list.remove('Bahrain')
        while 'Sakhir' in race_list:
            race_list.remove('Sakhir')

    elif year == 2023:
        while 'Bahrain' in race_list:
            race_list.remove('Bahrain')
        while 'Sakhir' in race_list:
            race_list.remove('Sakhir')

    elif year == 2022:
        race_list.remove('Spain')
        race_list.remove('Bahrain')

    elif year == 2021:
        race_list.remove('Sakhir')

    elif year == 2020:
        race_list.remove('Montmelo')
        race_list.remove('Montmelo')

    return race_list


# -------------------------------
# Core: Create dataframe per driver-session
# -------------------------------
def get_data(driver, session):
    # FIX 1: pick_drivers (plural) replaces the deprecated pick_driver
    session_driver = session.laps.pick_drivers(driver)

    # Reset index so all series are aligned 0, 1, 2, ...
    session_driver = session_driver.reset_index(drop=True)

    n_laps = len(session_driver)

    driver_lap_number   = session_driver['LapNumber']
    driver_sector1_time = (session_driver['Sector1Time'] / np.timedelta64(1, 's')).astype(float)
    driver_sector2_time = (session_driver['Sector2Time'] / np.timedelta64(1, 's')).astype(float)
    driver_sector3_time = (session_driver['Sector3Time'] / np.timedelta64(1, 's')).astype(float)
    driver_lap_time     = session_driver['LapTime'].apply(timedelta_to_seconds)

    # FIX 2: get_weather_data() returns one row per lap of the FULL session,
    # not per driver. Trim and reset its index to match the driver's lap count.
    weather    = session.laps.get_weather_data().reset_index(drop=True).iloc[:n_laps]
    rainfall   = np.where(weather['Rainfall'] == True, 1, 0)
    track_temp = weather['TrackTemp'].reset_index(drop=True)

    driver_list  = [driver] * n_laps
    race_list    = [session.event['Location']] * n_laps
    compound_col = session_driver['Compound'].reset_index(drop=True)

    df = pd.DataFrame({
        'Driver':    driver_list,
        'Race':      race_list,
        'Lap':       driver_lap_number.values,
        'Sector1':   driver_sector1_time.values,
        'Sector2':   driver_sector2_time.values,
        'Sector3':   driver_sector3_time.values,
        'LapTime':   driver_lap_time.values,
        'Rainfall':  rainfall,
        'TrackTemp': track_temp.values,
        'Compound':  compound_col.values
    })

    return df


# -------------------------------
# Load dataset (ALL drivers, ALL races)
# -------------------------------
def load_dataset(year_list):
    driver_race_data_list = []

    for year in year_list:
        race_list = get_race_list(year)
        driver_race_data = {}

        for race in race_list:
            # Retry each race up to 3 times to handle transient rate limit errors
            for attempt in range(3):
                try:
                    session = ff1.get_session(year, race, 'R')
                    session.load()

                    driver_list = pd.unique(session.laps['Driver'])

                    for driver in driver_list:
                        driver_encoded = drivers.get(driver, -1)
                        race_encoded   = races.get(race, -1)

                        data = get_data(driver, session)

                        data['Driver'] = driver_encoded
                        data['Race']   = race_encoded

                        data['Compound'] = data['Compound'].map(
                            lambda c: compound.get(str(c), -1) if pd.notna(c) else -1
                        )

                        data = data.fillna(-1)
                        arr  = data.values

                        while arr.shape[0] < MAX_LAPS:
                            lap_num = arr.shape[0] + 1
                            new_row = np.array([[driver_encoded, race_encoded,
                                                 lap_num, -1, -1, -1, -1, -1, -1, -1]])
                            arr = np.vstack((arr, new_row))

                        driver_race_data[(driver_encoded, race_encoded)] = arr

                    break   # success — move on to next race

                except RateLimitExceededError:
                    wait = 60 * (attempt + 1)   # 60s, 120s, 180s
                    print(f"Rate limit hit on {race} {year}. "
                          f"Waiting {wait}s before retry {attempt + 1}/3...")
                    time.sleep(wait)

                except Exception as e:
                    print(f"Skipping {race} {year}: {e}")
                    break   # non-rate-limit error — skip this race

        driver_race_data_list.append(driver_race_data)

    return driver_race_data_list


# -------------------------------
# Convert to 3D dataset (N × T × n)
# -------------------------------
def generate_dataset(year_list):

    driver_race_data_list = load_dataset(year_list)

    # Find shape from first entry
    first = next(iter(driver_race_data_list[0].values()))
    m, n = first.shape

    N = sum(len(d) for d in driver_race_data_list)

    full_dataset = np.zeros((N, m, n))

    i = 0
    for dataset in driver_race_data_list:
        for key, value in dataset.items():
            full_dataset[i] = value
            i += 1

    np.save('ex1_data.npy', full_dataset)

    return full_dataset


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    years = [2023,2024,2025]

    data = generate_dataset(years)

    print("Dataset shape:", data.shape)