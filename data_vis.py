# %%
import numpy as np
from data_api import drivers, races

data = np.load('ex1_data.npy')

drivers_reverse = {v: k for k, v in drivers.items()}
races_reverse   = {v: k for k, v in races.items()}

print(f"Total entries  : {data.shape[0]}")
print(f"Unique drivers : {len(np.unique(data[:, 0, 0]))}")
print(f"Unique races   : {len(np.unique(data[:, 0, 1]))}")

unknown_drivers = np.sum(data[:, 0, 0] == -1)
unknown_races   = np.sum(data[:, 0, 1] == -1)
print(f"\nUnknown drivers (-1): {unknown_drivers}")
print(f"Unknown races   (-1): {unknown_races}")

print("\nRaces in dataset:")
for enc in np.unique(data[:, 0, 1]):
    name = races_reverse.get(int(enc), "UNKNOWN")
    count = np.sum(data[:, 0, 1] == enc)
    print(f"  {name:25s} → {count} driver entries")


# %%
import numpy as np
from data_api import drivers, races

data = np.load('ex1_data.npy')

# Get all entries where driver is -1 and find what race they belong to
races_reverse = {v: k for k, v in races.items()}

unknown_mask = data[:, 0, 0] == -1
unknown_entries = data[unknown_mask]

print("Unknown driver entries by race:")
for enc in np.unique(unknown_entries[:, 0, 1]):
    name = races_reverse.get(int(enc), "UNKNOWN")
    count = np.sum(unknown_entries[:, 0, 1] == enc)
    print(f"  {name:25s} → {count} unknown driver entries")
# %%
import fastf1 as ff1
from data_api import drivers

years = [2023, 2024, 2025]
missing = set()

for year in years:
    schedule = ff1.get_event_schedule(year)
    for _, event in schedule.iterrows():
        try:
            session = ff1.get_session(year, event['Location'], 'R')
            session.load(laps=True, telemetry=False, 
                        weather=False, messages=False)
            for d in session.laps['Driver'].unique():
                if d not in drivers:
                    missing.add(d)
        except Exception:
            pass

print("Missing driver codes:", missing)
# %%

import numpy as np
import matplotlib.pyplot as plt

data = np.load('ex1_data.npy')

entry = data[0]                        # first driver-race
laps      = entry[:, 2]               # column 2 = Lap
lap_times = entry[:, 6]               # column 6 = LapTime

valid = laps != -1                     # remove padding rows
plt.plot(laps[valid], lap_times[valid])
plt.xlabel('Lap')
plt.ylabel('Lap time (s)')
plt.title('Lap time over race — entry 0')
plt.show()
# %%

import seaborn as sns

entry = data[0]
cols = ['Driver','Race','Lap','Sector1','Sector2',
        'Sector3','LapTime','Rainfall','TrackTemp','Compound']

valid = entry[:, 2] != -1             # remove padding rows
plt.figure(figsize=(14, 6))
sns.heatmap(entry[valid].T, cmap='coolwarm', yticklabels=cols)
plt.xlabel('Lap number')
plt.title('Feature heatmap — entry 0')
plt.show()

# %%
from data_api import races, drivers

races_reverse   = {v: k for k, v in races.items()}
drivers_reverse = {v: k for k, v in drivers.items()}

race_id = 6                            # Monaco = 6
mask    = data[:, 0, 1] == race_id
race_entries = data[mask]

for entry in race_entries:
    laps      = entry[:, 2]
    lap_times = entry[:, 6]
    driver_id = int(entry[0, 0])
    valid     = laps != -1
    plt.plot(laps[valid], lap_times[valid],
             label=drivers_reverse.get(driver_id, '?'),
             alpha=0.6)

plt.xlabel('Lap')
plt.ylabel('Lap time (s)')
plt.title('Monaco — all drivers lap times')
plt.legend(fontsize=7, ncol=2)
plt.show()
# %%
