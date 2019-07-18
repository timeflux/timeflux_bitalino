import pandas as pd

df = pd.read_hdf('20190718-103741.hdf5', 'offsets')
first = df.index.values[0]
last = df.index.values[-1]
total_seconds = round((last - first).astype(int) / 1e9)
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f'Total running duration: {hours} hours, {minutes} minutes, {seconds} seconds')
delta = pd.Timedelta(3, 'm')
offset_start = df.loc[first:first + delta]['time_offset'].mean()
offset_stop = df.loc[last - delta:last]['time_offset'].mean()
drift_us = int(offset_stop - offset_start)
drift_s = round((offset_stop - offset_start) / 1e6)
drift_percent = offset_start * 100 / offset_stop
print(f'Total drift (us): {drift_us}')
print(f'Total drift (s): {drift_s}')
