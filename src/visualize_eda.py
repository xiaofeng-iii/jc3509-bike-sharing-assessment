"""Visualization analysis: cnt distribution, holiday/weekday vs user types, weather factors vs cnt"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

HERE = Path(__file__).parent
DATA = HERE.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

day = pd.read_csv(DATA / "day.csv")
hour = pd.read_csv(DATA / "hour.csv")

# ---------- 1. cnt distribution ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(day['cnt'], bins=40, color='steelblue', edgecolor='white')
axes[0, 0].set_title('day.csv - cnt Histogram')
axes[0, 0].set_xlabel('cnt'); axes[0, 0].set_ylabel('Frequency')

axes[0, 1].boxplot(day['cnt'], vert=False)
axes[0, 1].set_title('day.csv - cnt Boxplot')
axes[0, 1].set_xlabel('cnt')

axes[1, 0].hist(hour['cnt'], bins=60, color='darkorange', edgecolor='white')
axes[1, 0].set_title('hour.csv - cnt Histogram')
axes[1, 0].set_xlabel('cnt'); axes[1, 0].set_ylabel('Frequency')

axes[1, 1].boxplot(hour['cnt'], vert=False)
axes[1, 1].set_title('hour.csv - cnt Boxplot')
axes[1, 1].set_xlabel('cnt')

plt.tight_layout()
plt.savefig(FIG / "01_cnt_distribution.png", dpi=120)
plt.close()

# ---------- 2. holiday vs casual/registered ----------
holiday_grp = day.groupby('holiday')[['casual', 'registered']].mean()
fig, ax = plt.subplots(figsize=(8, 5))
x = ['Non-holiday (0)', 'Holiday (1)']
width = 0.35
idx = range(len(x))
ax.bar([i - width/2 for i in idx], holiday_grp['casual'], width, label='casual', color='coral')
ax.bar([i + width/2 for i in idx], holiday_grp['registered'], width, label='registered', color='steelblue')
ax.set_xticks(list(idx)); ax.set_xticklabels(x)
ax.set_ylabel('Average Rental Count')
ax.set_title('Holiday vs User Type - Average Rental Count (day.csv)')
ax.legend()
for i, (c, r) in enumerate(zip(holiday_grp['casual'], holiday_grp['registered'])):
    ax.text(i - width/2, c, f'{c:.0f}', ha='center', va='bottom')
    ax.text(i + width/2, r, f'{r:.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(FIG / "02_holiday_vs_users.png", dpi=120)
plt.close()

# ---------- 3. weekday vs casual/registered ----------
wk_grp = day.groupby('weekday')[['casual', 'registered']].mean()
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
idx = range(7); width = 0.4
ax.bar([i - width/2 for i in idx], wk_grp['casual'], width, label='casual', color='coral')
ax.bar([i + width/2 for i in idx], wk_grp['registered'], width, label='registered', color='steelblue')
ax.set_xticks(list(idx)); ax.set_xticklabels(labels)
ax.set_ylabel('Average Rental Count')
ax.set_title('Weekday vs User Type - Average Rental Count (day.csv)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "03_weekday_vs_users.png", dpi=120)
plt.close()

# ---------- 4. Weather factors vs cnt (scatter + trend) ----------
import numpy as np
features = [('temp', 'Normalized Temperature', 41), ('atemp', 'Normalized Apparent Temperature', 50),
            ('hum', 'Normalized Humidity', 100), ('windspeed', 'Normalized Wind Speed', 67)]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, label, scale) in zip(axes.flat, features):
    ax.scatter(day[col], day['cnt'], alpha=0.4, s=15, color='teal')
    # Linear fit
    z = np.polyfit(day[col], day['cnt'], 1)
    xs = np.linspace(day[col].min(), day[col].max(), 100)
    ax.plot(xs, np.polyval(z, xs), color='red', lw=2, label=f'Linear Fit')
    corr = day[col].corr(day['cnt'])
    ax.set_title(f'{label} vs cnt (correlation={corr:.3f})')
    ax.set_xlabel(f'{col} (×{scale} to restore original value)')
    ax.set_ylabel('cnt')
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "04_weather_vs_cnt.png", dpi=120)
plt.close()

print("Generated 4 figures to:", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" -", p.name)
