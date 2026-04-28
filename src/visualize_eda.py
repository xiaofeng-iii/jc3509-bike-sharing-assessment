"""可视化分析: cnt 分布、节假日/weekday 与用户类型关系、气象因素与 cnt 关系"""
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

# ---------- 1. cnt 分布 ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(day['cnt'], bins=40, color='steelblue', edgecolor='white')
axes[0, 0].set_title('day.csv - cnt 直方图')
axes[0, 0].set_xlabel('cnt'); axes[0, 0].set_ylabel('频数')

axes[0, 1].boxplot(day['cnt'], vert=False)
axes[0, 1].set_title('day.csv - cnt 箱线图')
axes[0, 1].set_xlabel('cnt')

axes[1, 0].hist(hour['cnt'], bins=60, color='darkorange', edgecolor='white')
axes[1, 0].set_title('hour.csv - cnt 直方图')
axes[1, 0].set_xlabel('cnt'); axes[1, 0].set_ylabel('频数')

axes[1, 1].boxplot(hour['cnt'], vert=False)
axes[1, 1].set_title('hour.csv - cnt 箱线图')
axes[1, 1].set_xlabel('cnt')

plt.tight_layout()
plt.savefig(FIG / "01_cnt_distribution.png", dpi=120)
plt.close()

# ---------- 2. holiday vs casual/registered ----------
holiday_grp = day.groupby('holiday')[['casual', 'registered']].mean()
fig, ax = plt.subplots(figsize=(8, 5))
x = ['非节假日 (0)', '节假日 (1)']
width = 0.35
idx = range(len(x))
ax.bar([i - width/2 for i in idx], holiday_grp['casual'], width, label='casual', color='coral')
ax.bar([i + width/2 for i in idx], holiday_grp['registered'], width, label='registered', color='steelblue')
ax.set_xticks(list(idx)); ax.set_xticklabels(x)
ax.set_ylabel('平均租车数')
ax.set_title('节假日 vs 用户类型 平均租车数 (day.csv)')
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
labels = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
idx = range(7); width = 0.4
ax.bar([i - width/2 for i in idx], wk_grp['casual'], width, label='casual', color='coral')
ax.bar([i + width/2 for i in idx], wk_grp['registered'], width, label='registered', color='steelblue')
ax.set_xticks(list(idx)); ax.set_xticklabels(labels)
ax.set_ylabel('平均租车数')
ax.set_title('一周中各天 vs 用户类型 平均租车数 (day.csv)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "03_weekday_vs_users.png", dpi=120)
plt.close()

# ---------- 4. 气象因素 vs cnt (散点 + 趋势) ----------
import numpy as np
features = [('temp', '归一化温度', 41), ('atemp', '归一化体感温度', 50),
            ('hum', '归一化湿度', 100), ('windspeed', '归一化风速', 67)]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, label, scale) in zip(axes.flat, features):
    ax.scatter(day[col], day['cnt'], alpha=0.4, s=15, color='teal')
    # 拟合线
    z = np.polyfit(day[col], day['cnt'], 1)
    xs = np.linspace(day[col].min(), day[col].max(), 100)
    ax.plot(xs, np.polyval(z, xs), color='red', lw=2, label=f'线性拟合')
    corr = day[col].corr(day['cnt'])
    ax.set_title(f'{label} vs cnt (相关系数={corr:.3f})')
    ax.set_xlabel(f'{col} (×{scale} 还原原值)')
    ax.set_ylabel('cnt')
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "04_weather_vs_cnt.png", dpi=120)
plt.close()

print("已生成 4 张图至:", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" -", p.name)
