"""Bike Sharing Dataset EDA - 生成数据探索报告"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
day = pd.read_csv(BASE / "day.csv")
hour = pd.read_csv(BASE / "hour.csv")

col_meaning = {
    "instant": "记录索引",
    "dteday": "日期",
    "season": "季节 (1:春, 2:夏, 3:秋, 4:冬)",
    "yr": "年份 (0:2011, 1:2012)",
    "mnth": "月份 (1-12)",
    "hr": "小时 (0-23, 仅 hour.csv)",
    "holiday": "是否节假日 (0/1)",
    "weekday": "星期几 (0-6)",
    "workingday": "是否工作日 (1:是, 0:否)",
    "weathersit": "天气情况 (1:晴, 2:多云/雾, 3:小雨/小雪, 4:暴雨/暴雪)",
    "temp": "归一化温度 (除以41)",
    "atemp": "归一化体感温度 (除以50)",
    "hum": "归一化湿度 (除以100)",
    "windspeed": "归一化风速 (除以67)",
    "casual": "临时用户租车数",
    "registered": "注册用户租车数",
    "cnt": "总租车数 (目标变量) = casual + registered",
}

categorical = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
numerical = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]

def report(df, name):
    lines = [f"## {name}.csv\n"]
    lines.append(f"- **形状**: {df.shape[0]} 行 × {df.shape[1]} 列\n")
    lines.append(f"- **列名**: {list(df.columns)}\n")
    lines.append("\n### 列含义\n")
    lines.append("| 列名 | 含义 |\n|---|---|")
    for c in df.columns:
        lines.append(f"| `{c}` | {col_meaning.get(c, '-')} |")
    miss = df.isnull().sum().sum()
    dup = df.duplicated().sum()
    lines.append(f"\n### 缺失值与重复值\n- 缺失值总数: **{miss}**\n- 完全重复行数: **{dup}**\n")

    lines.append("### 目标变量 cnt 分布\n")
    desc = df["cnt"].describe()
    lines.append("```\n" + desc.to_string() + "\n```\n")
    lines.append(f"- 偏度 (skew): {df['cnt'].skew():.4f}")
    lines.append(f"- 峰度 (kurt): {df['cnt'].kurt():.4f}\n")

    cats = [c for c in categorical if c in df.columns]
    nums = [c for c in numerical if c in df.columns]
    lines.append(f"### 特征类型\n- **类别型** ({len(cats)}): {cats}\n- **数值型** ({len(nums)}): {nums}\n")

    lines.append("### 不同分组下 cnt 平均值\n")
    for g in ["season", "weathersit", "mnth", "weekday", "workingday", "yr"]:
        if g in df.columns:
            grp = df.groupby(g)["cnt"].agg(["mean", "median", "count"]).round(2)
            lines.append(f"\n**按 `{g}` 分组**\n```\n{grp.to_string()}\n```")
    if "hr" in df.columns:
        grp = df.groupby("hr")["cnt"].mean().round(2)
        lines.append(f"\n**按 `hr` 分组 (平均值)**\n```\n{grp.to_string()}\n```")
    return "\n".join(lines)

out = ["# 共享单车数据集 EDA 报告\n",
       "数据来源: Capital Bikeshare (Washington D.C., 2011-2012)\n"]
out.append(report(day, "day"))
out.append("\n---\n")
out.append(report(hour, "hour"))

(BASE / "EDA_报告.md").write_text("\n".join(out), encoding="utf-8")
print("EDA 报告已生成: EDA_报告.md")
print(f"day.csv: {day.shape}, hour.csv: {hour.shape}")
print(f"day 缺失={day.isnull().sum().sum()}, 重复={day.duplicated().sum()}")
print(f"hour 缺失={hour.isnull().sum().sum()}, 重复={hour.duplicated().sum()}")
