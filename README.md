# Global Weather AI — 全球城市次日气温预测

这是一个可以在 GitHub Actions 免费运行的 **Tmax/Tmin 预测基线**。项目目前覆盖 22 个全球城市，通过 Open-Meteo ECMWF 预报接口获取各城市当地 **明天** 的最高和最低气温。

> 当前还没有加入 Kaggle 训练、LightGBM 误差校正、飞机温度观测，也没有完成真实预报准确性评估。成功获取数据不等于预测准确。

## 开始使用

1. 打开 [Actions](https://github.com/Wuzzzzzzz/global-weather-ai/actions)。
2. 选择 **Daily city temperature forecast**。
3. 点击 **Run workflow**，选择 main 分支后运行。
4. 运行完成后，打开该次运行，在 **Artifacts** 下载 `forecast-<run_id>`，查看 `latest.csv`。
5. 每天 UTC 12:20（北京时间 20:20）也会自动触发一次；GitHub 的计划任务可能延迟。

该工作流将结果保存为 GitHub Actions Artifacts，不会在每次运行时自动提交数据到仓库。要扩大城市范围，请编辑 `data/cities.csv`。

## 项目文件

- `src/forecast.py`：请求气温、以 API 返回的时区判断当地明天、保存 CSV
- `data/cities.csv`：22 个城市经纬度
- `tests/test_forecast.py`：脱离网络的模拟 API 测试
- `.github/workflows/daily-forecast.yml`：每日自动运行和手动触发
- `requirements.txt`：依赖

## 在电脑或 Kaggle 上运行

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.forecast --output output
```

Kaggle 需要开启 Internet 权限。任何请求错误会在 CSV 的 `status` / `error` 列体现，若所有城市都失败，程序会以非零退出码结束。

## 研究限制

- 这是城市代表坐标的 2m 气温预报，不是行政区内部所有测站的最高或最低值。
- 采用每天一次全球 UTC 批处理，尚未严格执行各城市当地 20:00 的统一预报发布时间要求。
- 当前使用 Open-Meteo ECMWF API 提供的日极值，未单独验证接口返回的真实模型起报时间。
- 未接入独立气象站观测，不能声称优于 ECMWF。
- Open-Meteo 免费接口的使用须遵守服务方限制及授权条件。
- 后续可使用 Kaggle CPU 训练 LightGBM 校正器，再比较航空观测特征的独立增益。


## Kaggle 第二阶段：LightGBM 实验性校正

第一阶段 `src/forecast.py` 的每日基线预测仍正常独立运行。第二阶段的研究代码
位于 `research/train_baseline.py`，**不会自动更改生产预报**。

### Kaggle 开始运行（CPU 即可）

1. 前往 [Kaggle Code](https://www.kaggle.com/code)，创建 Notebook，语言 Python，Accelerator 选择 None，开启 Internet。
2. 在 Notebook 单元格按顺序执行：

```python
!git clone https://github.com/Wuzzzzzzz/global-weather-ai.git
%cd global-weather-ai
!pip install -q -r requirements-train.txt
!python -m pytest -q tests/test_research.py
!python -m research.train_baseline --cities 6 --days 180 --test-days 30
```

3. 检查 `research-output/metrics.json` 中 `max` / `min` 对应的 `baseline_mae_c` 与
   `corrected_mae_c`。校正值越低越好；**如果校正没有改进，不要启用校正模型**。
4. 可把 `--cities 6` 改成 `--cities 22`，但注意公共 API 的使用配额。

运行成功时会生成 `training_data.csv`、`temp_max_experimental.joblib`、
`temp_min_experimental.joblib` 和 `metrics.json`。

### 重要的科学限制

- 训练标签目前是 ERA5 **再分析**，并不是独立的城市测站实测；目前无法据此声称真实气温预测精度已提升。
- Previous Runs API 的 `temperature_2m_previous_day1` 是每个有效小时提前24小时的预报，
  再聚合为日极值；并非统一“前一天当地20:00发布”的次日完整预报。
- 后续严格验证需要 Single Runs API 记录固定起报时刻、用 NOAA/ISD 等独立测站观测做标签，
  并与每日基准预测进行相同条件的滚动回测。
- 目前未接入飞机温度观测，不能宣称有航空观测增强效果。
- 数据提供方的免费使用、API 流量和许可限制仍然适用。
