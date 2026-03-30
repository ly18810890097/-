# 第一部分实现思路（异常关系约束定位）

## 你这次提出的3个要求怎么落地
1. **活动名去语义化**：示例日志已改为 `t1~t10`。  
2. **加入持续时间属性**：每条事件含 `start_timestamp` 和 `time:timestamp`（完成时间），脚本自动计算 `duration_seconds`。  
3. **输出 Petri 网并做算法对比**：脚本同时跑 **IM / Alpha classic / Alpha+**，导出每个模型的 `pnml`（可选 `png`），并输出 fitness 与结构规模对比。

## 方法总览

### A. 结构候选定位（为第二部分做候选区域）
- PM4Py 发现 DFG + footprints。
- 识别三类候选：
  - split（并行/选择/混合分支）
  - loop（自环与双向环）
  - correlation（活动对显著正/负相关）

### B. 时间要素准备
- 若日志有 `start_timestamp`，脚本计算每事件 `duration_seconds`。
- 输出 `duration_seconds_summary`（mean/median/p90），便于第二部分做动态关系+时间维度扩展。

### C. 经典算法对比
- IM（Inductive Miner）
- Alpha classic
- Alpha+

对每个算法输出：
- Petri 网结构统计（places/transitions/arcs）
- replay fitness（log fitness 与 fitting traces）
- 模型文件路径（pnml，若环境可视化依赖齐全会额外输出 png）

## 数据格式
示例数据位于 `data/sample_process_log.csv`：
- `case:concept:name`
- `concept:name`（`t1~t10`）
- `start_timestamp`
- `time:timestamp`
- `org:resource`

## 运行
```bash
pip install -r requirements.txt
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json \
  --model_dir outputs/petri_nets
```

## 产物
- `outputs/structure_candidates.json`
- `outputs/petri_nets/inductive_miner.pnml`
- `outputs/petri_nets/alpha_classic.pnml`
- `outputs/petri_nets/alpha_plus.pnml`（如果当前 PM4Py 版本支持）
- 对应 PNG（若 graphviz 可用）
