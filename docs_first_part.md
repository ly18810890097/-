# 第一部分实现思路（异常关系约束定位）

## 本次算法升级（针对你的3点问题）

1. **score 是否合理？是否引入 phi / lift？**  
   已升级为 **phi + lift + support** 综合度量，不再只依赖单一分数：
   - `phi`：描述正/负相关方向与强度；
   - `lift`：描述活动共现相对于独立假设的偏离；
   - `support(joint_cases)`：避免低样本偶然相关。

   同时定义综合强度：
   `strength = 0.6 * |phi| + 0.4 * |log2(lift)|`

2. **是否加入敏感度阈值 + 数量门槛？**  
   已加入：
   - `--sensitivity`：基于 DFG 边频分布的动态阈值；
   - `--min_joint_cases`：联合出现最小案例数；
   - `--min_strength`：综合强度下限。  
   低频、低支持、低强度候选将被自动过滤，减少无效研究位置。

3. **除 IM/Alpha 外是否加入其他算法对比？**  
   已加入 **Heuristics Miner**（若当前 PM4Py 版本支持），用于与 IM / Alpha classic / Alpha+ 对比。

## 方法总览

### A. 结构候选定位
- PM4Py 发现 DFG + footprints。
- 三类候选：
  - split（并行/选择/混合分支）
  - loop（自环与双向环）
  - correlation（活动对显著正/负相关）

### B. 时间要素准备
- 事件含 `start_timestamp` + `time:timestamp`。
- 脚本计算 `duration_seconds`，并输出 `mean/median/p90`。

### C. 结构模型对比
- IM（Inductive Miner）
- Alpha classic
- Alpha+
- Heuristics Miner（可选，版本支持时启用）

对每个模型输出：
- Petri 网结构统计（places/transitions/arcs）
- replay fitness（log fitness 与 fitting traces）
- `pnml` 路径（以及可选 `png`）

## 运行
```bash
pip install -r requirements.txt
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json \
  --model_dir outputs/petri_nets \
  --sensitivity 0.7 \
  --min_joint_cases 2 \
  --min_strength 0.2
```

## 输出
- `outputs/structure_candidates.json`
- `outputs/petri_nets/*.pnml`
- `outputs/petri_nets/*.png`（若 graphviz 可用）
