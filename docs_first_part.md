# 第一部分实现思路（异常关系约束定位）

## 本次关键修复与升级

1. **修复 lift=0 问题（最重要）**
- 当 `joint_cases(n11)=0` 时，不再参与新强度计算（直接过滤），避免 `log(lift)` 类爆炸。
- baseline 旧分数仍保留用于对比。

2. **统一 strength 量纲**
- 不再使用无界 `log(lift)`。
- 改为 `normalized_lift_deviation = |lift-1|/(lift+1)`（范围 `[0,1)`）。
- 综合强度：`strength = 0.6*|phi| + 0.4*normalized_lift_deviation`。

3. **split 判定改为加权**
- 用“负相关强度总和 vs 正相关强度总和”判断 XOR/AND 倾向。
- 不再仅靠 pair 数量计数。

4. **valid_pair_count 增强**
- 新增 `valid_pair_ratio = valid_pair_count / total_pairs`。
- 同时输出 `max_strength`，提升不同分支间可比性。

5. **sensitivity 分离**
- `--edge_sensitivity`：控制 split 候选边筛选。
- `--loop_sensitivity`：控制 loop 候选边筛选。

6. **correlation 增加结构约束**
- 新增 `--corr_scope`：
  - `dfg_local`（默认）：仅在 DFG 邻域对内做相关分析；
  - `global`：全局活动对分析。

7. **baseline 对比**
- 保留 `legacy_score` 粗筛（`|score|>=0.2`）统计。
- 输出 baseline 与新方法候选数量对比，便于评估噪声与解释性改进。

## 模型对比算法
- IM（Inductive Miner）
- Alpha classic
- Alpha+
- Heuristics Miner（若当前 PM4Py 版本支持）

## 运行示例
```bash
pip install -r requirements.txt
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json \
  --model_dir outputs/petri_nets \
  --edge_sensitivity 0.7 \
  --loop_sensitivity 0.8 \
  --corr_scope dfg_local \
  --min_joint_cases 2 \
  --min_strength 0.2
```

## 产物
- `outputs/structure_candidates.json`
- `outputs/petri_nets/*.pnml`
- `outputs/petri_nets/*.png`（若 graphviz 可用）
