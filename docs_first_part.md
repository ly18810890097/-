# 第一部分实现思路（异常关系约束定位）

## 当前版本关键点（为第二部分 dm4py 做准备）

1. 基于 PM4Py 挖掘 DFG/footprints，并输出 split 与 loop 候选结构。  
2. 生成 `candidate_constraint_space`（第二部分核心输入），包含：
   - `directed_relations`：有向关系 `A→B`、是否双向、是否来自 split/loop；
   - `branch_scopes`：局部分支 `A→{B,C,...}` 及结构提示（xor/and/mixed）；
   - `loop_nodes`：循环相关节点集合。  
3. 保留多算法 Petri 网对比（IM / Alpha classic / Alpha+ / Heuristics*）。

## 运行示例
```bash
pip install -r requirements.txt
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json \
  --model_dir outputs/petri_nets \
  --edge_sensitivity 0.7 \
  --loop_sensitivity 0.8
```

## 产物
- `outputs/structure_candidates.json`
- `outputs/petri_nets/*.pnml`
- `outputs/petri_nets/*.png`（若 graphviz 可用）
