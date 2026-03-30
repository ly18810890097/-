# 第一部分实现思路（异常关系约束定位）

## 目标
基于 PM4Py 先做**结构级定位**：从事件日志中找到最可能存在复杂约束的位置，为第二部分动态关系挖掘提供候选区域。

## 方法分三层
1. **流程骨架发现**
   - 用 PM4Py 发现 DFG（直接跟随图）和 Footprints。
   - DFG 给出活动连接强度，Footprints 给出顺序/并行线索。

2. **复杂结构候选定位**
   - **分支候选（split）**：某活动有多个后继时，标记为候选。
   - 通过案例级共现行为估计偏向：
     - 倾向同现：可能并行/混合。
     - 倾向互斥：可能选择（XOR）/混合。

3. **异常关系优先级排序**
   - **循环候选**：自环、双向环（A→B 与 B→A）。
   - **相关候选**：用活动出现的二值向量计算 phi 相关系数，定位显著正/负相关活动对。

## 输出
脚本会输出 `outputs/structure_candidates.json`，包括：
- 数据集统计信息
- 高频 DFG 边
- Footprints 摘要
- 三类候选：split / loop / correlation

## 运行
```bash
pip install -r requirements.txt
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json
```

