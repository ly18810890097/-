# 第一部分输出规范（trace 统计版）

`src/first_part_pm4py_locator.py` 已按统一规范输出：
- traces 级统计（case-level）
- 互斥关系判定顺序：loop → parallel → choice → sequence
- blocks、splits、loops
- candidate_constraint_space（供第二部分 Declare 搜索约束）

## 运行
```bash
python src/first_part_pm4py_locator.py \
  --csv data/sample_process_log.csv \
  --out outputs/structure_candidates.json
```

## 第二部分/统一流程
- 第二部分：`src/second_part_dm4py_dynamic.py`
- 统一四层脚本：`src/pipeline_structured_relations.py`
