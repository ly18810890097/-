#!/usr/bin/env python3
"""
第二部分：dm4py 约束挖掘 + 时间稳定性增强。

流程：
结构候选生成(第一部分) -> dm4py约束挖掘 -> 时间维度增强 -> 输出动态约束

注意：
- 关系类型由 dm4py 决定（response/precedence/co-existence...）。
- phi/lift 仅用于过滤与排序，不用于定义关系类型。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]


@dataclass
class DynamicConfig:
    window: str = "7D"
    min_phi_abs: float = 0.05
    min_lift_dev: float = 0.05
    stable_ratio_threshold: float = 0.7


def _contingency(x: pd.Series, y: pd.Series) -> Tuple[int, int, int, int]:
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    return n11, n10, n01, n00


def phi_binary(x: pd.Series, y: pd.Series) -> float:
    n11, n10, n01, n00 = _contingency(x, y)
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / den


def lift_binary(x: pd.Series, y: pd.Series) -> float:
    n11, n10, n01, n00 = _contingency(x, y)
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 1.0
    p_x = (n11 + n10) / n
    p_y = (n11 + n01) / n
    if p_x == 0 or p_y == 0:
        return 0.0
    return (n11 / n) / (p_x * p_y)


def norm_lift_dev(lift: float) -> float:
    return abs(max(0.0, lift) - 1.0) / (max(0.0, lift) + 1.0)


def case_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    return (m > 0).astype(int)


def compute_ab_intervals(df: pd.DataFrame, a: str, b: str) -> List[float]:
    """A->B 时间间隔（秒），按 case 内顺序匹配“每个A后的最近B”。"""
    intervals = []
    for _, g in df.groupby("case:concept:name"):
        g = g.sort_values("time:timestamp")
        a_times = g[g["concept:name"] == a]["time:timestamp"].tolist()
        b_times = g[g["concept:name"] == b]["time:timestamp"].tolist()
        if not a_times or not b_times:
            continue
        j = 0
        for ta in a_times:
            while j < len(b_times) and b_times[j] <= ta:
                j += 1
            if j < len(b_times):
                intervals.append((b_times[j] - ta).total_seconds())
    return intervals


class DM4PyConstraintAdapter:
    """dm4py 适配器：把第一部分候选约束空间作为搜索范围输入。"""

    def __init__(self):
        self.dm4py = self._try_import()

    @staticmethod
    def _try_import():
        try:
            return importlib.import_module("dm4py")
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self.dm4py is not None

    def discover_constraints(self, log_df: pd.DataFrame, search_space: dict) -> List[dict]:
        """
        关系类型必须来自 dm4py。
        这里仅做 API 适配，不重写约束挖掘算法。
        """
        if self.dm4py is None:
            return []

        candidate_fns = [
            "discover_constraints",
            "mine_constraints",
            "discover_declare_constraints",
        ]
        for fn_name in candidate_fns:
            fn = getattr(self.dm4py, fn_name, None)
            if callable(fn):
                try:
                    out = fn(log_df, search_space=search_space)  # type: ignore[misc]
                    if isinstance(out, list):
                        return out
                except Exception:
                    continue
        return []


class DynamicConstraintMiner:
    def __init__(self, cfg: DynamicConfig):
        self.cfg = cfg
        self.adapter = DM4PyConstraintAdapter()

    @staticmethod
    def load_log(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if miss:
            raise ValueError(f"CSV 缺少必要列: {miss}")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    @staticmethod
    def load_first_output(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def window_iter(df: pd.DataFrame, freq: str):
        for bucket, sub in df.groupby(pd.Grouper(key="time:timestamp", freq=freq)):
            if len(sub) == 0:
                continue
            yield bucket, sub

    @staticmethod
    def normalize_constraint(c: dict) -> Optional[dict]:
        """标准化 dm4py 返回格式（仅解析，不定义类型）。"""
        ctype = c.get("constraint_type") or c.get("type")
        a = c.get("source") or c.get("A") or c.get("antecedent")
        b = c.get("target") or c.get("B") or c.get("consequent")
        if not ctype or not a or not b:
            return None
        return {
            "constraint_type": str(ctype),
            "source": str(a),
            "target": str(b),
            "raw": c,
        }

    @staticmethod
    def structure_tags(first: dict, source: str, target: str) -> List[str]:
        tags = []
        space = first.get("candidate_constraint_space", {})

        directed = space.get("directed_relations", [])
        edge = next((e for e in directed if e.get("source") == source and e.get("target") == target), None)
        if edge:
            if edge.get("from_loop"):
                tags.append("loop_structure")
            if edge.get("from_split"):
                tags.append("split_structure")
            if edge.get("bidirectional"):
                tags.append("bidirectional_edge")

        for b in space.get("branch_scopes", []):
            if b.get("split_source") == source and target in b.get("targets", []):
                hint = b.get("structure_hint")
                if hint == "xor_branch":
                    tags.append("xor_candidate")
                elif hint == "and_branch":
                    tags.append("and_candidate")
                elif hint:
                    tags.append("mixed_branch")
                break

        return tags

    def confidence_metrics(self, df: pd.DataFrame, a: str, b: str) -> dict:
        m = case_activity_matrix(df)
        if a not in m.columns or b not in m.columns:
            return {"phi": 0.0, "lift": 0.0, "norm_lift_dev": 0.0}
        x, y = m[a], m[b]
        phi = phi_binary(x, y)
        lift = lift_binary(x, y)
        return {
            "phi": round(phi, 4),
            "lift": round(lift, 4),
            "norm_lift_dev": round(norm_lift_dev(lift), 4),
        }

    def pass_confidence_filter(self, conf: dict) -> bool:
        return abs(conf["phi"]) >= self.cfg.min_phi_abs or conf["norm_lift_dev"] >= self.cfg.min_lift_dev

    def analyze_temporal_support(
        self,
        df: pd.DataFrame,
        search_space: dict,
        first_result: dict,
        global_constraints: List[dict],
    ):
        normalized_global = []
        for c in global_constraints:
            nc = self.normalize_constraint(c)
            if nc is not None:
                normalized_global.append(nc)

        windows = list(self.window_iter(df, self.cfg.window))
        total_windows = len(windows)

        win_constraints = []
        for bucket, sub in windows:
            cands = self.adapter.discover_constraints(sub, search_space)
            present = set()
            norm = []
            for c in cands:
                nc = self.normalize_constraint(c)
                if nc is None:
                    continue
                key = (nc["constraint_type"], nc["source"], nc["target"])
                present.add(key)
                norm.append(nc)
            win_constraints.append(
                {
                    "window_start": bucket.isoformat() if hasattr(bucket, "isoformat") else str(bucket),
                    "window_freq": self.cfg.window,
                    "n_events": int(len(sub)),
                    "n_cases": int(sub["case:concept:name"].nunique()),
                    "constraints": norm,
                    "present_keys": present,
                    "window_df": sub,
                }
            )

        records = []
        for gc in normalized_global:
            key = (gc["constraint_type"], gc["source"], gc["target"])
            hit_windows = [w for w in win_constraints if key in w["present_keys"]]
            temporal_support = (len(hit_windows) / total_windows) if total_windows > 0 else 0.0

            conf = self.confidence_metrics(df, gc["source"], gc["target"])
            if not self.pass_confidence_filter(conf):
                continue

            intervals = compute_ab_intervals(df, gc["source"], gc["target"])
            if intervals:
                s = pd.Series(intervals)
                interval_stats = {
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                    "p90": round(float(s.quantile(0.9)), 4),
                    "count": int(len(intervals)),
                }
            else:
                interval_stats = {"mean": None, "median": None, "p90": None, "count": 0}

            records.append(
                {
                    "constraint_type": gc["constraint_type"],
                    "source": gc["source"],
                    "target": gc["target"],
                    "structure_tags": self.structure_tags(first_result, gc["source"], gc["target"]),
                    "confidence": conf,
                    "temporal_support": round(temporal_support, 4),
                    "active_windows": len(hit_windows),
                    "total_windows": total_windows,
                    "a_to_b_interval_seconds": interval_stats,
                }
            )

        return records, win_constraints

    def run(self, csv_path: str, first_json: str, out_json: str):
        df = self.load_log(csv_path)
        first_result = self.load_first_output(first_json)

        search_space = first_result.get("candidate_constraint_space", {})
        global_constraints = self.adapter.discover_constraints(df, search_space)

        records, window_details = self.analyze_temporal_support(
            df=df,
            search_space=search_space,
            first_result=first_result,
            global_constraints=global_constraints,
        )

        stable = [r for r in records if r["temporal_support"] >= self.cfg.stable_ratio_threshold]
        dynamic = [r for r in records if r["temporal_support"] < self.cfg.stable_ratio_threshold]

        # 窗口明细精简输出，移除中间字段
        simple_windows = []
        for w in window_details:
            simple_windows.append(
                {
                    "window_start": w["window_start"],
                    "window_freq": w["window_freq"],
                    "n_events": w["n_events"],
                    "n_cases": w["n_cases"],
                    "constraints": [
                        {
                            "constraint_type": c["constraint_type"],
                            "source": c["source"],
                            "target": c["target"],
                        }
                        for c in w["constraints"]
                    ],
                }
            )

        output = {
            "pipeline": [
                "structure_candidate_generation",
                "dm4py_constraint_mining",
                "time_dimension_enhancement",
                "dynamic_constraint_output",
            ],
            "input": {
                "csv": csv_path,
                "first_part_json": first_json,
            },
            "config": {
                "window": self.cfg.window,
                "min_phi_abs": self.cfg.min_phi_abs,
                "min_lift_dev": self.cfg.min_lift_dev,
                "stable_ratio_threshold": self.cfg.stable_ratio_threshold,
                "dm4py_available": self.adapter.available,
            },
            "search_space_summary": search_space.get("summary", {}),
            "global_constraints_raw_count": len(global_constraints),
            "constraints_after_confidence_filter": len(records),
            "global_existing_constraints": stable,
            "dynamic_changing_constraints": dynamic,
            "window_constraints": simple_windows,
        }

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("=" * 92)
        print("第二部分：dm4py约束挖掘 + 时间稳定性分析")
        print(f"输入日志: {csv_path}")
        print(f"结构候选输入: {first_json}")
        print(f"输出文件: {out_json}")
        print(f"dm4py可用: {self.adapter.available}")
        print(f"全局存在约束数: {len(stable)}")
        print(f"动态变化约束数: {len(dynamic)}")
        print("=" * 92)


def parse_args():
    p = argparse.ArgumentParser(description="第二部分：dm4py约束 + 时间稳定性")
    p.add_argument("--csv", default="data/sample_process_log.csv")
    p.add_argument("--first_part_json", default="outputs/structure_candidates.json")
    p.add_argument("--out", default="outputs/dynamic_constraints.json")
    p.add_argument("--window", default="7D")
    p.add_argument("--min_phi_abs", type=float, default=0.05)
    p.add_argument("--min_lift_dev", type=float, default=0.05)
    p.add_argument("--stable_ratio_threshold", type=float, default=0.7)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DynamicConfig(
        window=args.window,
        min_phi_abs=args.min_phi_abs,
        min_lift_dev=args.min_lift_dev,
        stable_ratio_threshold=args.stable_ratio_threshold,
    )
    DynamicConstraintMiner(cfg).run(args.csv, args.first_part_json, args.out)
