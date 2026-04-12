#!/usr/bin/env python3
"""
第二部分：在第一部分输出基础上进行“动态关系挖掘（含时间维度）”。

设计目标：
1) 复用第一部分输出（structure_candidates.json）定位到的候选结构；
2) 采用“dm4py 基础能力 + 小幅增强”的方式，而不是重写全流程；
3) 输出“关系在什么时间窗成立”的动态结果，以及对应持续时间统计。

说明：
- 若环境已安装 dm4py，则优先调用 dm4py 基础关系挖掘接口；
- 若 dm4py 不可用，则退化到轻量 fallback 计算（便于脚本可运行与调试）。
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
    min_joint_cases: int = 2
    min_strength: float = 0.2
    max_choice_coexist_ratio: float = 0.2


def _contingency(x: pd.Series, y: pd.Series) -> Tuple[int, int, int, int]:
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    return n11, n10, n01, n00


def phi_binary(x: pd.Series, y: pd.Series) -> float:
    n11, n10, n01, n00 = _contingency(x, y)
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denominator == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denominator


def lift_binary(x: pd.Series, y: pd.Series) -> float:
    n11, n10, n01, n00 = _contingency(x, y)
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 1.0
    p_x = (n11 + n10) / n
    p_y = (n11 + n01) / n
    p_xy = n11 / n
    if p_x == 0 or p_y == 0:
        return 0.0
    return p_xy / (p_x * p_y)


def normalized_lift_deviation(lift: float) -> float:
    lift = max(lift, 0.0)
    return abs(lift - 1.0) / (lift + 1.0)


def relation_strength(phi: float, lift: float) -> float:
    return 0.6 * abs(phi) + 0.4 * normalized_lift_deviation(lift)


class DM4PyDynamicMiner:
    """
    “dm4py 基础 + 小幅增强”实现：
    - base: 窗口内关系挖掘（优先 dm4py）
    - add-on: 候选位置过滤 + 时间窗稳定性输出 + 持续时间统计
    """

    def __init__(self, config: DynamicConfig):
        self.config = config
        self.dm4py_module = self._try_import_dm4py()

    @staticmethod
    def _try_import_dm4py():
        try:
            return importlib.import_module("dm4py")
        except Exception:
            return None

    @staticmethod
    def load_log(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要列: {missing}")

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        if "start_timestamp" in df.columns:
            df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True, errors="coerce")
            df["duration_seconds"] = (
                df["time:timestamp"] - df["start_timestamp"]
            ).dt.total_seconds().clip(lower=0)
        else:
            df["duration_seconds"] = pd.NA

        return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    @staticmethod
    def load_first_part_output(json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_candidate_pairs(first_part_result: dict) -> List[Tuple[str, str]]:
        pairs = set()

        # 来自 split 的 pairwise_stats
        for split in first_part_result.get("candidates", {}).get("split_candidates", []):
            for p in split.get("pairwise_stats", []):
                pair = p.get("pair", [])
                if len(pair) == 2:
                    pairs.add(tuple(sorted(pair)))

        # 来自 correlation 候选
        for rel in first_part_result.get("candidates", {}).get("correlation_candidates", []):
            pair = rel.get("activities", [])
            if len(pair) == 2:
                pairs.add(tuple(sorted(pair)))

        return sorted(pairs)

    @staticmethod
    def _window_iterator(df: pd.DataFrame, window: str):
        # 以事件完成时间分窗
        for bucket, sub in df.groupby(pd.Grouper(key="time:timestamp", freq=window)):
            if len(sub) == 0:
                continue
            yield bucket, sub

    @staticmethod
    def _case_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
        m = df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
        return (m > 0).astype(int)

    @staticmethod
    def _direct_follow_counts(df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for _, g in df.groupby("case:concept:name"):
            acts = g.sort_values("time:timestamp")["concept:name"].tolist()
            for i in range(len(acts) - 1):
                edge = (acts[i], acts[i + 1])
                counts[edge] = counts.get(edge, 0) + 1
        return counts

    def _mine_pair_by_dm4py_base(self, window_df: pd.DataFrame, pair: Tuple[str, str]) -> Optional[dict]:
        """
        这里体现“在 dm4py 基础上修改”：
        - 如果环境内 dm4py 提供关系接口，优先调用其基础结果；
        - 再叠加本研究的时间窗过滤与阈值逻辑（非重写）。
        """
        if self.dm4py_module is None:
            return None

        # 尽量宽松适配不同 dm4py API（避免固定某一版本函数名）
        candidate_fn_names = [
            "discover_dynamic_relations",
            "discover_relations",
            "mine_relations",
        ]
        for fn_name in candidate_fn_names:
            fn = getattr(self.dm4py_module, fn_name, None)
            if callable(fn):
                try:
                    return fn(window_df, pair)  # type: ignore[misc]
                except Exception:
                    continue
        return None

    def _mine_pair_fallback(self, window_df: pd.DataFrame, pair: Tuple[str, str]) -> dict:
        a, b = pair
        m = self._case_activity_matrix(window_df)
        if a not in m.columns or b not in m.columns:
            return {
                "activities": [a, b],
                "joint_cases": 0,
                "phi": 0.0,
                "lift": 0.0,
                "strength": 0.0,
                "coexist_ratio": 0.0,
            }

        x, y = m[a], m[b]
        n11, n10, n01, n00 = _contingency(x, y)
        phi = phi_binary(x, y) if n11 > 0 else 0.0
        lift = lift_binary(x, y) if n11 > 0 else 0.0
        strength = relation_strength(phi, lift) if n11 > 0 else 0.0

        total_cases = max(1, n11 + n10 + n01 + n00)
        coexist_ratio = n11 / total_cases

        return {
            "activities": [a, b],
            "joint_cases": n11,
            "phi": round(phi, 4),
            "lift": round(lift, 4),
            "strength": round(strength, 4),
            "coexist_ratio": round(coexist_ratio, 4),
        }

    def _classify_relation(self, rel: dict, dfg_counts: Dict[Tuple[str, str], int]) -> dict:
        a, b = rel["activities"]
        n11 = rel.get("joint_cases", 0)
        phi = rel.get("phi", 0.0) or 0.0
        strength = rel.get("strength", 0.0) or 0.0
        coexist_ratio = rel.get("coexist_ratio", 0.0) or 0.0

        ab = dfg_counts.get((a, b), 0)
        ba = dfg_counts.get((b, a), 0)

        tags = []
        if ab > 0 and ba > 0:
            tags.append("loop_possible")
        if coexist_ratio <= self.config.max_choice_coexist_ratio and phi < 0:
            tags.append("choice_possible")
        if phi > 0:
            tags.append("positive_correlation")
        elif phi < 0:
            tags.append("negative_correlation")

        passed = n11 >= self.config.min_joint_cases and strength >= self.config.min_strength

        return {
            **rel,
            "dfg_ab": int(ab),
            "dfg_ba": int(ba),
            "relation_tags": tags,
            "passed_threshold": bool(passed),
        }

    @staticmethod
    def _duration_summary_for_pair(window_df: pd.DataFrame, pair: Tuple[str, str]) -> dict:
        a, b = pair
        sub = window_df[window_df["concept:name"].isin([a, b])]
        if len(sub) == 0 or "duration_seconds" not in sub.columns or sub["duration_seconds"].isna().all():
            return {"mean": None, "median": None, "p90": None}
        return {
            "mean": round(float(sub["duration_seconds"].mean()), 4),
            "median": round(float(sub["duration_seconds"].median()), 4),
            "p90": round(float(sub["duration_seconds"].quantile(0.9)), 4),
        }

    def run(self, csv_path: str, first_part_json: str, out_json: str):
        df = self.load_log(csv_path)
        first = self.load_first_part_output(first_part_json)
        pairs = self._extract_candidate_pairs(first)

        all_windows = []
        for bucket, sub in self._window_iterator(df, self.config.window):
            dfg_counts = self._direct_follow_counts(sub)
            window_result = {
                "window_start": bucket.isoformat() if hasattr(bucket, "isoformat") else str(bucket),
                "window_freq": self.config.window,
                "n_events": int(len(sub)),
                "n_cases": int(sub["case:concept:name"].nunique()),
                "relations": [],
            }

            for pair in pairs:
                dm4py_base = self._mine_pair_by_dm4py_base(sub, pair)
                rel = dm4py_base if isinstance(dm4py_base, dict) else self._mine_pair_fallback(sub, pair)

                rel = self._classify_relation(rel, dfg_counts)
                rel["duration_seconds_summary"] = self._duration_summary_for_pair(sub, pair)

                # 仅保留当前窗口有效关系
                if rel.get("passed_threshold", False):
                    window_result["relations"].append(rel)

            all_windows.append(window_result)

        output = {
            "input": {
                "csv": csv_path,
                "first_part_json": first_part_json,
                "window": self.config.window,
            },
            "config": {
                "min_joint_cases": self.config.min_joint_cases,
                "min_strength": self.config.min_strength,
                "max_choice_coexist_ratio": self.config.max_choice_coexist_ratio,
                "dm4py_available": self.dm4py_module is not None,
            },
            "candidate_pair_count": len(pairs),
            "windows": all_windows,
        }

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("=" * 88)
        print("第二部分：动态关系挖掘（dm4py-base + 时间窗增强）")
        print(f"输入日志: {csv_path}")
        print(f"第一部分结果: {first_part_json}")
        print(f"输出文件: {out_json}")
        print(f"窗口频率: {self.config.window}")
        print(f"候选活动对数量: {len(pairs)}")
        print(f"dm4py 可用: {self.dm4py_module is not None}")
        print("=" * 88)


def parse_args():
    parser = argparse.ArgumentParser(description="第二部分：动态关系挖掘（基于第一部分输出）")
    parser.add_argument("--csv", default="data/sample_process_log.csv", help="输入事件日志 CSV")
    parser.add_argument(
        "--first_part_json",
        default="outputs/structure_candidates.json",
        help="第一部分输出 JSON",
    )
    parser.add_argument(
        "--out",
        default="outputs/dynamic_relations_with_time.json",
        help="第二部分输出 JSON",
    )
    parser.add_argument("--window", default="7D", help="时间窗频率（如 1D/7D/1M）")
    parser.add_argument("--min_joint_cases", type=int, default=2, help="最小联合出现案例数")
    parser.add_argument("--min_strength", type=float, default=0.2, help="最小关系强度")
    parser.add_argument(
        "--max_choice_coexist_ratio",
        type=float,
        default=0.2,
        help="判定 choice_possible 的最大共现比例阈值",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = DynamicConfig(
        window=args.window,
        min_joint_cases=args.min_joint_cases,
        min_strength=args.min_strength,
        max_choice_coexist_ratio=args.max_choice_coexist_ratio,
    )
    miner = DM4PyDynamicMiner(config)
    miner.run(args.csv, args.first_part_json, args.out)
