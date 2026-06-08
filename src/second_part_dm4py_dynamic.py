#!/usr/bin/env python3
"""
第二部分：Declare4Py约束挖掘 + 动态关系分析（四层流程）
结构层 -> 逻辑层 -> 统计层 -> 动态层

关键原则：
- 所有概率/相关计算均基于 case-level（二值是否出现）。
- Declare 关系类型由 Declare4Py 给出；统计量只做增强/过滤。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import variance
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import pm4py

REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]


@dataclass
class DynamicConfig:
    window: str = "1D"
    min_window_cases: int = 3

    # 统计层阈值
    positive_delta_threshold: float = 0.1
    min_support_cases: int = 2
    min_support_ratio: float = 0.05
    negative_coexist_threshold: float = 0.1

    # Declare增强（仅非顺序关系启用）
    min_phi_abs: float = 0.05
    min_lift_dev: float = 0.05

    # 动态层阈值
    stable_ratio_threshold: float = 0.7
    prob_range_threshold: float = 0.3
    prob_variance_threshold: float = 0.02

    # Declare4Py
    declare_min_support: float = 0.2
    declare_itemsets_support: float = 0.9


def case_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    return (m > 0).astype(int)


def case_level_pair_metrics(df: pd.DataFrame, a: str, b: str) -> dict:
    m = case_activity_matrix(df)
    if a not in m.columns or b not in m.columns:
        return {
            "p_b_given_a": 0.0,
            "p_b": 0.0,
            "support_count": 0,
            "support_ratio": 0.0,
            "coexist_ratio": 0.0,
            "phi": 0.0,
            "lift": 0.0,
        }

    x, y = m[a], m[b]
    n = len(m)
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())

    a_cases = n11 + n10
    p_b_given_a = (n11 / a_cases) if a_cases > 0 else 0.0
    p_b = (n11 + n01) / max(1, n)
    support_ratio = n11 / max(1, n)
    coexist_ratio = n11 / max(1, n)

    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = ((n11 * n00 - n10 * n01) / den) if den > 0 else 0.0

    px = (n11 + n10) / max(1, n)
    py = (n11 + n01) / max(1, n)
    lift = ((n11 / max(1, n)) / (px * py)) if px > 0 and py > 0 else 0.0

    return {
        "p_b_given_a": round(p_b_given_a, 4),
        "p_b": round(p_b, 4),
        "support_count": int(n11),
        "support_ratio": round(support_ratio, 4),
        "coexist_ratio": round(coexist_ratio, 4),
        "phi": round(phi, 4),
        "lift": round(lift, 4),
    }


def compute_ab_intervals(df: pd.DataFrame, a: str, b: str) -> dict:
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

    if not intervals:
        return {"mean": None, "median": None, "p90": None, "count": 0}

    s = pd.Series(intervals)
    return {
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "p90": round(float(s.quantile(0.9)), 4),
        "count": int(len(intervals)),
    }


class Declare4PyAdapter:
    def __init__(self, min_support: float, itemsets_support: float):
        self.min_support = min_support
        self.itemsets_support = itemsets_support
        self.d4_classes = self._try_import()

    @staticmethod
    def _try_import():
        try:
            from Declare4Py.D4PyEventLog import D4PyEventLog
            from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner

            return {"D4PyEventLog": D4PyEventLog, "DeclareMiner": DeclareMiner}
        except Exception:
            return None

    @property
    def available(self):
        return self.d4_classes is not None

    @staticmethod
    def _parse_serialized(serialized: str) -> Optional[dict]:
        m = re.match(r"^(?P<tpl>[^\[]+)\[(?P<acts>[^\]]*)\]", serialized.strip())
        if not m:
            return None
        tpl = m.group("tpl").strip()
        acts = [a.strip() for a in m.group("acts").split(",") if a.strip()]
        if len(acts) < 2:
            return None  # 过滤 target=None 单事件约束
        return {"constraint_type": tpl, "source": acts[0], "target": acts[1], "serialized": serialized}

    @staticmethod
    def _filter_search_space(constraints: List[dict], search_space: dict) -> List[dict]:
        allowed = {
            (d.get("source", d.get("A")), d.get("target", d.get("B")))
            for d in search_space.get("directed_relations", [])
            if (d.get("source", d.get("A")) is not None) and (d.get("target", d.get("B")) is not None)
        }
        return [c for c in constraints if (c["source"], c["target"]) in allowed]

    @staticmethod
    def _rank_template(t: str) -> int:
        x = t.lower().replace(" ", "")
        if "chain" in x:
            return 3
        if "alternate" in x:
            return 2
        if "response" in x or "precedence" in x or "succession" in x:
            return 1
        return 0

    def _dedup_keep_strongest(self, constraints: List[dict]) -> List[dict]:
        best = {}
        for c in constraints:
            key = (c["source"], c["target"])
            r = self._rank_template(c["constraint_type"])
            if key not in best or r > best[key][0]:
                best[key] = (r, c)
        return [v[1] for v in best.values()]

    def discover_constraints(self, log_df: pd.DataFrame, search_space: dict) -> List[dict]:
        if self.d4_classes is None:
            return []
        try:
            D4PyEventLog = self.d4_classes["D4PyEventLog"]
            DeclareMiner = self.d4_classes["DeclareMiner"]

            log_fmt = pm4py.format_dataframe(
                log_df.copy(),
                case_id="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
            elog = pm4py.convert_to_event_log(log_fmt)
            d4log = D4PyEventLog(case_name="case:concept:name", log=elog)

            miner = DeclareMiner(
                log=d4log,
                consider_vacuity=False,
                min_support=self.min_support,
                itemsets_support=self.itemsets_support,
                max_declare_cardinality=1,
            )
            model = miner.run()
            serialized = getattr(model, "serialized_constraints", []) or []
            parsed = [c for c in (self._parse_serialized(s) for s in serialized) if c is not None]
            filtered = self._filter_search_space(parsed, search_space)
            return self._dedup_keep_strongest(filtered)
        except Exception:
            return []


class DynamicAnalyzer:
    def __init__(self, cfg: DynamicConfig):
        self.cfg = cfg
        self.adapter = Declare4PyAdapter(cfg.declare_min_support, cfg.declare_itemsets_support)

    @staticmethod
    def load_log(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if miss:
            raise ValueError(f"CSV 缺少必要列: {miss}")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
        return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    @staticmethod
    def load_first(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def windows(df: pd.DataFrame, freq: str):
        for bucket, sub in df.groupby(pd.Grouper(key="time:timestamp", freq=freq)):
            if len(sub) > 0:
                yield bucket, sub

    @staticmethod
    def structure_tags(first: dict, source: str, target: str) -> List[str]:
        tags = []
        space = first.get("candidate_constraint_space", {})
        edge = next(
            (
                e
                for e in space.get("directed_relations", [])
                if e.get("source", e.get("A")) == source and e.get("target", e.get("B")) == target
            ),
            None,
        )
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

    @staticmethod
    def _is_sequential(t: str) -> bool:
        x = t.lower().replace(" ", "")
        keys = ["response", "precedence", "succession", "chain", "alternate"]
        return any(k in x for k in keys)

    def _phi_lift_filter_non_seq(self, ctype: str, metrics: dict) -> bool:
        if self._is_sequential(ctype):
            return True
        phi = abs(metrics.get("phi", 0.0))
        lift = metrics.get("lift", 1.0)
        norm_lift = abs(max(lift, 0.0) - 1.0) / (max(lift, 0.0) + 1.0)
        return (phi >= self.cfg.min_phi_abs) or (norm_lift >= self.cfg.min_lift_dev)

    def _dynamic_type(self, temporal_support: float, probs: List[float], global_present: bool) -> str:
        if not probs:
            return "weak_dynamic"
        p_range = max(probs) - min(probs)
        p_var = variance(probs) if len(probs) > 1 else 0.0

        if global_present and temporal_support < 1.0:
            # 全局存在但部分窗口缺失
            if p_range > self.cfg.prob_range_threshold or p_var > self.cfg.prob_variance_threshold:
                return "fluctuating"
            return "disappearing"

        if p_range > self.cfg.prob_range_threshold or p_var > self.cfg.prob_variance_threshold:
            return "fluctuating"

        if temporal_support >= self.cfg.stable_ratio_threshold:
            return "stable"
        return "weak_dynamic"

    def _statistical_branch_relations(self, df: pd.DataFrame, first: dict, window_details: List[dict]) -> List[dict]:
        out = []
        branch_scopes = first.get("candidate_constraint_space", {}).get("branch_scopes", [])
        n_cases = df["case:concept:name"].nunique()

        for bs in branch_scopes:
            split_source = bs.get("split_source")
            targets = bs.get("targets", [])
            for i in range(len(targets)):
                for j in range(i + 1, len(targets)):
                    a, b = targets[i], targets[j]
                    m = case_level_pair_metrics(df, a, b)
                    delta_pos = m["p_b_given_a"] - m["p_b"]

                    rel_type = None
                    if (
                        delta_pos > self.cfg.positive_delta_threshold
                        and m["support_count"] >= self.cfg.min_support_cases
                        and m["support_ratio"] >= self.cfg.min_support_ratio
                    ):
                        rel_type = "positive"
                    elif m["coexist_ratio"] < self.cfg.negative_coexist_threshold:
                        rel_type = "negative"

                    if rel_type is None:
                        continue

                    ctx_probs = []
                    probs = []
                    low_conf_windows = 0
                    for w in window_details:
                        if w["n_cases"] < self.cfg.min_window_cases:
                            low_conf_windows += 1
                            continue
                        pm = case_level_pair_metrics(w["window_df"], a, b)
                        p_ctx = pm["p_b_given_a"]
                        probs.append(p_ctx)
                        ctx_probs.append(
                            {
                                "window_start": w["window_start"],
                                "p_ctx_b_given_a": round(p_ctx, 4),
                                "delta": round(p_ctx - m["p_b_given_a"], 4),
                            }
                        )

                    temporal_support = len(ctx_probs) / max(1, len(window_details))
                    d_type = self._dynamic_type(temporal_support, probs, global_present=True)

                    out.append(
                        {
                            "relation_type": rel_type,
                            "relation_category": "statistical",
                            "source": a,
                            "target": b,
                            "split_source": split_source,
                            "structure_tags": [bs.get("structure_hint", "mixed_branch"), "split_scope"],
                            "p_b_given_a": m["p_b_given_a"],
                            "coexist_ratio": m["coexist_ratio"],
                            "support_count": m["support_count"],
                            "support_ratio": m["support_ratio"],
                            "window_probabilities": ctx_probs,
                            "temporal_support": round(temporal_support, 4),
                            "dynamic_type": d_type,
                            "low_confidence_windows": low_conf_windows,
                        }
                    )
        return out

    def _branch_switching(self, window_details: List[dict], first: dict) -> List[dict]:
        result = []
        for bs in first.get("candidate_constraint_space", {}).get("branch_scopes", []):
            src = bs.get("split_source")
            targets = bs.get("targets", [])
            if not src or len(targets) < 2:
                continue

            winners = []
            for w in window_details:
                if w["n_cases"] < self.cfg.min_window_cases:
                    continue
                scores = []
                for t in targets:
                    p = case_level_pair_metrics(w["window_df"], src, t)["p_b_given_a"]
                    scores.append((t, p))
                scores.sort(key=lambda x: -x[1])
                winners.append(scores[0][0])

            if len(set(winners)) > 1:
                result.append(
                    {
                        "relation_type": "branch_switching",
                        "relation_category": "dynamic",
                        "split_source": src,
                        "targets": targets,
                        "structure_tags": [bs.get("structure_hint", "mixed_branch"), "split_scope"],
                        "dynamic_type": "branch_switching",
                    }
                )
        return result

    def run(self, csv_path: str, first_json: str, out_json: str):
        df = self.load_log(csv_path)
        first = self.load_first(first_json)
        search_space = first.get("candidate_constraint_space", {})

        global_constraints = self.adapter.discover_constraints(df, search_space)

        window_details = []
        window_constraints_map = []
        for bucket, sub in self.windows(df, self.cfg.window):
            cands = self.adapter.discover_constraints(sub, search_space)
            window_details.append(
                {
                    "window_start": bucket.isoformat() if hasattr(bucket, "isoformat") else str(bucket),
                    "window_df": sub,
                    "n_cases": int(sub["case:concept:name"].nunique()),
                    "n_events": int(len(sub)),
                    "constraints": cands,
                }
            )
            window_constraints_map.append(
                {
                    "window_start": window_details[-1]["window_start"],
                    "window_freq": self.cfg.window,
                    "n_cases": window_details[-1]["n_cases"],
                    "n_events": window_details[-1]["n_events"],
                    "constraints": cands,
                }
            )

        total_windows = len(window_details)
        global_keys = {(c["constraint_type"], c["source"], c["target"]) for c in global_constraints}
        window_keys = []
        for w in window_details:
            ks = {(c["constraint_type"], c["source"], c["target"]) for c in w["constraints"]}
            window_keys.append((w["window_start"], w["n_cases"], ks, w["window_df"]))

        declare_records = []
        for c in global_constraints:
            key = (c["constraint_type"], c["source"], c["target"])
            hits = [wk for wk in window_keys if key in wk[2]]
            temporal_support = len(hits) / max(1, total_windows)

            global_m = case_level_pair_metrics(df, c["source"], c["target"])
            if not self._phi_lift_filter_non_seq(c["constraint_type"], global_m):
                continue

            ctx_probs, probs = [], []
            low_conf = 0
            for win_start, n_cases, ks, win_df in window_keys:
                if n_cases < self.cfg.min_window_cases:
                    low_conf += 1
                    continue
                p_ctx = case_level_pair_metrics(win_df, c["source"], c["target"])["p_b_given_a"]
                probs.append(p_ctx)
                ctx_probs.append(
                    {
                        "window_start": win_start,
                        "p_ctx_b_given_a": round(p_ctx, 4),
                        "delta": round(p_ctx - global_m["p_b_given_a"], 4),
                    }
                )

            d_type = self._dynamic_type(temporal_support, probs, global_present=True)
            declare_records.append(
                {
                    "relation_type": c["constraint_type"],
                    "relation_category": "declare",
                    "source": c["source"],
                    "target": c["target"],
                    "structure_tags": self.structure_tags(first, c["source"], c["target"]),
                    "p_b_given_a": global_m["p_b_given_a"],
                    "coexist_ratio": global_m["coexist_ratio"],
                    "window_probabilities": ctx_probs,
                    "temporal_support": round(temporal_support, 4),
                    "dynamic_type": d_type,
                    "a_to_b_interval_seconds": compute_ab_intervals(df, c["source"], c["target"]),
                    "low_confidence_windows": low_conf,
                }
            )

        # emerging: 窗口出现但全局不存在
        emerging = []
        for win_start, n_cases, ks, win_df in window_keys:
            if n_cases < self.cfg.min_window_cases:
                continue
            for key in ks:
                if key not in global_keys:
                    ctype, a, b = key
                    gm = case_level_pair_metrics(df, a, b)
                    emerging.append(
                        {
                            "relation_type": ctype,
                            "relation_category": "dynamic",
                            "source": a,
                            "target": b,
                            "structure_tags": self.structure_tags(first, a, b),
                            "p_b_given_a": gm["p_b_given_a"],
                            "coexist_ratio": gm["coexist_ratio"],
                            "window_probabilities": [{"window_start": win_start, "p_ctx_b_given_a": gm["p_b_given_a"], "delta": 0.0}],
                            "temporal_support": round(1 / max(1, total_windows), 4),
                            "dynamic_type": "emerging",
                        }
                    )

        statistical_records = self._statistical_branch_relations(df, first, window_details)
        switching_records = self._branch_switching(window_details, first)

        structural_loop_records = []
        for loop in first.get("candidates", {}).get("loop_candidates", []):
            if loop.get("type") == "two_activity_loop":
                a, b = loop.get("activities", [None, None])
                if a and b:
                    structural_loop_records.append(
                        {
                            "relation_type": "loop",
                            "relation_category": "structural",
                            "source": a,
                            "target": b,
                            "frequency_ab": loop.get("frequency_ab"),
                            "frequency_ba": loop.get("frequency_ba"),
                            "structure_tags": ["loop_structure"],
                        }
                    )

        all_relations = declare_records + statistical_records + structural_loop_records + switching_records + emerging
        global_existing = [r for r in all_relations if r.get("dynamic_type") == "stable" or r.get("relation_category") == "structural"]
        dynamic_changes = [r for r in all_relations if r.get("dynamic_type") not in (None, "stable")]

        output = {
            "pipeline": [
                "structure_layer",
                "logical_layer_declare4py",
                "statistical_layer",
                "dynamic_layer",
            ],
            "config": {
                "window": self.cfg.window,
                "min_window_cases": self.cfg.min_window_cases,
                "positive_delta_threshold": self.cfg.positive_delta_threshold,
                "min_support_cases": self.cfg.min_support_cases,
                "min_support_ratio": self.cfg.min_support_ratio,
                "negative_coexist_threshold": self.cfg.negative_coexist_threshold,
                "stable_ratio_threshold": self.cfg.stable_ratio_threshold,
                "prob_range_threshold": self.cfg.prob_range_threshold,
                "prob_variance_threshold": self.cfg.prob_variance_threshold,
                "declare4py_available": self.adapter.available,
            },
            "search_space_summary": search_space.get("summary", {}),
            "global_existing_constraints": global_existing,
            "dynamic_changing_constraints": dynamic_changes,
            "window_constraints": [
                {
                    "window_start": w["window_start"],
                    "window_freq": w["window_freq"],
                    "n_cases": w["n_cases"],
                    "n_events": w["n_events"],
                    "low_confidence": w["n_cases"] < self.cfg.min_window_cases,
                    "constraints": [
                        {
                            "relation_type": c["constraint_type"],
                            "relation_category": "declare",
                            "source": c["source"],
                            "target": c["target"],
                        }
                        for c in w["constraints"]
                    ],
                }
                for w in window_constraints_map
            ],
        }

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("=" * 92)
        print("第二部分：四层动态关系分析完成")
        print(f"输入日志: {csv_path}")
        print(f"结构候选输入: {first_json}")
        print(f"输出文件: {out_json}")
        print(f"Declare4Py可用: {self.adapter.available}")
        print(f"全局存在约束数: {len(global_existing)}")
        print(f"动态变化约束数: {len(dynamic_changes)}")
        print("=" * 92)


def parse_args():
    p = argparse.ArgumentParser(description="第二部分：Declare4Py约束 + 动态演化")
    p.add_argument("--csv", default="data/sample_process_log.csv")
    p.add_argument("--first_part_json", default="outputs/structure_candidates.json")
    p.add_argument("--out", default="outputs/dynamic_constraints.json")
    p.add_argument("--window", default="1D")
    p.add_argument("--min_window_cases", type=int, default=3)

    p.add_argument("--positive_delta_threshold", type=float, default=0.1)
    p.add_argument("--min_support_cases", type=int, default=2)
    p.add_argument("--min_support_ratio", type=float, default=0.05)
    p.add_argument("--negative_coexist_threshold", type=float, default=0.1)

    p.add_argument("--min_phi_abs", type=float, default=0.05)
    p.add_argument("--min_lift_dev", type=float, default=0.05)

    p.add_argument("--stable_ratio_threshold", type=float, default=0.7)
    p.add_argument("--prob_range_threshold", type=float, default=0.3)
    p.add_argument("--prob_variance_threshold", type=float, default=0.02)

    p.add_argument("--declare_min_support", type=float, default=0.2)
    p.add_argument("--declare_itemsets_support", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DynamicConfig(
        window=args.window,
        min_window_cases=args.min_window_cases,
        positive_delta_threshold=args.positive_delta_threshold,
        min_support_cases=args.min_support_cases,
        min_support_ratio=args.min_support_ratio,
        negative_coexist_threshold=args.negative_coexist_threshold,
        min_phi_abs=args.min_phi_abs,
        min_lift_dev=args.min_lift_dev,
        stable_ratio_threshold=args.stable_ratio_threshold,
        prob_range_threshold=args.prob_range_threshold,
        prob_variance_threshold=args.prob_variance_threshold,
        declare_min_support=args.declare_min_support,
        declare_itemsets_support=args.declare_itemsets_support,
    )
    DynamicAnalyzer(cfg).run(args.csv, args.first_part_json, args.out)
