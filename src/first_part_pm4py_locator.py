#!/usr/bin/env python3
"""
第一部分：基于 PM4Py 定位复杂关系约束候选位置，并输出多算法 Petri 网。

本版本重点：
- 修复 lift=0 导致 strength 异常放大问题。
- strength 量纲统一（phi + 归一化 lift 偏离）。
- split 类型判定改为“强度加权”。
- 增加 valid_pair_ratio / max_strength。
- 分离 edge_sensitivity 与 loop_sensitivity。
- correlation 增加结构约束（DFG 邻域）。
- 提供 baseline 对比（旧 score vs 新指标）。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.format_dataframe import format_dataframe
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

try:
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
except Exception:
    heuristics_miner = None

REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]


def load_event_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df = format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    if "start_timestamp" in df.columns:
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True, errors="coerce")
        df["duration_seconds"] = (
            df["time:timestamp"] - df["start_timestamp"]
        ).dt.total_seconds().clip(lower=0)
    else:
        df["duration_seconds"] = pd.NA

    return df


def discover_dfg_and_footprints(df: pd.DataFrame):
    dfg = dfg_discovery.apply(df)
    footprints = footprints_discovery.apply(df)
    return dfg, footprints


def get_case_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    case_act = (
        df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    )
    return (case_act > 0).astype(int)


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
    """标准 lift。n11=0 时返回 0（后续由上层过滤）。"""
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
    """把 lift 偏离归一到 [0,1)：|lift-1|/(lift+1)。"""
    lift = max(lift, 0.0)
    return abs(lift - 1.0) / (lift + 1.0)


def relation_strength(phi: float, lift: float, w_phi: float = 0.6, w_lift: float = 0.4) -> float:
    phi_part = abs(phi)  # [0,1]
    lift_part = normalized_lift_deviation(lift)  # [0,1)
    return w_phi * phi_part + w_lift * lift_part


def legacy_score(n11: int, n10: int, n01: int) -> float:
    """旧版评分：用于 baseline 对比。"""
    only_one = n10 + n01
    return (n11 - only_one) / max(1, n11 + only_one)


def dynamic_threshold_from_freqs(freqs: List[int], sensitivity: float) -> int:
    if not freqs:
        return 1
    sensitivity = min(max(sensitivity, 0.0), 1.0)
    # sensitivity 越高，阈值越高（更保守）
    sorted_freqs = sorted(int(v) for v in freqs)
    idx = int(round((1.0 - sensitivity) * (len(sorted_freqs) - 1)))
    return max(1, sorted_freqs[idx])


def edge_sensitivity_threshold(dfg: Dict[Tuple[str, str], int], sensitivity: float) -> int:
    return dynamic_threshold_from_freqs(list(dfg.values()), sensitivity)


def build_dfg_local_scope(dfg: Dict[Tuple[str, str], int]) -> Set[Tuple[str, str]]:
    """构建结构约束候选对：仅保留 DFG 邻域内活动对（无向1跳）。"""
    undirected = defaultdict(set)
    for (a, b), _ in dfg.items():
        undirected[a].add(b)
        undirected[b].add(a)

    valid_pairs = set()
    nodes = list(undirected.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            # 直接相邻 or 共享邻居，限制相关分析范围
            if b in undirected[a] or len(undirected[a].intersection(undirected[b])) > 0:
                valid_pairs.add(tuple(sorted((a, b))))
    return valid_pairs


def extract_split_candidates(
    dfg: Dict[Tuple[str, str], int],
    case_activity_matrix: pd.DataFrame,
    edge_sensitivity: float = 0.7,
    min_out_degree: int = 2,
    min_joint_cases: int = 2,
    min_strength: float = 0.2,
) -> List[dict]:
    dyn_edge_thr = edge_sensitivity_threshold(dfg, edge_sensitivity)

    outgoing = defaultdict(list)
    for (a, b), freq in dfg.items():
        if int(freq) >= dyn_edge_thr:
            outgoing[a].append((b, int(freq)))

    candidates = []
    for src, targets in outgoing.items():
        if len(targets) < min_out_degree:
            continue

        tgt_names = [t for t, _ in targets]
        pair_stats = []
        valid_pair_count = 0
        neg_weight_sum = 0.0
        pos_weight_sum = 0.0
        max_strength = 0.0
        baseline_scores = []

        for i in range(len(tgt_names)):
            for j in range(i + 1, len(tgt_names)):
                t1, t2 = tgt_names[i], tgt_names[j]
                x, y = case_activity_matrix[t1], case_activity_matrix[t2]
                n11, n10, n01, n00 = _contingency(x, y)

                # 修复 lift=0 问题：n11=0 时仅保留 baseline，不进入新强度计算
                base = legacy_score(n11, n10, n01)
                baseline_scores.append(base)
                if n11 == 0:
                    pair_stats.append(
                        {
                            "pair": [t1, t2],
                            "joint_cases": n11,
                            "only_left_cases": n10,
                            "only_right_cases": n01,
                            "none_cases": n00,
                            "phi": None,
                            "lift": 0.0,
                            "strength": 0.0,
                            "legacy_score": round(base, 4),
                            "filtered_reason": "joint_cases=0",
                        }
                    )
                    continue

                phi = phi_binary(x, y)
                lift = lift_binary(x, y)
                strength = relation_strength(phi, lift)
                max_strength = max(max_strength, strength)

                if phi < 0:
                    neg_weight_sum += strength
                else:
                    pos_weight_sum += strength

                stat = {
                    "pair": [t1, t2],
                    "joint_cases": n11,
                    "only_left_cases": n10,
                    "only_right_cases": n01,
                    "none_cases": n00,
                    "phi": round(phi, 4),
                    "lift": round(lift, 4),
                    "strength": round(strength, 4),
                    "legacy_score": round(base, 4),
                }
                pair_stats.append(stat)

                if n11 >= min_joint_cases and strength >= min_strength:
                    valid_pair_count += 1

        total_pairs = len(pair_stats)
        if total_pairs == 0:
            continue

        valid_pair_ratio = valid_pair_count / total_pairs
        avg_strength = sum(p["strength"] for p in pair_stats) / total_pairs
        avg_abs_phi = (
            sum(abs(p["phi"]) for p in pair_stats if p["phi"] is not None)
            / max(1, sum(1 for p in pair_stats if p["phi"] is not None))
        )
        avg_legacy = sum(baseline_scores) / max(1, len(baseline_scores))

        if valid_pair_count == 0:
            continue

        est_type = "potential_XOR_or_mixed" if neg_weight_sum > pos_weight_sum else "potential_AND_or_mixed"

        candidates.append(
            {
                "source": src,
                "outgoing": [{"target": t, "freq": f} for t, f in sorted(targets, key=lambda z: -z[1])],
                "avg_strength": round(avg_strength, 4),
                "avg_abs_phi": round(avg_abs_phi, 4),
                "max_strength": round(max_strength, 4),
                "valid_pair_count": valid_pair_count,
                "valid_pair_ratio": round(valid_pair_ratio, 4),
                "weighted_negative_strength": round(neg_weight_sum, 4),
                "weighted_positive_strength": round(pos_weight_sum, 4),
                "estimated_type": est_type,
                "baseline_avg_score": round(avg_legacy, 4),
                "pairwise_stats": pair_stats,
                "rule": {
                    "dynamic_edge_threshold": dyn_edge_thr,
                    "min_joint_cases": min_joint_cases,
                    "min_strength": min_strength,
                },
                "reason": "多后继节点 + 关系强度显著（含样本支持），是复杂分支优先研究位置",
            }
        )

    return candidates


def extract_loop_candidates(
    dfg: Dict[Tuple[str, str], int],
    loop_sensitivity: float = 0.8,
    min_loop_freq: int = 2,
) -> List[dict]:
    loops = []
    dyn_loop_thr = max(min_loop_freq, edge_sensitivity_threshold(dfg, loop_sensitivity))

    for (a, b), freq in dfg.items():
        if a == b and int(freq) >= dyn_loop_thr:
            loops.append(
                {
                    "type": "self_loop",
                    "activities": [a],
                    "frequency": int(freq),
                    "rule": {"dynamic_loop_threshold": dyn_loop_thr},
                    "reason": "自环频次达到 loop 阈值，可能是返工/重试约束",
                }
            )

    visited = set()
    for (a, b), f1 in dfg.items():
        if a == b:
            continue
        if (b, a) in dfg and (b, a) not in visited and (a, b) not in visited:
            f2 = dfg[(b, a)]
            if min(int(f1), int(f2)) >= dyn_loop_thr:
                loops.append(
                    {
                        "type": "two_activity_loop",
                        "activities": [a, b],
                        "frequency_ab": int(f1),
                        "frequency_ba": int(f2),
                        "rule": {"dynamic_loop_threshold": dyn_loop_thr},
                        "reason": "双向边均超过 loop 阈值，循环关系可信度较高",
                    }
                )
            visited.add((a, b))
            visited.add((b, a))

    return loops


def extract_correlation_candidates(
    case_activity_matrix: pd.DataFrame,
    dfg: Dict[Tuple[str, str], int],
    corr_scope: str = "dfg_local",
    min_joint_cases: int = 2,
    min_abs_phi: float = 0.2,
    min_lift_dev: float = 0.15,
    min_strength: float = 0.2,
) -> List[dict]:
    acts = list(case_activity_matrix.columns)
    candidates = []

    local_pairs = build_dfg_local_scope(dfg) if corr_scope == "dfg_local" else None
    baseline_candidates = 0

    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            a, b = acts[i], acts[j]
            pair_key = tuple(sorted((a, b)))
            if local_pairs is not None and pair_key not in local_pairs:
                continue

            x, y = case_activity_matrix[a], case_activity_matrix[b]
            n11, n10, n01, n00 = _contingency(x, y)
            base = legacy_score(n11, n10, n01)
            if abs(base) >= 0.2:
                baseline_candidates += 1

            if n11 < min_joint_cases:
                continue

            phi = phi_binary(x, y)
            lift = lift_binary(x, y)
            if abs(phi) < min_abs_phi and normalized_lift_deviation(lift) < min_lift_dev:
                continue

            strength = relation_strength(phi, lift)
            if strength < min_strength:
                continue

            candidates.append(
                {
                    "activities": [a, b],
                    "joint_cases": n11,
                    "phi": round(phi, 4),
                    "lift": round(lift, 4),
                    "norm_lift_dev": round(normalized_lift_deviation(lift), 4),
                    "strength": round(strength, 4),
                    "legacy_score": round(base, 4),
                    "relation": "positive" if phi > 0 else "negative",
                    "scope": corr_scope,
                    "reason": "通过 phi + normalized_lift + support + 结构约束筛选",
                }
            )

    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return candidates, baseline_candidates


def summarize_footprints(footprints: dict) -> dict:
    return {
        "start_activities": sorted(list(footprints.get("start_activities", []))),
        "end_activities": sorted(list(footprints.get("end_activities", []))),
        "sequence_count": len(footprints.get("sequence", [])),
        "parallel_count": len(footprints.get("parallel", [])),
    }


def model_stats(net, im, fm) -> dict:
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
        "initial_marking_tokens": int(sum(im.values())),
        "final_marking_tokens": int(sum(fm.values())),
    }


def safe_export_petri(net, im, fm, out_dir: Path, algo_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pnml_path = out_dir / f"{algo_name}.pnml"
    pnml_exporter.apply(net, im, pnml_path.as_posix(), final_marking=fm)

    image_path = None
    try:
        gviz = pn_visualizer.apply(net, im, fm)
        image_path = out_dir / f"{algo_name}.png"
        pn_visualizer.save(gviz, image_path.as_posix())
    except Exception:
        image_path = None

    return pnml_path.as_posix(), image_path.as_posix() if image_path else None


def discover_models(df: pd.DataFrame):
    models = {}

    tree = inductive_miner.apply(df)
    net_im, im_im, fm_im = tree_converter.apply(tree)
    models["inductive_miner"] = (net_im, im_im, fm_im)

    net_alpha, im_alpha, fm_alpha = alpha_miner.apply(
        df, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC
    )
    models["alpha_classic"] = (net_alpha, im_alpha, fm_alpha)

    try:
        net_alphap, im_alphap, fm_alphap = alpha_miner.apply(
            df, variant=alpha_miner.Variants.ALPHA_VERSION_PLUS
        )
        models["alpha_plus"] = (net_alphap, im_alphap, fm_alphap)
    except Exception:
        pass

    if heuristics_miner is not None:
        try:
            net_h, im_h, fm_h = heuristics_miner.apply_petri_net(df)
            models["heuristics_miner"] = (net_h, im_h, fm_h)
        except Exception:
            pass

    return models


def evaluate_model_fitness(df: pd.DataFrame, net, im, fm) -> dict:
    try:
        fit = replay_fitness.apply(df, net, im, fm)
        return {
            "log_fitness": round(float(fit.get("log_fitness", 0.0)), 4),
            "perc_fit_traces": round(float(fit.get("percentage_of_fitting_traces", 0.0)), 4),
        }
    except Exception:
        return {"log_fitness": None, "perc_fit_traces": None}


def run(
    csv_path: str,
    output_json: str,
    model_dir: str,
    edge_sensitivity: float,
    loop_sensitivity: float,
    corr_scope: str,
    min_joint_cases: int,
    min_strength: float,
):
    df = load_event_log(csv_path)
    dfg, footprints = discover_dfg_and_footprints(df)
    case_activity_matrix = get_case_activity_matrix(df)

    split_candidates = extract_split_candidates(
        dfg=dfg,
        case_activity_matrix=case_activity_matrix,
        edge_sensitivity=edge_sensitivity,
        min_joint_cases=min_joint_cases,
        min_strength=min_strength,
    )

    loop_candidates = extract_loop_candidates(
        dfg=dfg,
        loop_sensitivity=loop_sensitivity,
        min_loop_freq=min_joint_cases,
    )

    corr_candidates, corr_baseline_count = extract_correlation_candidates(
        case_activity_matrix=case_activity_matrix,
        dfg=dfg,
        corr_scope=corr_scope,
        min_joint_cases=min_joint_cases,
        min_strength=min_strength,
    )

    models = discover_models(df)
    model_results = {}
    out_dir = Path(model_dir)
    for name, (net, im, fm) in models.items():
        pnml_path, image_path = safe_export_petri(net, im, fm, out_dir, name)
        model_results[name] = {
            "petri_stats": model_stats(net, im, fm),
            "fitness": evaluate_model_fitness(df, net, im, fm),
            "pnml": pnml_path,
            "image": image_path,
        }

    duration_summary = None
    if df["duration_seconds"].notna().any():
        duration_summary = {
            "mean": round(float(df["duration_seconds"].mean()), 4),
            "median": round(float(df["duration_seconds"].median()), 4),
            "p90": round(float(df["duration_seconds"].quantile(0.9)), 4),
        }

    baseline_split_count = sum(1 for c in split_candidates if abs(c.get("baseline_avg_score", 0.0)) >= 0.2)

    result = {
        "dataset": csv_path,
        "n_events": int(len(df)),
        "n_cases": int(df["case:concept:name"].nunique()),
        "n_activities": int(df["concept:name"].nunique()),
        "duration_seconds_summary": duration_summary,
        "config": {
            "edge_sensitivity": edge_sensitivity,
            "loop_sensitivity": loop_sensitivity,
            "corr_scope": corr_scope,
            "min_joint_cases": min_joint_cases,
            "min_strength": min_strength,
        },
        "top_dfg_edges": [
            {"edge": [a, b], "freq": int(f)}
            for (a, b), f in sorted(dfg.items(), key=lambda x: -x[1])[:20]
        ],
        "footprints_summary": summarize_footprints(footprints),
        "candidates": {
            "split_candidates": split_candidates,
            "loop_candidates": loop_candidates,
            "correlation_candidates": corr_candidates,
        },
        "baseline_comparison": {
            "baseline_split_candidate_count": baseline_split_count,
            "baseline_corr_candidate_count": corr_baseline_count,
            "new_split_candidate_count": len(split_candidates),
            "new_corr_candidate_count": len(corr_candidates),
            "note": "baseline 使用 legacy_score(|score|>=0.2) 粗筛，仅用于对比候选规模与噪声趋势",
        },
        "model_comparison": model_results,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print("第一部分：复杂关系定位（phi+normalized_lift+support） + Petri 网多算法对比")
    print(f"数据集: {csv_path}")
    print(f"事件数: {result['n_events']}, 案例数: {result['n_cases']}, 活动数: {result['n_activities']}")
    print(
        f"edge_sensitivity={edge_sensitivity}, loop_sensitivity={loop_sensitivity}, "
        f"corr_scope={corr_scope}, min_joint_cases={min_joint_cases}, min_strength={min_strength}"
    )
    print(f"输出 JSON: {output_json}")
    print(f"Petri 网输出目录: {model_dir}")
    if duration_summary:
        print(f"平均持续时间(秒): {duration_summary['mean']}")
    print("-" * 92)
    print(f"split 候选数量: {len(split_candidates)}")
    print(f"loop 候选数量 : {len(loop_candidates)}")
    print(f"相关候选数量 : {len(corr_candidates)}")
    print(f"模型数量     : {len(model_results)}")
    print("=" * 92)


def parse_args():
    parser = argparse.ArgumentParser(description="基于 PM4Py 的复杂关系候选结构定位与 Petri 网对比")
    parser.add_argument("--csv", default="data/sample_process_log.csv", help="输入 CSV 路径")
    parser.add_argument("--out", default="outputs/structure_candidates.json", help="输出 JSON 路径")
    parser.add_argument("--model_dir", default="outputs/petri_nets", help="Petri 网输出目录（pnml/png）")
    parser.add_argument(
        "--edge_sensitivity",
        type=float,
        default=0.7,
        help="分支边筛选敏感度[0,1]，越高越保守（默认 0.7）",
    )
    parser.add_argument(
        "--loop_sensitivity",
        type=float,
        default=0.8,
        help="循环边筛选敏感度[0,1]，建议高于 edge_sensitivity（默认 0.8）",
    )
    parser.add_argument(
        "--corr_scope",
        choices=["global", "dfg_local"],
        default="dfg_local",
        help="相关分析范围：全局或 DFG 局部邻域（默认 dfg_local）",
    )
    parser.add_argument(
        "--min_joint_cases",
        type=int,
        default=2,
        help="联合出现最小案例数，低于该数量忽略（默认 2）",
    )
    parser.add_argument(
        "--min_strength",
        type=float,
        default=0.2,
        help="综合强度阈值（默认 0.2）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path=args.csv,
        output_json=args.out,
        model_dir=args.model_dir,
        edge_sensitivity=args.edge_sensitivity,
        loop_sensitivity=args.loop_sensitivity,
        corr_scope=args.corr_scope,
        min_joint_cases=args.min_joint_cases,
        min_strength=args.min_strength,
    )
