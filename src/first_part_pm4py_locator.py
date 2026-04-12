#!/usr/bin/env python3
"""
第一部分：基于 PM4Py 定位“可能存在复杂关系约束”的结构位置，
并输出多算法（IM / Alpha / Heuristics）的 Petri 网用于对比。

本版本重点增强：
1) 活动关系度量：联合使用 phi + lift + support（替代单一 score）。
2) 候选过滤：增加统计数量门槛 + 敏感度阈值，降低噪声候选。
3) 结构对比算法：新增 Heuristics Miner（若环境支持）。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
except Exception:  # pragma: no cover
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
    numerator = n11 * n00 - n10 * n01
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def lift_binary(x: pd.Series, y: pd.Series) -> float:
    n11, n10, n01, n00 = _contingency(x, y)
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 1.0

    p_x = (n11 + n10) / n
    p_y = (n11 + n01) / n
    p_xy = n11 / n

    if p_x == 0 or p_y == 0:
        return 1.0
    return p_xy / (p_x * p_y)


def relation_strength(phi: float, lift: float) -> float:
    """
    综合强度（0~1+）：考虑后续可扩展性，采用可解释的线性组合。
    - |phi|: 捕捉线性相关（正/负）
    - |log2(lift)|: 捕捉共现偏离独立性的幅度
    """
    phi_part = abs(phi)
    lift_part = abs(math.log2(max(lift, 1e-9)))
    return 0.6 * phi_part + 0.4 * lift_part


def edge_sensitivity_threshold(dfg: Dict[Tuple[str, str], int], sensitivity: float) -> int:
    freqs = sorted(int(v) for v in dfg.values())
    if not freqs:
        return 1
    sensitivity = min(max(sensitivity, 0.0), 1.0)
    idx = int(round((1.0 - sensitivity) * (len(freqs) - 1)))
    return max(1, freqs[idx])


def extract_split_candidates(
    dfg: Dict[Tuple[str, str], int],
    case_activity_matrix: pd.DataFrame,
    min_out_degree: int = 2,
    min_joint_cases: int = 2,
    sensitivity: float = 0.7,
    min_strength: float = 0.2,
) -> List[dict]:
    dynamic_edge_threshold = edge_sensitivity_threshold(dfg, sensitivity)

    outgoing = defaultdict(list)
    for (a, b), freq in dfg.items():
        if freq >= dynamic_edge_threshold:
            outgoing[a].append((b, int(freq)))

    candidates = []
    for src, targets in outgoing.items():
        if len(targets) < min_out_degree:
            continue

        tgt_names = [t for t, _ in targets]
        pair_stats = []
        valid_pair_count = 0

        for i in range(len(tgt_names)):
            for j in range(i + 1, len(tgt_names)):
                t1, t2 = tgt_names[i], tgt_names[j]
                x, y = case_activity_matrix[t1], case_activity_matrix[t2]
                n11, n10, n01, n00 = _contingency(x, y)
                phi = phi_binary(x, y)
                lift = lift_binary(x, y)
                strength = relation_strength(phi, lift)

                stat = {
                    "pair": [t1, t2],
                    "joint_cases": n11,
                    "only_left_cases": n10,
                    "only_right_cases": n01,
                    "none_cases": n00,
                    "phi": round(phi, 4),
                    "lift": round(lift, 4),
                    "strength": round(strength, 4),
                }
                pair_stats.append(stat)

                if n11 >= min_joint_cases and strength >= min_strength:
                    valid_pair_count += 1

        if not pair_stats:
            continue

        avg_strength = sum(p["strength"] for p in pair_stats) / len(pair_stats)
        avg_phi = sum(abs(p["phi"]) for p in pair_stats) / len(pair_stats)

        # 结构倾向：负 phi 较多 => XOR 倾向；正 phi + lift>1 较多 => AND/共现倾向
        neg_pairs = sum(1 for p in pair_stats if p["phi"] < 0)
        pos_assoc_pairs = sum(1 for p in pair_stats if p["phi"] > 0 and p["lift"] > 1)
        if neg_pairs > pos_assoc_pairs:
            est_type = "potential_XOR_or_mixed"
        else:
            est_type = "potential_AND_or_mixed"

        if valid_pair_count == 0:
            continue

        candidates.append(
            {
                "source": src,
                "outgoing": [{"target": t, "freq": f} for t, f in sorted(targets, key=lambda x: -x[1])],
                "avg_strength": round(avg_strength, 4),
                "avg_abs_phi": round(avg_phi, 4),
                "valid_pair_count": valid_pair_count,
                "estimated_type": est_type,
                "pairwise_stats": pair_stats,
                "rule": {
                    "dynamic_edge_threshold": dynamic_edge_threshold,
                    "min_joint_cases": min_joint_cases,
                    "min_strength": min_strength,
                },
                "reason": "多后继节点 + 显著活动对关系，是复杂分支的优先研究位置",
            }
        )

    return candidates


def extract_loop_candidates(
    dfg: Dict[Tuple[str, str], int],
    min_loop_freq: int = 2,
    sensitivity: float = 0.7,
) -> List[dict]:
    loops = []
    dynamic_edge_threshold = max(min_loop_freq, edge_sensitivity_threshold(dfg, sensitivity))

    for (a, b), freq in dfg.items():
        if a == b and freq >= dynamic_edge_threshold:
            loops.append(
                {
                    "type": "self_loop",
                    "activities": [a],
                    "frequency": int(freq),
                    "reason": "活动重复执行频次达到敏感度阈值，可能是返工/重试约束",
                }
            )

    visited = set()
    for (a, b), f1 in dfg.items():
        if a == b:
            continue
        if (b, a) in dfg and (b, a) not in visited and (a, b) not in visited:
            f2 = dfg[(b, a)]
            if min(f1, f2) >= dynamic_edge_threshold:
                loops.append(
                    {
                        "type": "two_activity_loop",
                        "activities": [a, b],
                        "frequency_ab": int(f1),
                        "frequency_ba": int(f2),
                        "reason": "双向边频次均达到阈值，循环/回退关系可信度较高",
                    }
                )
            visited.add((a, b))
            visited.add((b, a))

    return loops


def extract_correlation_candidates(
    case_activity_matrix: pd.DataFrame,
    min_joint_cases: int = 2,
    min_abs_phi: float = 0.2,
    min_lift_dev: float = 0.15,
    min_strength: float = 0.2,
) -> List[dict]:
    acts = list(case_activity_matrix.columns)
    candidates = []

    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            a, b = acts[i], acts[j]
            x, y = case_activity_matrix[a], case_activity_matrix[b]
            n11, n10, n01, n00 = _contingency(x, y)
            phi = phi_binary(x, y)
            lift = lift_binary(x, y)
            strength = relation_strength(phi, lift)

            if n11 < min_joint_cases:
                continue
            if abs(phi) < min_abs_phi and abs(lift - 1.0) < min_lift_dev:
                continue
            if strength < min_strength:
                continue

            candidates.append(
                {
                    "activities": [a, b],
                    "joint_cases": n11,
                    "phi": round(phi, 4),
                    "lift": round(lift, 4),
                    "strength": round(strength, 4),
                    "relation": "positive" if phi > 0 else "negative",
                    "reason": "通过 phi + lift + support 的综合阈值筛选得到",
                }
            )

    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return candidates


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

    # 新增：Heuristics Miner（某些 PM4Py 版本可能不提供 apply_petri_net）
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
    sensitivity: float,
    min_joint_cases: int,
    min_strength: float,
):
    df = load_event_log(csv_path)
    dfg, footprints = discover_dfg_and_footprints(df)
    case_activity_matrix = get_case_activity_matrix(df)

    split_candidates = extract_split_candidates(
        dfg,
        case_activity_matrix,
        min_joint_cases=min_joint_cases,
        sensitivity=sensitivity,
        min_strength=min_strength,
    )
    loop_candidates = extract_loop_candidates(
        dfg,
        min_loop_freq=min_joint_cases,
        sensitivity=sensitivity,
    )
    corr_candidates = extract_correlation_candidates(
        case_activity_matrix,
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

    result = {
        "dataset": csv_path,
        "n_events": int(len(df)),
        "n_cases": int(df["case:concept:name"].nunique()),
        "n_activities": int(df["concept:name"].nunique()),
        "duration_seconds_summary": duration_summary,
        "config": {
            "sensitivity": sensitivity,
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
        "model_comparison": model_results,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("第一部分：复杂关系结构定位（phi+lift+support） + Petri 网多算法对比")
    print(f"数据集: {csv_path}")
    print(f"事件数: {result['n_events']}, 案例数: {result['n_cases']}, 活动数: {result['n_activities']}")
    print(f"敏感度 sensitivity={sensitivity}, min_joint_cases={min_joint_cases}, min_strength={min_strength}")
    print(f"输出 JSON: {output_json}")
    print(f"Petri 网输出目录: {model_dir}")
    if duration_summary:
        print(f"平均持续时间(秒): {duration_summary['mean']}")
    print("-" * 88)
    print(f"split 候选数量: {len(split_candidates)}")
    print(f"loop 候选数量 : {len(loop_candidates)}")
    print(f"相关候选数量 : {len(corr_candidates)}")
    print(f"模型数量     : {len(model_results)}")
    print("=" * 88)


def parse_args():
    parser = argparse.ArgumentParser(description="基于 PM4Py 的复杂关系候选结构定位与 Petri 网对比")
    parser.add_argument("--csv", default="data/sample_process_log.csv", help="输入 CSV 路径")
    parser.add_argument("--out", default="outputs/structure_candidates.json", help="输出 JSON 路径")
    parser.add_argument("--model_dir", default="outputs/petri_nets", help="Petri 网输出目录（pnml/png）")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="敏感度阈值[0,1]，越高越保守（默认 0.7）",
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
        help="综合强度阈值（基于 phi 与 lift），低于则忽略（默认 0.2）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path=args.csv,
        output_json=args.out,
        model_dir=args.model_dir,
        sensitivity=args.sensitivity,
        min_joint_cases=args.min_joint_cases,
        min_strength=args.min_strength,
    )
