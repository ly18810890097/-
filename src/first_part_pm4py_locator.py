#!/usr/bin/env python3
"""
第一部分：基于 PM4Py 定位“可能存在复杂关系约束”的结构位置，
并输出多算法（IM / Alpha / Alpha+）的 Petri 网用于对比。
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
        # 无开始时间时置空，方便后续第二部分统一接口
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


def phi_binary(x: pd.Series, y: pd.Series) -> float:
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())

    numerator = n11 * n00 - n10 * n01
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def extract_split_candidates(
    dfg: Dict[Tuple[str, str], int],
    case_activity_matrix: pd.DataFrame,
    min_out_degree: int = 2,
    min_edge_freq: int = 1,
    and_threshold: float = 0.15,
) -> List[dict]:
    outgoing = defaultdict(list)
    for (a, b), freq in dfg.items():
        if freq >= min_edge_freq:
            outgoing[a].append((b, freq))

    candidates = []
    for src, targets in outgoing.items():
        if len(targets) < min_out_degree:
            continue

        tgt_names = [t for t, _ in targets]
        pair_stats = []
        for i in range(len(tgt_names)):
            for j in range(i + 1, len(tgt_names)):
                t1, t2 = tgt_names[i], tgt_names[j]
                both = int(((case_activity_matrix[t1] == 1) & (case_activity_matrix[t2] == 1)).sum())
                only_one = int(((case_activity_matrix[t1] != case_activity_matrix[t2])).sum())
                score = (both - only_one) / max(1, both + only_one)
                pair_stats.append(
                    {
                        "pair": [t1, t2],
                        "coexist_cases": both,
                        "xor_tendency_cases": only_one,
                        "split_relation_score": round(score, 4),
                    }
                )

        avg_score = sum(p["split_relation_score"] for p in pair_stats) / max(1, len(pair_stats))
        split_type = "potential_AND_or_mixed" if avg_score >= and_threshold else "potential_XOR_or_mixed"

        candidates.append(
            {
                "source": src,
                "outgoing": [{"target": t, "freq": f} for t, f in sorted(targets, key=lambda x: -x[1])],
                "avg_split_relation_score": round(avg_score, 4),
                "estimated_type": split_type,
                "pairwise_stats": pair_stats,
                "reason": "一个活动有多个后继，是复杂关系（并行/选择/混合）的优先检查位置",
            }
        )

    return candidates


def extract_loop_candidates(dfg: Dict[Tuple[str, str], int], min_loop_freq: int = 1) -> List[dict]:
    loops = []

    for (a, b), freq in dfg.items():
        if a == b and freq >= min_loop_freq:
            loops.append(
                {
                    "type": "self_loop",
                    "activities": [a],
                    "frequency": freq,
                    "reason": "活动存在重复执行，可能是返工或重试约束",
                }
            )

    visited = set()
    for (a, b), f1 in dfg.items():
        if a == b:
            continue
        if (b, a) in dfg and (b, a) not in visited and (a, b) not in visited:
            f2 = dfg[(b, a)]
            if f1 + f2 >= min_loop_freq:
                loops.append(
                    {
                        "type": "two_activity_loop",
                        "activities": [a, b],
                        "frequency_ab": f1,
                        "frequency_ba": f2,
                        "reason": "A->B 与 B->A 均出现，可能存在循环或回退关系",
                    }
                )
            visited.add((a, b))
            visited.add((b, a))

    return loops


def extract_correlation_candidates(
    case_activity_matrix: pd.DataFrame,
    positive_threshold: float = 0.25,
    negative_threshold: float = -0.25,
) -> List[dict]:
    acts = list(case_activity_matrix.columns)
    candidates = []

    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            a, b = acts[i], acts[j]
            phi = phi_binary(case_activity_matrix[a], case_activity_matrix[b])
            if phi >= positive_threshold or phi <= negative_threshold:
                candidates.append(
                    {
                        "activities": [a, b],
                        "phi": round(phi, 4),
                        "relation": "positive" if phi > 0 else "negative",
                        "reason": "活动同现/互斥倾向明显，是异常关系约束定位的候选",
                    }
                )

    candidates.sort(key=lambda x: abs(x["phi"]), reverse=True)
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
        "initial_marking_tokens": sum(im.values()),
        "final_marking_tokens": sum(fm.values()),
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
        # 若环境缺 graphviz 等，仅保证 pnml 可用
        image_path = None

    return pnml_path.as_posix(), image_path.as_posix() if image_path else None


def discover_models(df: pd.DataFrame):
    models = {}

    # 1) IM
    tree = inductive_miner.apply(df)
    net_im, im_im, fm_im = tree_converter.apply(tree)
    models["inductive_miner"] = (net_im, im_im, fm_im)

    # 2) Alpha classic
    net_alpha, im_alpha, fm_alpha = alpha_miner.apply(
        df, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC
    )
    models["alpha_classic"] = (net_alpha, im_alpha, fm_alpha)

    # 3) Alpha+
    try:
        net_alphap, im_alphap, fm_alphap = alpha_miner.apply(
            df, variant=alpha_miner.Variants.ALPHA_VERSION_PLUS
        )
        models["alpha_plus"] = (net_alphap, im_alphap, fm_alphap)
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


def run(csv_path: str, output_json: str, model_dir: str):
    df = load_event_log(csv_path)
    dfg, footprints = discover_dfg_and_footprints(df)
    case_activity_matrix = get_case_activity_matrix(df)

    split_candidates = extract_split_candidates(dfg, case_activity_matrix)
    loop_candidates = extract_loop_candidates(dfg)
    corr_candidates = extract_correlation_candidates(case_activity_matrix)

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

    print("=" * 80)
    print("第一部分：复杂关系结构候选定位 + Petri 网对比")
    print(f"数据集: {csv_path}")
    print(f"事件数: {result['n_events']}, 案例数: {result['n_cases']}, 活动数: {result['n_activities']}")
    print(f"输出 JSON: {output_json}")
    print(f"Petri 网输出目录: {model_dir}")
    if duration_summary:
        print(f"平均持续时间(秒): {duration_summary['mean']}")
    print("-" * 80)
    print(f"split 候选数量: {len(split_candidates)}")
    print(f"loop 候选数量 : {len(loop_candidates)}")
    print(f"相关候选数量 : {len(corr_candidates)}")
    print(f"模型数量     : {len(model_results)}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="基于 PM4Py 的复杂关系候选结构定位与 Petri 网对比")
    parser.add_argument("--csv", default="data/sample_process_log.csv", help="输入 CSV 路径")
    parser.add_argument(
        "--out",
        default="outputs/structure_candidates.json",
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--model_dir",
        default="outputs/petri_nets",
        help="Petri 网输出目录（pnml/png）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.csv, args.out, args.model_dir)
