#!/usr/bin/env python3
"""
第一部分：结构候选生成（含“候选约束空间”）。

目标：
1) 发现流程结构（DFG/footprints/Petri）。
2) 输出结构化候选约束空间，供第二部分 dm4py 缩小搜索范围。
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
    m = df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0)
    return (m > 0).astype(int)


def _contingency(x: pd.Series, y: pd.Series):
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


def normalized_lift_deviation(lift: float) -> float:
    lift = max(0.0, lift)
    return abs(lift - 1.0) / (lift + 1.0)


def relation_strength(phi: float, lift: float) -> float:
    return 0.6 * abs(phi) + 0.4 * normalized_lift_deviation(lift)


def dynamic_threshold_from_freqs(freqs: List[int], sensitivity: float) -> int:
    if not freqs:
        return 1
    sensitivity = min(max(sensitivity, 0.0), 1.0)
    sorted_freqs = sorted(int(v) for v in freqs)
    idx = int(round((1.0 - sensitivity) * (len(sorted_freqs) - 1)))
    return max(1, sorted_freqs[idx])


def extract_split_candidates(
    dfg: Dict[Tuple[str, str], int],
    case_activity_matrix: pd.DataFrame,
    edge_sensitivity: float = 0.7,
    min_out_degree: int = 2,
):
    edge_thr = dynamic_threshold_from_freqs(list(dfg.values()), edge_sensitivity)
    outgoing = defaultdict(list)
    for (a, b), f in dfg.items():
        if int(f) >= edge_thr:
            outgoing[a].append((b, int(f)))

    result = []
    for src, targets in outgoing.items():
        if len(targets) < min_out_degree:
            continue
        tgt_names = [t for t, _ in targets]
        pairwise = []
        coexist_rates = []
        for i in range(len(tgt_names)):
            for j in range(i + 1, len(tgt_names)):
                a, b = tgt_names[i], tgt_names[j]
                x, y = case_activity_matrix[a], case_activity_matrix[b]
                n11, n10, n01, n00 = _contingency(x, y)
                total = max(1, n11 + n10 + n01 + n00)
                coexist = n11 / total
                coexist_rates.append(coexist)
                phi = phi_binary(x, y) if n11 > 0 else 0.0
                lift = lift_binary(x, y) if n11 > 0 else 0.0
                pairwise.append(
                    {
                        "pair": [a, b],
                        "joint_cases": n11,
                        "coexist_ratio": round(coexist, 4),
                        "phi": round(phi, 4),
                        "lift": round(lift, 4),
                        "strength": round(relation_strength(phi, lift), 4) if n11 > 0 else 0.0,
                    }
                )

        avg_coexist = sum(coexist_rates) / max(1, len(coexist_rates))
        if avg_coexist <= 0.2:
            hint = "xor_branch"
        elif avg_coexist >= 0.5:
            hint = "and_branch"
        else:
            hint = "mixed_branch"

        result.append(
            {
                "source": src,
                "outgoing": [{"target": t, "freq": f} for t, f in sorted(targets, key=lambda z: -z[1])],
                "avg_coexist_ratio": round(avg_coexist, 4),
                "structure_hint": hint,
                "pairwise_stats": pairwise,
                "rule": {"dynamic_edge_threshold": edge_thr},
            }
        )

    return result


def extract_loop_candidates(dfg: Dict[Tuple[str, str], int], loop_sensitivity: float = 0.8):
    thr = dynamic_threshold_from_freqs(list(dfg.values()), loop_sensitivity)
    loops = []
    visited = set()
    for (a, b), f1 in dfg.items():
        if a == b and int(f1) >= thr:
            loops.append({"type": "self_loop", "activities": [a], "frequency": int(f1)})
            continue
        if (b, a) in dfg and (a, b) not in visited and (b, a) not in visited:
            f2 = int(dfg[(b, a)])
            if min(int(f1), f2) >= thr:
                loops.append(
                    {
                        "type": "two_activity_loop",
                        "activities": [a, b],
                        "frequency_ab": int(f1),
                        "frequency_ba": int(f2),
                    }
                )
            visited.add((a, b))
            visited.add((b, a))
    return loops


def build_constraint_search_space(
    dfg: Dict[Tuple[str, str], int],
    split_candidates: List[dict],
    loop_candidates: List[dict],
):
    split_sources = {s["source"] for s in split_candidates}
    split_edges = {(s["source"], o["target"]) for s in split_candidates for o in s.get("outgoing", [])}

    loop_edges = set()
    loop_nodes = set()
    for l in loop_candidates:
        acts = l.get("activities", [])
        if len(acts) == 1:
            loop_edges.add((acts[0], acts[0]))
            loop_nodes.add(acts[0])
        elif len(acts) == 2:
            a, b = acts
            loop_edges.add((a, b))
            loop_edges.add((b, a))
            loop_nodes.add(a)
            loop_nodes.add(b)

    directed_relations = []
    for (a, b), f in sorted(dfg.items(), key=lambda x: -x[1]):
        directed_relations.append(
            {
                "source": a,
                "target": b,
                "frequency": int(f),
                "bidirectional": (b, a) in dfg,
                "from_split": (a, b) in split_edges,
                "from_loop": (a, b) in loop_edges,
            }
        )

    branch_scopes = []
    for s in split_candidates:
        branch_scopes.append(
            {
                "split_source": s["source"],
                "targets": [o["target"] for o in s.get("outgoing", [])],
                "structure_hint": s.get("structure_hint", "mixed_branch"),
                "avg_coexist_ratio": s.get("avg_coexist_ratio"),
            }
        )

    return {
        "directed_relations": directed_relations,
        "branch_scopes": branch_scopes,
        "loop_nodes": sorted(loop_nodes),
        "summary": {
            "n_directed_relations": len(directed_relations),
            "n_branch_scopes": len(branch_scopes),
            "n_loop_nodes": len(loop_nodes),
            "n_split_sources": len(split_sources),
        },
    }


def summarize_footprints(footprints: dict):
    return {
        "start_activities": sorted(list(footprints.get("start_activities", []))),
        "end_activities": sorted(list(footprints.get("end_activities", []))),
        "sequence_count": len(footprints.get("sequence", [])),
        "parallel_count": len(footprints.get("parallel", [])),
    }


def model_stats(net, im, fm):
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
    models["inductive_miner"] = tree_converter.apply(tree)
    models["alpha_classic"] = alpha_miner.apply(df, variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC)
    try:
        models["alpha_plus"] = alpha_miner.apply(df, variant=alpha_miner.Variants.ALPHA_VERSION_PLUS)
    except Exception:
        pass
    if heuristics_miner is not None:
        try:
            models["heuristics_miner"] = heuristics_miner.apply_petri_net(df)
        except Exception:
            pass
    return models


def eval_fitness(df: pd.DataFrame, net, im, fm):
    try:
        fit = replay_fitness.apply(df, net, im, fm)
        return {
            "log_fitness": round(float(fit.get("log_fitness", 0.0)), 4),
            "perc_fit_traces": round(float(fit.get("percentage_of_fitting_traces", 0.0)), 4),
        }
    except Exception:
        return {"log_fitness": None, "perc_fit_traces": None}


def run(csv_path: str, output_json: str, model_dir: str, edge_sensitivity: float, loop_sensitivity: float):
    df = load_event_log(csv_path)
    dfg, footprints = discover_dfg_and_footprints(df)
    case_m = get_case_activity_matrix(df)

    split_candidates = extract_split_candidates(dfg, case_m, edge_sensitivity=edge_sensitivity)
    loop_candidates = extract_loop_candidates(dfg, loop_sensitivity=loop_sensitivity)
    constraint_space = build_constraint_search_space(dfg, split_candidates, loop_candidates)

    models = discover_models(df)
    model_results = {}
    out_dir = Path(model_dir)
    for name, (net, im, fm) in models.items():
        pnml, png = safe_export_petri(net, im, fm, out_dir, name)
        model_results[name] = {
            "petri_stats": model_stats(net, im, fm),
            "fitness": eval_fitness(df, net, im, fm),
            "pnml": pnml,
            "image": png,
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
            "edge_sensitivity": edge_sensitivity,
            "loop_sensitivity": loop_sensitivity,
        },
        "top_dfg_edges": [
            {"edge": [a, b], "freq": int(f)}
            for (a, b), f in sorted(dfg.items(), key=lambda x: -x[1])[:20]
        ],
        "footprints_summary": summarize_footprints(footprints),
        "candidates": {
            "split_candidates": split_candidates,
            "loop_candidates": loop_candidates,
        },
        "candidate_constraint_space": constraint_space,
        "model_comparison": model_results,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("第一部分：结构候选生成 + 候选约束空间输出")
    print(f"数据集: {csv_path}")
    print(f"输出 JSON: {output_json}")
    print(f"Petri 网输出目录: {model_dir}")
    print(f"split 候选: {len(split_candidates)}, loop 候选: {len(loop_candidates)}")
    print(f"候选有向关系数: {constraint_space['summary']['n_directed_relations']}")
    print("=" * 88)


def parse_args():
    p = argparse.ArgumentParser(description="第一部分：结构候选 + 候选约束空间")
    p.add_argument("--csv", default="data/sample_process_log.csv")
    p.add_argument("--out", default="outputs/structure_candidates.json")
    p.add_argument("--model_dir", default="outputs/petri_nets")
    p.add_argument("--edge_sensitivity", type=float, default=0.7)
    p.add_argument("--loop_sensitivity", type=float, default=0.8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path=args.csv,
        output_json=args.out,
        model_dir=args.model_dir,
        edge_sensitivity=args.edge_sensitivity,
        loop_sensitivity=args.loop_sensitivity,
    )
