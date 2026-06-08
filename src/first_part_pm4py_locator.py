#!/usr/bin/env python3
"""第一部分：基于 trace 统计的关系判定与候选约束空间生成。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["case:concept:name", "concept:name", "time:timestamp"]


def jaccard(a: Set[str], b: Set[str]) -> float:
    u = a.union(b)
    if not u:
        return 0.0
    return len(a.intersection(b)) / len(u)


def load_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    if "start_timestamp" in df.columns:
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True, errors="coerce")

    # 稳定排序：同时间戳按原始顺序
    df["_orig_order"] = range(len(df))
    df = df.sort_values(["case:concept:name", "time:timestamp", "_orig_order"]).reset_index(drop=True)
    return df


def build_traces(df: pd.DataFrame) -> Tuple[List[List[str]], List[str], Dict[str, List[str]]]:
    traces = []
    case_ids = []
    case_to_trace = {}
    for cid, g in df.groupby("case:concept:name", sort=False):
        tr = g["concept:name"].tolist()
        traces.append(tr)
        case_ids.append(cid)
        case_to_trace[cid] = tr
    return traces, case_ids, case_to_trace


def compute_behavior_stats(df: pd.DataFrame, traces: List[List[str]], activities: List[str]) -> dict:
    activity_set = set(activities)
    dfg_count = defaultdict(int)
    cooccur_cases = defaultdict(int)
    cases_with_A = defaultdict(int)
    has_AB = defaultdict(bool)
    has_BA = defaultdict(bool)
    loop_evidence = defaultdict(bool)
    self_loop = defaultdict(bool)
    pred_set = {a: set() for a in activities}
    succ_set = {a: set() for a in activities}
    time_overlap = defaultdict(int)

    # case-level 出现
    for tr in traces:
        uniq = set(tr)
        for a in uniq:
            cases_with_A[a] += 1
        for a, b in combinations(sorted(uniq), 2):
            cooccur_cases[(a, b)] += 1
            cooccur_cases[(b, a)] += 1

    # trace-level 顺序与循环证据
    for tr in traces:
        for i in range(len(tr) - 1):
            a, b = tr[i], tr[i + 1]
            dfg_count[(a, b)] += 1
            has_AB[(a, b)] = True
            has_BA[(b, a)] = True
            pred_set[b].add(a)
            succ_set[a].add(b)

        cnt = defaultdict(int)
        for a in tr:
            cnt[a] += 1
        for a, c in cnt.items():
            if c >= 2:
                self_loop[a] = True

        for i in range(len(tr) - 2):
            a, b, c = tr[i], tr[i + 1], tr[i + 2]
            if a == c and a != b:
                loop_evidence[(a, b)] = True
                loop_evidence[(b, a)] = True

    # 可选时间重叠
    if "start_timestamp" in df.columns:
        for _, g in df.groupby("case:concept:name", sort=False):
            rows = g[["concept:name", "start_timestamp", "time:timestamp"]].values.tolist()
            for i in range(len(rows)):
                ai, si, ei = rows[i]
                for j in range(i + 1, len(rows)):
                    aj, sj, ej = rows[j]
                    if ai == aj:
                        continue
                    if pd.isna(si) or pd.isna(sj) or pd.isna(ei) or pd.isna(ej):
                        continue
                    if si < ej and sj < ei:
                        key1 = (ai, aj)
                        key2 = (aj, ai)
                        time_overlap[key1] += 1
                        time_overlap[key2] += 1

    return {
        "dfg_count": dfg_count,
        "cooccur_cases": cooccur_cases,
        "cases_with_A": cases_with_A,
        "has_AB": has_AB,
        "has_BA": has_BA,
        "loop_evidence": loop_evidence,
        "self_loop": self_loop,
        "pred_set": pred_set,
        "succ_set": succ_set,
        "time_overlap": time_overlap,
        "activity_set": activity_set,
    }


def phi_case_level(case_matrix: pd.DataFrame, a: str, b: str) -> float:
    x, y = case_matrix[a], case_matrix[b]
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    if den == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / den


def relation_type_for_pair(a: str, b: str, stats: dict, n_cases: int, theta1=0.6, theta2=0.5, theta3=0.1) -> Tuple[str, dict]:
    has_ab = stats["has_AB"].get((a, b), False)
    has_ba = stats["has_AB"].get((b, a), False)
    co = stats["cooccur_cases"].get((a, b), 0)
    ca = stats["cases_with_A"].get(a, 0)
    cb = stats["cases_with_A"].get(b, 0)
    co_ratio = co / max(1, min(ca, cb))
    pred_sim = jaccard(stats["pred_set"][a], stats["pred_set"][b])
    succ_sim = jaccard(stats["succ_set"][a], stats["succ_set"][b])

    ev = {
        "has_AB": bool(has_ab),
        "has_BA": bool(has_ba),
        "cooccur_cases": int(co),
        "loop_evidence": bool(stats["loop_evidence"].get((a, b), False)),
        "pred_similarity": round(pred_sim, 4),
        "succ_similarity": round(succ_sim, 4),
    }

    # 1) loop
    if stats["self_loop"].get(a, False) or stats["self_loop"].get(b, False) or stats["loop_evidence"].get((a, b), False):
        return "loop", ev

    # 2) parallel
    if has_ab and has_ba and co_ratio >= theta1 and pred_sim >= theta2 and succ_sim >= theta2:
        return "parallel", ev

    # 3) choice
    common_upstream = len(stats["pred_set"][a].intersection(stats["pred_set"][b])) > 0
    if (co == 0 or (co / max(1, n_cases)) <= theta3) and common_upstream:
        return "choice", ev

    # 4) sequence
    if has_ab and not has_ba:
        return "sequence", ev

    return "sequence", ev


def build_blocks(case_matrix: pd.DataFrame, activities: List[str], stats: dict, n_cases: int, theta4=0.8, phi_thr=0.4):
    parent = {a: a for a in activities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in combinations(activities, 2):
        co = stats["cooccur_cases"].get((a, b), 0)
        ca = stats["cases_with_A"].get(a, 0)
        cb = stats["cases_with_A"].get(b, 0)
        if co < theta4 * min(ca, cb):
            continue
        phi = abs(phi_case_level(case_matrix, a, b)) if a in case_matrix.columns and b in case_matrix.columns else 0.0
        if phi >= phi_thr:
            union(a, b)

    groups = defaultdict(list)
    for a in activities:
        groups[find(a)].append(a)

    blocks = []
    block_map = {}
    idx = 1
    for members in groups.values():
        if len(members) >= 2:
            bid = f"blk_{idx}"
            idx += 1
            blocks.append({"id": bid, "members": sorted(members), "reason": "high_cooccurrence_and_strong_correlation"})
            for m in members:
                block_map[m] = bid
    return blocks, block_map


def replace_with_blocks(trace: List[str], block_map: Dict[str, str]) -> List[str]:
    out = []
    for a in trace:
        out.append(block_map.get(a, a))
    # 压缩连续重复节点
    compact = []
    for x in out:
        if not compact or compact[-1] != x:
            compact.append(x)
    return compact


def detect_splits(traces: List[List[str]], block_map: Dict[str, str], pair_types: Dict[Tuple[str, str], str]) -> List[dict]:
    succ = defaultdict(set)
    for tr in traces:
        rt = replace_with_blocks(tr, block_map)
        for i in range(len(rt) - 1):
            succ[rt[i]].add(rt[i + 1])

    splits = []
    for src, tgts in succ.items():
        if len(tgts) < 2:
            continue
        tlist = sorted(tgts)
        # split 类型：根据 targets 间关系投票
        votes = {"xor": 0, "and": 0, "mixed": 0}
        for a, b in combinations(tlist, 2):
            # block id 无法直接查pair，视作 mixed
            if a.startswith("blk_") or b.startswith("blk_"):
                votes["mixed"] += 1
                continue
            t = pair_types.get((a, b)) or pair_types.get((b, a))
            if t == "choice":
                votes["xor"] += 1
            elif t == "parallel":
                votes["and"] += 1
            else:
                votes["mixed"] += 1
        split_type = max(votes.items(), key=lambda x: x[1])[0]
        splits.append({"source": src, "targets": tlist, "type": split_type})
    return splits


def run(csv_path: str, out_json: str, theta1=0.6, theta2=0.5, theta3=0.1):
    df = load_log(csv_path)
    traces, case_ids, _ = build_traces(df)
    activities = sorted(df["concept:name"].unique().tolist())
    n_cases = len(case_ids)

    stats = compute_behavior_stats(df, traces, activities)

    # case-level matrix for block building
    case_matrix = (df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0) > 0).astype(int)

    relations = []
    pair_types = {}
    loops = []
    loop_nodes = set()

    for a, b in combinations(activities, 2):
        rtype, ev = relation_type_for_pair(a, b, stats, n_cases, theta1=theta1, theta2=theta2, theta3=theta3)
        pair_types[(a, b)] = rtype
        pair_types[(b, a)] = rtype

        relations.append({"A": a, "B": b, "type": rtype, "evidence": ev})

        if rtype == "loop":
            loop_nodes.add(a)
            loop_nodes.add(b)
            if stats["self_loop"].get(a, False):
                loops.append({"type": "self", "activities": [a]})
            if stats["self_loop"].get(b, False):
                loops.append({"type": "self", "activities": [b]})
            if stats["loop_evidence"].get((a, b), False):
                loops.append({"type": "two_activity", "activities": [a, b]})

    # 去重 loops
    loops_unique = []
    seen = set()
    for l in loops:
        key = (l["type"], tuple(l["activities"]))
        if key not in seen:
            seen.add(key)
            loops_unique.append(l)

    blocks, block_map = build_blocks(case_matrix, activities, stats, n_cases)
    splits = detect_splits(traces, block_map, pair_types)

    directed_relations = []
    for r in relations:
        a, b, t = r["A"], r["B"], r["type"]
        ev = r["evidence"]
        if ev["has_AB"]:
            directed_relations.append({"A": a, "B": b, "type": t, "allowed": True})
        if ev["has_BA"]:
            directed_relations.append({"A": b, "B": a, "type": t, "allowed": True})

    result = {
        "dataset": csv_path,
        "n_cases": int(n_cases),
        "n_events": int(len(df)),
        "activities": activities,
        "relations": relations,
        "blocks": blocks,
        "splits": splits,
        "loops": loops_unique,
        "candidate_constraint_space": {
            "directed_relations": directed_relations,
            "branch_scopes": splits,
            "loop_nodes": sorted(loop_nodes),
        },
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"cases={n_cases}, events={len(df)}, activities={len(activities)}")
    print(f"relations={len(relations)}, blocks={len(blocks)}, splits={len(splits)}, loops={len(loops_unique)}")
    print(f"output={out_json}")


def parse_args():
    p = argparse.ArgumentParser(description="第一部分：trace统计关系判定与候选空间生成")
    p.add_argument("--csv", default="data/sample_process_log.csv")
    p.add_argument("--out", default="outputs/structure_candidates.json")
    p.add_argument("--theta1", type=float, default=0.6)
    p.add_argument("--theta2", type=float, default=0.5)
    p.add_argument("--theta3", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.csv, args.out, theta1=args.theta1, theta2=args.theta2, theta3=args.theta3)
