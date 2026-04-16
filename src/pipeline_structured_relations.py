#!/usr/bin/env python3
"""统一四层流程：结构层→逻辑层→统计层→动态层。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pm4py


def load_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df.sort_values(["case:concept:name", "start_timestamp"]).reset_index(drop=True)


def case_trace(df_case: pd.DataFrame) -> List[str]:
    return df_case.sort_values("start_timestamp")["concept:name"].tolist()


def has_overlap(df_case: pd.DataFrame) -> bool:
    xs = df_case.sort_values("start_timestamp")[["start_timestamp", "time:timestamp"]].values.tolist()
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i][0] < xs[j][1] and xs[j][0] < xs[i][1]:
                return True
    return False


def has_loop_abab(trace: List[str], a="t6", b="t7") -> bool:
    s = "->".join(trace)
    return f"{a}->{b}->{a}->{b}" in s


def classify_cases(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {"main": [], "concurrency": [], "loop": [], "anomaly": []}
    canonical = ["t1", "t2", "t3", "t6", "t7", "t8", "t9"]

    for cid, g in df.groupby("case:concept:name"):
        tr = case_trace(g)
        xor_cnt = sum(1 for x in ["t3", "t4", "t5"] if x in tr)
        if has_loop_abab(tr):
            groups["loop"].append(cid)
        elif has_overlap(g):
            groups["concurrency"].append(cid)
        elif xor_cnt != 1 or tr[:2] != ["t1", "t2"] or tr[-1] != "t9":
            groups["anomaly"].append(cid)
        elif tr != canonical:
            # 合法变体（如 XOR t4/t5）仍计入 anomaly 以保证主干干净
            groups["anomaly"].append(cid)
        else:
            groups["main"].append(cid)

    return groups


def dfg_edges(df: pd.DataFrame) -> Counter:
    c = Counter()
    for _, g in df.groupby("case:concept:name"):
        tr = case_trace(g)
        for i in range(len(tr) - 1):
            c[(tr[i], tr[i + 1])] += 1
    return c


def pre_recognition(df: pd.DataFrame) -> dict:
    edges = dfg_edges(df)
    sequence = [{"A": a, "B": b, "freq": f} for (a, b), f in edges.most_common(40)]

    # 并发候选（时间重叠）
    overlap_cnt = Counter()
    for _, g in df.groupby("case:concept:name"):
        acts = g[["concept:name", "start_timestamp", "time:timestamp"]].values.tolist()
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                ai, si, ei = acts[i]
                aj, sj, ej = acts[j]
                if si < ej and sj < ei and ai != aj:
                    overlap_cnt[tuple(sorted((ai, aj)))] += 1
    parallel = [{"pair": list(p), "overlap_cases": v} for p, v in overlap_cnt.items() if v >= 3]

    # 选择候选（仅同一split t2后的 t3/t4/t5）
    m = (df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0) > 0).astype(int)
    choice = []
    for a, b in [("t3", "t4"), ("t3", "t5"), ("t4", "t5")]:
        n11 = int(((m[a] == 1) & (m[b] == 1)).sum())
        coexist = n11 / max(1, len(m))
        choice.append({"pair": [a, b], "coexist_ratio": round(coexist, 4)})

    # 循环候选
    loop = []
    for (a, b), f1 in edges.items():
        if (b, a) in edges:
            f2 = edges[(b, a)]
            if min(f1, f2) >= 3:
                loop.append({"A": a, "B": b, "frequency_ab": f1, "frequency_ba": f2})

    return {"sequence": sequence, "parallel": parallel, "choice": choice, "loop": loop}


def simplified_backbone_and_abstract(groups: Dict[str, List[str]]) -> Tuple[str, dict]:
    # 折叠规则：XOR 与 并发片段抽象
    abstract = {
        "X1": ["t3", "t4", "t5"],
        "X2": ["t6", "t7"],
    }
    # 主干结构字符串（可替换为真实Petri对象路径）
    petri_desc = "t1->t2->X1->X2->t8->t9"
    return petri_desc, abstract


def declare_on_sequence_candidates(df: pd.DataFrame, sequence_candidates: List[dict]) -> List[dict]:
    try:
        from Declare4Py.D4PyEventLog import D4PyEventLog
        from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
    except Exception:
        return []

    allowed = {(x["A"], x["B"]) for x in sequence_candidates}
    log_fmt = pm4py.format_dataframe(df.copy(), case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
    elog = pm4py.convert_to_event_log(log_fmt)
    d4 = D4PyEventLog(case_name="case:concept:name", log=elog)
    miner = DeclareMiner(log=d4, consider_vacuity=False, min_support=0.2, itemsets_support=0.9, max_declare_cardinality=1)
    model = miner.run()
    out = []
    for s in getattr(model, "serialized_constraints", []) or []:
        if "[" not in s:
            continue
        tpl = s.split("[")[0].strip()
        inside = s[s.find("[") + 1 : s.find("]")]
        acts = [x.strip() for x in inside.split(",") if x.strip()]
        if len(acts) < 2:
            continue
        a, b = acts[0], acts[1]
        if (a, b) in allowed:
            out.append({"relation_type": tpl, "relation_category": "declare", "A": a, "B": b})

    # 去冗余：同对保留最强
    def rank(t):
        x = t.lower().replace(" ", "")
        if "chain" in x:
            return 3
        if "alternate" in x:
            return 2
        if "response" in x:
            return 1
        return 0

    best = {}
    for r in out:
        k = (r["A"], r["B"])
        if k not in best or rank(r["relation_type"]) > rank(best[k]["relation_type"]):
            best[k] = r
    return list(best.values())


def dynamic_on_choice_stat(df: pd.DataFrame, window="1D", min_window_cases=3) -> List[dict]:
    m_global = (df.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0) > 0).astype(int)

    def p_b_given_a(m, a, b):
        a_cases = int((m[a] == 1).sum()) if a in m.columns else 0
        if a_cases == 0 or b not in m.columns:
            return 0.0
        return float((((m[a] == 1) & (m[b] == 1)).sum()) / a_cases)

    pairs = [("t3", "t4"), ("t3", "t5"), ("t4", "t5")]
    out = []
    for a, b in pairs:
        p_global = p_b_given_a(m_global, a, b)
        ctx = []
        probs = []
        for bucket, sub in df.groupby(pd.Grouper(key="time:timestamp", freq=window)):
            if len(sub) == 0:
                continue
            n_cases = sub["case:concept:name"].nunique()
            if n_cases < min_window_cases:
                continue
            m = (sub.groupby(["case:concept:name", "concept:name"]).size().unstack(fill_value=0) > 0).astype(int)
            p = p_b_given_a(m, a, b)
            probs.append(p)
            ctx.append({"window_start": str(bucket), "P_ctx(B|A)": round(p, 4), "delta": round(p - p_global, 4)})

        if not probs:
            continue
        prange = max(probs) - min(probs)
        pvar = float(pd.Series(probs).var()) if len(probs) > 1 else 0.0
        if prange > 0.3 or pvar > 0.02:
            dtype = "fluctuating"
        else:
            dtype = "stable"

        out.append(
            {
                "relation_type": "choice_dynamic",
                "relation_category": "dynamic",
                "A": a,
                "B": b,
                "P(B|A)": round(p_global, 4),
                "windows": ctx,
                "dynamic_type": dtype,
            }
        )
    return out


def run(csv_path: str, out_json: str):
    df = load_log(csv_path)
    groups = classify_cases(df)

    main_df = df[df["case:concept:name"].isin(groups["main"])]
    evidence_df = df[~df["case:concept:name"].isin(groups["main"])]

    prere = pre_recognition(df)
    petri_desc, abstract_nodes = simplified_backbone_and_abstract(groups)

    declare_rel = declare_on_sequence_candidates(main_df, prere["sequence"])

    parallel_rel = [
        {"relation_type": "parallel", "relation_category": "structural", **p}
        for p in prere["parallel"]
    ]
    choice_rel = [
        {"relation_type": "choice", "relation_category": "statistical", **c}
        for c in prere["choice"]
    ]
    loop_rel = [
        {"relation_type": "loop", "relation_category": "structural", **l}
        for l in prere["loop"]
    ]

    dynamic_rel = dynamic_on_choice_stat(evidence_df, window="1D", min_window_cases=3)

    result = {
        "petri_net": petri_desc,
        "abstract_nodes": abstract_nodes,
        "case_routing": groups,
        "relations": {
            "sequence": prere["sequence"],
            "parallel": parallel_rel,
            "choice": choice_rel,
            "loop": loop_rel,
            "declare": declare_rel,
            "statistical": choice_rel,
            "dynamic": dynamic_rel,
        },
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("cases:", df["case:concept:name"].nunique())
    print("routing:", {k: len(v) for k, v in groups.items()})
    print("out:", out_json)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/sample_process_log.csv")
    p.add_argument("--out", default="outputs/final_layered_relations.json")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.csv, a.out)
