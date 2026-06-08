#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Context-aware Relation Mining Framework

支持关系：

1. Loop
2. PR
3. NR
4. NRwAP

====================================================

统一框架：

1. 日志读取
2. traces构建
3. Loop挖掘
4. XOR上下文关系挖掘
5. 统一JSON输出

====================================================
"""

from __future__ import annotations

import argparse
import csv
import json

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing import Dict
from typing import List
from typing import Sequence
from typing import Optional


# =========================================================
# 时间解析
# =========================================================

def _parse_ts(value: str) -> datetime:

    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


# =========================================================
# 读取日志
# =========================================================

def load_event_log(path: Path) -> List[dict]:

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    expected = {
        "concept:name",
        "time:timestamp",
        "case:concept:name"
    }

    missing = expected.difference(
        rows[0].keys() if rows else set()
    )

    if missing:
        raise ValueError(
            f"CSV缺失必要字段: {sorted(missing)}"
        )

    return rows


# =========================================================
# 构建 traces
# =========================================================

def build_traces(
    rows: List[dict]
) -> Dict[str, List[str]]:

    grouped = defaultdict(list)

    for row in rows:

        cid = str(row["case:concept:name"])

        grouped[cid].append(

            (
                _parse_ts(
                    row["time:timestamp"]
                ),

                row["concept:name"]
            )
        )

    traces = {}

    for cid, events in grouped.items():

        events.sort(key=lambda x: x[0])

        traces[cid] = [

            e for _, e in events

        ]

    return traces


# =========================================================
# 查找路径
# =========================================================

def find_path_index(
    trace: List[str],
    path: Sequence[str]
) -> int:

    L = len(path)

    if L == 0:
        return -1

    for i in range(len(trace) - L + 1):

        if trace[i:i + L] == list(path):
            return i

    return -1


# =========================================================
# Loop标准化
# =========================================================

def canonical_loop(pattern):

    rotations = []

    n = len(pattern)

    for i in range(n):

        rotated = pattern[i:] + pattern[:i]

        rotations.append(tuple(rotated))

    return min(rotations)


# =========================================================
# Loop Mining
# =========================================================

def evaluate_loops(
    traces,
    config
):

    loop_cfg = config.get("loop", {})

    MAX_LOOP_LENGTH = loop_cfg.get(
        "max_loop_length",
        5
    )

    MIN_SUPPORT = loop_cfg.get(
        "min_support",
        2
    )

    MAX_EXAMPLES = loop_cfg.get(
        "max_examples",
        3
    )

    loop_patterns = defaultdict(int)

    loop_examples = defaultdict(list)

    for case_id, trace in traces.items():

        n = len(trace)

        trace_loops = set()

        trace_loop_examples = {}

        for loop_len in range(
            2,
            MAX_LOOP_LENGTH + 1
        ):

            for i in range(
                n - 2 * loop_len + 1
            ):

                pattern = trace[
                    i:i + loop_len
                ]

                next_pattern = trace[
                    i + loop_len:
                    i + 2 * loop_len
                ]

                # =====================================
                # loop detected
                # =====================================

                if pattern == next_pattern:

                    canonical = canonical_loop(
                        pattern
                    )

                    trace_loops.add(canonical)

                    if (
                        canonical
                        not in trace_loop_examples
                    ):

                        trace_loop_examples[
                            canonical
                        ] = trace

        # =========================================
        # 一个trace只贡献一次support
        # =========================================

        for loop in trace_loops:

            loop_patterns[loop] += 1

            if (
                len(loop_examples[loop])
                <
                MAX_EXAMPLES
            ):

                loop_examples[loop].append(

                    trace_loop_examples[loop]
                )

    results = []

    for pattern, freq in sorted(

        loop_patterns.items(),

        key=lambda x: x[1],

        reverse=True
    ):

        if freq < MIN_SUPPORT:
            continue

        results.append({

            "type": "Loop",

            "loop_path":
                list(pattern),

            "loop_str":
                " -> ".join(pattern),

            "support":
                freq,

            "examples":
                loop_examples[pattern]
        })

    return results


# =========================================================
# XOR choice
# =========================================================

@dataclass
class XORChoice:

    branch_index: int

    branch_path: Sequence[str]

    choice_pos: int


# =========================================================
# 提取 XOR 决策
# =========================================================

def extract_xor_choice(

    trace: List[str],

    branches: List[Sequence[str]]

) -> Optional[XORChoice]:

    candidate = []

    for idx, branch in enumerate(branches):

        pos = find_path_index(
            trace,
            branch
        )

        if pos != -1:

            candidate.append(

                (
                    pos,
                    idx,
                    branch
                )
            )

    if not candidate:
        return None

    candidate.sort(key=lambda x: x[0])

    pos, idx, branch = candidate[0]

    return XORChoice(

        branch_index=idx,

        branch_path=branch,

        choice_pos=pos
    )


# =========================================================
# 最近 trigger
# =========================================================

def latest_trigger_before_choice(
    trace,
    trigger_event,
    choice_pos
):
    latest = -1

    for i in range(choice_pos):

        if trace[i] == trigger_event:
            latest = i

    return latest


# =========================================================
# XOR statistics
# =========================================================

def compute_xor_statistics(
    traces,
    trigger_event,
    xor_block
):

    branches = xor_block["branches"]

    trigger_counter = defaultdict(int)

    no_trigger_counter = defaultdict(int)

    trigger_support = 0

    no_trigger_support = 0

    for trace in traces.values():

        choice = extract_xor_choice(
            trace,
            branches
        )

        if choice is None:
            continue

        latest_trigger = latest_trigger_before_choice(
    trace,
    trigger_event,
    choice.choice_pos
)

        # ============================================
        # trigger before choice
        # ============================================

        if latest_trigger != -1:

            trigger_support += 1

            trigger_counter[
                choice.branch_index
            ] += 1

        # ============================================
        # no trigger
        # ============================================

        else:

            no_trigger_support += 1

            no_trigger_counter[
                choice.branch_index
            ] += 1

    trigger_prob = {}

    no_trigger_prob = {}

    for idx in range(len(branches)):

        trigger_prob[idx] = (

            trigger_counter[idx]
            /
            trigger_support

            if trigger_support else 0
        )

        no_trigger_prob[idx] = (

            no_trigger_counter[idx]
            /
            no_trigger_support

            if no_trigger_support else 0
        )

    return {

        "trigger_support":
            trigger_support,

        "no_trigger_support":
            no_trigger_support,

        "trigger_prob":
            trigger_prob,

        "no_trigger_prob":
            no_trigger_prob
    }


# =========================================================
# NR statistics
# =========================================================

def compute_nr_statistics(
        traces,
        trigger_event,
        target_event
):

    trigger_cases = 0
    non_trigger_cases = 0

    target_given_trigger = 0
    target_given_non_trigger = 0

    for trace in traces.values():

        if trigger_event in trace:

            trigger_cases += 1

            if target_event in trace:
                target_given_trigger += 1

        else:

            non_trigger_cases += 1

            if target_event in trace:
                target_given_non_trigger += 1

    p_target_trigger = (
        target_given_trigger /
        trigger_cases
        if trigger_cases else 0
    )

    p_not_target_trigger = (
        1 - p_target_trigger
    )

    p_target_non_trigger = (
        target_given_non_trigger /
        non_trigger_cases
        if non_trigger_cases else 0
    )

    internal_suppression = (
        p_not_target_trigger
        -
        p_target_trigger
    )

    relative_suppression = (
        p_target_non_trigger
        -
        p_target_trigger
    )

    return {

        "trigger_cases":
            trigger_cases,

        "non_trigger_cases":
            non_trigger_cases,

        "P(target|trigger)":
            p_target_trigger,

        "P(¬target|trigger)":
            p_not_target_trigger,

        "P(target|¬trigger)":
            p_target_non_trigger,

        "internal_suppression":
            internal_suppression,

        "relative_suppression":
            relative_suppression
    }

# =========================================================
# evaluate relations
# =========================================================

def evaluate_relations(
    traces,
    config
):

    triggers = config["triggers"]

    xor_blocks = config["xor_blocks"]

    target_events = config["target_events"]

    pr_threshold = config.get(
        "pr_threshold",
        0.2
    )

    nrwap_threshold = config.get(
        "nrwap_threshold",
        0.2
    )

    nr_threshold = config.get(
        "nr_threshold",
        0.2
    )

    min_support = config.get(
        "min_support",
        10
    )

    results = {

        "Loop": [],

        "PR": [],

        "NR": [],

        "NRwAP": []

    }

    # =====================================================
    # Loop
    # =====================================================

    results["Loop"] = evaluate_loops(
        traces,
        config
    )

    # =====================================================
    # PR / NRwAP
    # =====================================================

    for trigger in triggers:

        for xor_block in xor_blocks:

            st = compute_xor_statistics(

                traces,

                trigger,

                xor_block
            )

            if (
                st["trigger_support"]
                <
                min_support
            ):
                continue

            branches = xor_block["branches"]

            # =============================================
            # 每个branch单独分析
            # =============================================

            for idx, branch in enumerate(branches):

                p_trigger = st[
                    "trigger_prob"
                ][idx]

                p_no_trigger = st[
                    "no_trigger_prob"
                ][idx]

                gain = (

                    p_trigger
                    -
                    p_no_trigger
                )

                # =========================================
                # PR
                # =========================================

                if gain > pr_threshold:

                    results["PR"].append({
                        "id":
                            f"PR_{trigger}_{'_'.join(branch)}",

                        "type": "PR",

                        "trigger":
                            trigger,

                        "target_path":
                            branch,

                        "xor_block":
                            xor_block["name"],

                        "metrics": {

                            "trigger_support":
                                st["trigger_support"],

                            "P(path|trigger)":
                                round(
                                    p_trigger,
                                    6
                                ),

                            "P(path|¬trigger)":
                                round(
                                    p_no_trigger,
                                    6
                                ),

                            "gain":
                                round(
                                    gain,
                                    6
                                )
                        }
                    })

                # =========================================
                # NRwAP
                #
                # 条件：
                #
                # 1. 当前branch下降
                # 2. 存在其它branch显著上升
                # =========================================

                if gain < -nrwap_threshold:

                    # =====================================
                    # 检查 alternative branch
                    # =====================================

                    has_alternative = False

                    alternative_branch = None

                    alternative_gain = 0

                    for alt_idx, alt_branch in enumerate(branches):

                        if alt_idx == idx:
                            continue

                        alt_p_trigger = st[
                            "trigger_prob"
                        ][alt_idx]

                        alt_p_no_trigger = st[
                            "no_trigger_prob"
                        ][alt_idx]

                        alt_gain = (

                                alt_p_trigger
                                -
                                alt_p_no_trigger
                        )

                        # =================================
                        # alternative enhancement
                        # =================================

                        if alt_gain > 0:
                            has_alternative = True

                            alternative_branch = alt_branch

                            alternative_gain = alt_gain

                            break

                    # =====================================
                    # 真正的 NRwAP
                    # =====================================

                    if has_alternative:
                        results["NRwAP"].append({
                            "id":
                                f"NRWAP_{trigger}_{'_'.join(branch)}",

                            "type": "NRwAP",

                            "trigger":
                                trigger,

                            "suppressed_path":
                                branch,

                            "alternative_path":
                                alternative_branch,

                            "xor_block":
                                xor_block["name"],

                            "metrics": {

                                "trigger_support":
                                    st["trigger_support"],

                                "P(path|trigger)":
                                    round(
                                        p_trigger,
                                        6
                                    ),

                                "P(path|¬trigger)":
                                    round(
                                        p_no_trigger,
                                        6
                                    ),

                                "drop":
                                    round(
                                        abs(gain),
                                        6
                                    ),

                                "alternative_gain":
                                    round(
                                        alternative_gain,
                                        6
                                    )
                            }
                        })

    # =====================================================
    # NR
    # =====================================================

    for trigger in triggers:

        for target_event in target_events:

            st = compute_nr_statistics(
                traces,
                trigger,
                target_event
            )

            if st["trigger_cases"] < min_support:
                continue

            # 非Trigger样本太少
            if st["non_trigger_cases"] < min_support:

                score = (
                    st["internal_suppression"]
                )

            else:

                score = (

                        0.5 *
                        st["internal_suppression"]

                        +

                        0.5 *
                        st["relative_suppression"]

                )

            if score > nr_threshold:
                results["NR"].append({
                    "id":
                        f"NR_{trigger}_{target_event}",

                    "type": "NR",

                    "trigger":
                        trigger,

                    "target_event":
                        target_event,

                    "metrics": {

                    "trigger_cases":
                        st["trigger_cases"],

                    "non_trigger_cases":
                        st["non_trigger_cases"],

                    "P(target|trigger)":
                        round(
                            st["P(target|trigger)"],
                            6
                        ),

                    "P(¬target|trigger)":
                        round(
                            st["P(¬target|trigger)"],
                            6
                        ),

                    "P(target|¬trigger)":
                        round(
                            st["P(target|¬trigger)"],
                            6
                        ),

                    "internal_suppression":
                        round(
                            st["internal_suppression"],
                            6
                        ),

                    "relative_suppression":
                        round(
                            st["relative_suppression"],
                            6
                        ),

                    "score":
                        round(
                            score,
                            6
                        )
                }
                })

    return results
# =====================================================
# Generate Sliding Windows
# =====================================================

def generate_case_windows(
        traces,
        window_size=200,
        step=50):
    case_ids = sorted(
        traces.keys(),
        key=lambda x: int(x)
    )

    windows = []

    for start in range(
            0,
            len(case_ids) - window_size + 1,
            step):

        sub_ids = case_ids[
            start:start + window_size
        ]

        windows.append({

            "start_case":
                sub_ids[0],

            "end_case":
                sub_ids[-1],

            "case_ids":
                sub_ids
        })

    return windows
# =====================================================
# Build Window Traces
# =====================================================

def build_sub_traces(
        traces,
        case_ids):

    return {

        cid: traces[cid]

        for cid in case_ids
    }
# =====================================================
# Temporal Relation Discovery
# =====================================================

def discover_temporal_relations(
        traces,
        config,
        window_size=200,
        step=50):

    windows = generate_case_windows(
        traces,
        window_size,
        step
    )

    temporal_results = []

    print(
        f"\nSliding Windows: {len(windows)}"
    )

    for idx, window in enumerate(windows):

        print(
            f"Window {idx + 1}/{len(windows)}"
        )

        sub_traces = build_sub_traces(
            traces,
            window["case_ids"]
        )

        result = evaluate_relations(
            sub_traces,
            config
        )

        relations = []

        # ==========================
        # PR
        # ==========================

        for rel in result["PR"]:

            relations.append({

                "id":
                    rel["id"],

                "type":
                    "PR",

                "strength":
                    rel["metrics"]["gain"]
            })

        # ==========================
        # NR
        # ==========================

        for rel in result["NR"]:

            relations.append({

                "id":
                    rel["id"],

                "type":
                    "NR",

                "strength":
                    rel["metrics"]["score"]
            })

        # ==========================
        # NRwAP
        # ==========================

        for rel in result["NRwAP"]:

            relations.append({

                "id":
                    rel["id"],

                "type":
                    "NRwAP",

                "strength":
                    rel["metrics"]["drop"]
            })

        temporal_results.append({

            "start_case":
                int(window["start_case"]),

            "end_case":
                int(window["end_case"]),

            "relations":
                relations
        })

    return temporal_results

# =====================================================
# Collect Relation Windows
# =====================================================

def collect_relation_windows(
        temporal_results):

    relation_windows = {}

    for window in temporal_results:

        start_case = window["start_case"]

        end_case = window["end_case"]

        for rel in window["relations"]:

            rid = rel["id"]

            relation_windows\
                .setdefault(
                    rid,
                    []
                ).append({

                    "start_case":
                        start_case,

                    "end_case":
                        end_case,

                    "strength":
                        rel["strength"],

                    "type":
                        rel["type"]
                })

    return relation_windows

# =====================================================
# Merge Overlap Windows
# =====================================================

def merge_windows(
        windows):

    if not windows:

        return []

    windows = sorted(
        windows,
        key=lambda x: x[0]
    )

    merged = [

        list(windows[0])
    ]

    for current in windows[1:]:

        last = merged[-1]

        if current[0] <= last[1]:

            last[1] = max(

                last[1],

                current[1]
            )

        else:

            merged.append(

                list(current)
            )

    return merged

# =====================================================
# Extract Valid Ranges
# =====================================================

def summarize_temporal_relations(
        temporal_results,
        config,
        min_duration=300):

    relation_windows = \
        collect_relation_windows(
            temporal_results
        )

    final_results = []

    for rid, windows in \
            relation_windows.items():

        if not windows:
            continue

        rel_type = windows[0]["type"]

        # =========================
        # threshold
        # =========================

        if rel_type == "PR":

            threshold = \
                config["pr_threshold"]

        elif rel_type == "NR":

            threshold = \
                config["nr_threshold"]

        else:

            threshold = \
                config["nrwap_threshold"]

        # =========================
        # 保留强度满足条件窗口
        # =========================

        valid_windows = []

        for w in windows:

            if w["strength"] >= threshold:

                valid_windows.append(

                    (
                        w["start_case"],
                        w["end_case"]
                    )
                )

        if not valid_windows:

            continue

        merged = merge_windows(
            valid_windows
        )

        # =========================
        # 输出最终区间
        # =========================

        for start_case, end_case \
                in merged:

            duration = (

                int(end_case)
                -
                int(start_case)
                +
                1
            )

            if duration < min_duration:

                continue

            final_results.append({

                "relation":
                    rid,

                "type":
                    rel_type,

                "start_case":
                    start_case,

                "end_case":
                    end_case,

                "duration":
                    duration
            })

    return final_results
# =========================================================
# main
# =========================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "Unified Context-aware Relation Mining"
    )

    parser.add_argument(

        "--log",

        type=Path,

        default=Path(
            "data/修改后_PNML_最终版_注入三种关系.csv"
        )
    )

    parser.add_argument(

        "--config",

        type=Path,

        default=Path(
            "先验知识0607.json"
        )
    )

    parser.add_argument(

        "--out",

        type=Path,

        default=Path(
            "outputs/json结果/修改后_PNML_最终版_注入三种关系_带时间_nr修改.json"
        )
    )

    args = parser.parse_args()

    # =====================================================
    # load
    # =====================================================

    rows = load_event_log(
        args.log
    )

    traces = build_traces(
        rows
    )

    config = json.loads(

        args.config.read_text(
            encoding="utf-8"
        )
    )

    # =====================================================
    # evaluate
    # =====================================================

    result = evaluate_relations(

        traces,

        config
    )
    # =====================================================
    # temporal evaluate
    # =====================================================

    temporal_results = discover_temporal_relations(

        traces,

        config,

        window_size=200,

        step=50
    )

    valid_ranges = summarize_temporal_relations(

        temporal_results,
        config,
        min_duration=300
    )

    result["TemporalRanges"] = valid_ranges
    # =====================================================
    # save
    # =====================================================

    args.out.write_text(

        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )

    print(

        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


# =========================================================
# entry
# =========================================================

if __name__ == "__main__":
    main()