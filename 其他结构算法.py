#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结构发现实验

支持算法：
1. Inductive Miner (IM)
2. Alpha Miner
3. Alpha+ Miner
4. Heuristics Miner (HM)

输出：
- PNML
- PNG
"""

from pathlib import Path

import pandas as pd
import pm4py

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

from pm4py.objects.conversion.process_tree import converter as tree_converter

from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer


# =====================================================
# 读取日志
# =====================================================

def load_event_log(csv_path):

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except:
        df = pd.read_csv(csv_path, encoding="latin1")

    df.columns = (
        df.columns
        .str.strip()
        .str.replace('"', '', regex=False)
    )

    case_col, activity_col, timestamp_col = detect_columns(df)

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    df = pm4py.format_dataframe(
        df,
        case_id=case_col,
        activity_key=activity_col,
        timestamp_key=timestamp_col
    )

    return df

def detect_columns(df):

    cols = [c.strip() for c in df.columns]

    case_col = None
    activity_col = None
    timestamp_col = None

    for col in cols:

        lc = col.lower()

        # Case
        if "case:concept:name" == lc:
            case_col = col

        elif "case concept:name" == lc:
            case_col = col

        # Activity
        elif lc == "concept:name":
            activity_col = col

        elif lc == "event concept:name":
            activity_col = col

        # Timestamp
        elif lc == "time:timestamp":
            timestamp_col = col

        elif lc == "event time:timestamp":
            timestamp_col = col

    if case_col is None:
        raise ValueError("未识别Case列")

    if activity_col is None:
        raise ValueError("未识别Activity列")

    if timestamp_col is None:
        raise ValueError("未识别Timestamp列")

    print("\n自动识别列名")
    print("Case列:", case_col)
    print("Activity列:", activity_col)
    print("Timestamp列:", timestamp_col)

    return case_col, activity_col, timestamp_col
# =====================================================
# 保存Petri Net
# =====================================================

def save_petri(net, im, fm, output_dir, model_name):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pnml_path = output_dir / f"{model_name}.pnml"

    pnml_exporter.apply(
        net,
        im,
        pnml_path.as_posix(),
        final_marking=fm
    )

    print(f"PNML保存成功: {pnml_path}")

    try:

        gviz = pn_visualizer.apply(
            net,
            im,
            fm
        )

        png_path = output_dir / f"{model_name}.png"

        pn_visualizer.save(
            gviz,
            png_path.as_posix()
        )

        print(f"PNG保存成功 : {png_path}")

    except Exception as e:

        print(
            f"{model_name} PNG生成失败: {e}"
        )


# =====================================================
# IM
# =====================================================

def run_im(df, output_dir):

    print("\n========== IM ==========")

    tree = inductive_miner.apply(df)

    net, im, fm = tree_converter.apply(tree)

    save_petri(
        net,
        im,
        fm,
        output_dir,
        "IM"
    )


# =====================================================
# Alpha
# =====================================================

def run_alpha(df, output_dir):

    print("\n========== Alpha ==========")

    net, im, fm = alpha_miner.apply(
        df,
        variant=alpha_miner.Variants.ALPHA_VERSION_CLASSIC
    )

    save_petri(
        net,
        im,
        fm,
        output_dir,
        "Alpha"
    )


# =====================================================
# Alpha+
# =====================================================

def run_alpha_plus(df, output_dir):

    print("\n========== Alpha+ ==========")

    try:

        net, im, fm = alpha_miner.apply(
            df,
            variant=alpha_miner.Variants.ALPHA_VERSION_PLUS
        )

        save_petri(
            net,
            im,
            fm,
            output_dir,
            "AlphaPlus"
        )

    except Exception as e:

        print(f"Alpha+失败: {e}")


# =====================================================
# HM
# =====================================================

def run_hm(df, output_dir):

    print("\n========== HM ==========")

    try:

        net, im, fm = heuristics_miner.apply(df)

        save_petri(
            net,
            im,
            fm,
            output_dir,
            "HM"
        )

    except Exception as e:

        print(f"HM失败: {e}")

# =====================================================
# 主程序
# =====================================================

def main():

    csv_file = r"data/Road_Traffic_Fine_Management_Process_sample1000.csv"

    output_dir = r"outputs/PetriNets/Road_Traffic_Fine_Management_Process_sample1000"

    df = load_event_log(
        csv_file
    )

    run_im(
        df,
        output_dir
    )

    run_alpha(
        df,
        output_dir
    )

    run_alpha_plus(
        df,
        output_dir
    )

    run_hm(
        df,
        output_dir
    )

    print("\n===================================")
    print("全部模型挖掘完成")
    print(f"输出目录: {output_dir}")
    print("===================================")


if __name__ == "__main__":
    main()