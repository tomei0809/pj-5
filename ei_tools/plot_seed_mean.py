"""
plot_seed_mean.py ― EI の個人用ツール（チームのリポジトリには入れない）

tools/demo_mosu.py の結果（demo_out/<system>_<modality>_seed<n>/pred.csv）を、3シードで平均して、
モダリティごとに1枚のグラフにまとめる。各グラフには MoSu の正解と、システム（CNN・GRU・Transformer）の
シード平均を重ねて描く。

使い方（サーバのログインノード、プロジェクトのルートで）:
    python ei_tools/plot_seed_mean.py --demo_dir demo_out --out demo_out/seed_mean
入力:   {demo_dir}/{system}_{modality}_seed{n}/pred.csv   video_id, t, score（動画ごとに0〜1）
        mosu/mosu_gt.h5（正解の gt_score。なければ自動でダウンロード）
出力:   {out}/visual.png, {out}/audio.png, {out}/text.png   行＝動画、赤＝正解、各システムのシード平均
        {out}/scores.csv                                   動画ごと・平均の Spearman ρ
"""
import argparse, os, sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")  # 画面のないサーバーでも画像を保存できるようにする
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.prepare_mosu import fetch_small

LABEL = {"_smoke": "CNN", "gru": "GRU", "transformer": "Transformer"}  # 凡例での名前
COLOR = {"_smoke": "#7a7a7a", "gru": "#2a9d55", "transformer": "#3b6fb6"}
GT_COLOR = "#d0453a"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", default="demo_out")
    p.add_argument("--out", default="demo_out/seed_mean")
    p.add_argument("--systems", nargs="+", default=["_smoke", "gru", "transformer"])
    p.add_argument("--modalities", nargs="+", default=["visual", "audio", "text"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--mosu_dir", default="mosu")
    return p.parse_args()


def normalize(x):
    width = x.max() - x.min()
    return np.zeros_like(x) if width <= 1e-12 else (x - x.min()) / width


def seed_mean(demo_dir, system, modality, seeds):
    """{video_id: シード平均の予測（0〜1）}, 使えたシードの数"""
    per_video, used = {}, 0
    for n in seeds:
        path = f"{demo_dir}/{system}_{modality}_seed{n}/pred.csv"
        if not os.path.exists(path):
            print(f"警告: {path} がないので飛ばす", file=sys.stderr)
            continue
        df = pd.read_csv(path, dtype={"video_id": str}).sort_values(["video_id", "t"])
        for vid, g in df.groupby("video_id"):
            per_video.setdefault(vid, []).append(g["score"].to_numpy(dtype=float))
        used += 1
    mean = {}
    for vid, arrays in per_video.items():
        if len(arrays) != used:  # 一部のシードにしかない動画は使わない
            continue
        T = min(len(a) for a in arrays)
        mean[vid] = normalize(np.mean([a[:T] for a in arrays], axis=0))
    return mean, used


def rho(pred, gt):
    if np.std(pred) < 1e-12 or np.std(gt) < 1e-12:
        return float("nan")  # どの秒も同じ値だと順位がつかない
    return float(spearmanr(pred, gt).statistic)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    with h5py.File(fetch_small("mosu_gt.h5", args.mosu_dir), "r") as gt_file:
        for m in args.modalities:
            preds = {}
            for s in args.systems:
                mean, used = seed_mean(args.demo_dir, s, m, args.seeds)
                if mean:
                    preds[s] = (mean, used)
            if not preds:
                print(f"警告: {m} の結果が1つもないので飛ばす", file=sys.stderr)
                continue
            videos = sorted(set.intersection(*(set(mean) for mean, _ in preds.values())))
            if not videos:
                print(f"警告: {m} でシステムに共通の動画がない（demo_mosu.py の --seed をそろえる）", file=sys.stderr)
                continue

            fig, axes = plt.subplots(len(videos), 1, figsize=(13, 2.9 * len(videos)), squeeze=False)
            for ax, vid in zip(axes[:, 0], videos):
                gt = np.asarray(gt_file[vid]["gt_score"][()], dtype=float).reshape(-1)
                T = min([len(gt)] + [len(mean[vid]) for mean, _ in preds.values()])
                ax.plot(gt[:T], color=GT_COLOR, lw=2.4, label="MoSu gt_score")
                parts = []
                for s, (mean, used) in preds.items():
                    r = rho(mean[vid][:T], gt[:T])
                    rows.append({"modality": m, "system": s, "video_id": vid, "seeds": used, "spearman": r})
                    parts.append(f"{LABEL.get(s, s)} ρ={r:.3f}")
                    ax.plot(mean[vid][:T], color=COLOR.get(s), lw=1.3, alpha=0.9,
                            label=f"{LABEL.get(s, s)} (mean of {used} seeds)")
                ax.set_title(f"{vid}   " + "   ".join(parts), fontsize=10)
                ax.set_xlabel("second"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
            fig.suptitle(f"{m}: ground truth vs. seed-averaged predictions", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            path = f"{args.out}/{m}.png"
            fig.savefig(path, dpi=110); plt.close(fig)
            print(f"保存: {path}（動画 {len(videos)}本）")

    if not rows:
        sys.exit("グラフにできる結果がなかった")
    df = pd.DataFrame(rows)
    avg = df.groupby(["modality", "system"], as_index=False)["spearman"].mean().assign(video_id="(mean)")
    out = pd.concat([df, avg], ignore_index=True)
    out.to_csv(f"{args.out}/scores.csv", index=False)
    print(f"保存: {args.out}/scores.csv\n")
    table = avg.pivot(index="system", columns="modality", values="spearman")
    print("Spearman ρ（動画の平均、シード平均の予測で計算）")
    print(table.reindex(index=[s for s in args.systems if s in table.index],
                        columns=[m for m in args.modalities if m in table.columns]).round(3).to_string())


if __name__ == "__main__":
    main()
