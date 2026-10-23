"""Render the completed four-budget comparison from its verified results."""

import json

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from experiment import FOLDER, digest, save

COLORS = ["#68727e", "#2878b5", "#e47c24", "#9461ad", "#179d8e"]
SHORT_LABELS = ["Original", "139", "499", "NN 1,057", "1,948"]


def interval(item, delta=False):
    point = item["estimate"] * 100
    low, high = np.array(item["paired_bootstrap_95_ci"]) * 100
    if delta:
        return f"{point:+.1f} [{low:+.1f}, {high:+.1f}]"
    return f"{point:.1f}% [{low:.1f}, {high:.1f}]"


def save_figure(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(FOLDER / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dfe3e8", linewidth=0.7)
    ax.set_axisbelow(True)


def curves(result):
    budgets = result["time_budgets_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.8))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.77, wspace=0.22)
    for index, (model, color) in enumerate(zip(result["models"], COLORS)):
        model_id = model["id"]
        for family, ax in zip(("league", "anchor"), axes):
            reports = [
                result["league"][str(t)]["models"][model_id]
                if family == "league"
                else result["anchor"][model_id]["budgets"][str(t)]
                for t in budgets
            ]
            points = np.array([r["score"] for r in reports]) * 100
            bounds = (
                np.array(
                    [r["score_interval"]["paired_bootstrap_95_ci"] for r in reports]
                )
                * 100
            )
            positions = np.array(budgets) * 1.045 ** (index - 2)
            ax.errorbar(
                positions,
                points,
                yerr=np.stack((points - bounds[:, 0], bounds[:, 1] - points)),
                label=model["label"],
                color=color,
                marker="o",
                markersize=5,
                linewidth=2.6 if model_id == "expanded499" else 1.7,
                elinewidth=0.9,
                capsize=3,
            )
    for ax in axes:
        ax.set_xscale("log", base=4)
        ax.set_xticks(budgets, [f"{t:,}" for t in budgets])
        ax.set_xlim(19, 2100)
        ax.set_xlabel("Candidate search budget (ms / move)")
        ax.set_ylabel("Score (%)")
        ax.axhline(50, color="#9298a0", linewidth=1, linestyle="--")
        style_axis(ax)
    axes[0].set_title(
        "Equal-budget round robin\n128 games per model at each budget",
        loc="left",
        fontsize=12,
    )
    axes[1].set_title(
        "Against Original fixed at 25 ms\n32 games per model at each budget",
        loc="left",
        fontsize=12,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "Five frozen engines across a 64× search-time range",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.supxlabel(
        "Win = 1, draw = ½. Bars: pointwise 95% whole-deal bootstrap intervals; points offset horizontally for clarity.\n"
        "Exploratory continuation. At 1,600 ms, existing flip-depth and tablebase-prefetch thresholds also change.",
        fontsize=10,
        y=0.025,
    )
    save_figure(fig, "comparison")


def heatmaps(result):
    ids = [m["id"] for m in result["models"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.8))
    fig.subplots_adjust(
        left=0.09, right=0.92, bottom=0.17, top=0.92, hspace=0.42, wspace=0.27
    )
    palette = plt.get_cmap("RdBu").copy()
    palette.set_bad("#f0f2f5")
    for budget, ax in zip(result["time_budgets_ms"], axes.flat):
        matrix = np.full((len(ids), len(ids)), np.nan)
        annotations = {}
        for cell in result["cells"].values():
            job = cell["config"]
            if job["family"] != "rr" or job["time_ms"] != budget:
                continue
            ai, bi = ids.index(job["a_id"]), ids.index(job["b_id"])
            for row, col, side in ((ai, bi, "a"), (bi, ai, "b")):
                item = cell[side]
                matrix[row, col] = item["score"] * 100
                annotations[
                    row, col
                ] = f"{item['score'] * 100:.1f}%\n{item['wins']}/{item['draws']}/{item['losses']}"
        picture = ax.imshow(
            matrix, cmap=palette, norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
        )
        for row in range(len(ids)):
            for col in range(len(ids)):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    annotations.get((row, col), "—"),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white"
                    if np.isfinite(value) and abs(value - 50) > 33
                    else "#26313d",
                )
        ax.set_xticks(range(len(ids)), SHORT_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(ids)), SHORT_LABELS)
        ax.set_xlabel("Opponent")
        ax.set_ylabel("Row engine")
        ax.set_title(f"{budget:,} ms / move", fontweight="bold", pad=9)
        ax.spines[:].set_visible(False)
    bar_ax = fig.add_axes((0.28, 0.075, 0.44, 0.016))
    fig.colorbar(picture, cax=bar_ax, orientation="horizontal").set_label(
        "Row engine score (%)"
    )
    fig.suptitle(
        "Head-to-head results at equal time", fontsize=17, fontweight="bold", y=0.985
    )
    fig.supxlabel(
        "Each cell: score; wins / draws / losses. 32 games from 16 paired deals; exploratory small-sample comparisons.",
        fontsize=10,
        y=0.015,
    )
    save_figure(fig, "head-to-head")


def gain_plot(result):
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.6), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.21, top=0.76, wspace=0.15)
    for ax, contrast, title in zip(
        axes,
        ("400->1600", "25->1600"),
        ("400 → 1,600 ms: 4× more time", "25 → 1,600 ms: 64× more time"),
    ):
        for index, (model, color) in enumerate(zip(result["models"], COLORS)):
            item = result["anchor"][model["id"]]["gains"][contrast]
            point = item["estimate"] * 100
            low, high = np.array(item["paired_bootstrap_95_ci"]) * 100
            ax.errorbar(
                point,
                index,
                xerr=[[point - low], [high - point]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
                linewidth=1.6,
            )
        ax.axvline(0, color="#9298a0", linewidth=1, linestyle="--")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("Score change (percentage points)")
        ax.set_yticks(range(len(SHORT_LABELS)), SHORT_LABELS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dfe3e8", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    limits = [ax.get_xlim() for ax in axes]
    for ax in axes:
        ax.set_xlim(min(x[0] for x in limits), max(x[1] for x in limits))
    fig.suptitle(
        "Time gains against Original fixed at 25 ms",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    fig.supxlabel(
        "Matched deals across time; pointwise 95% whole-deal bootstrap intervals.\nDifferent models use different deal blocks; versions also differ in training and features.",
        fontsize=10,
        y=0.035,
    )
    save_figure(fig, "time-gains")


def diagnostics(result):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.73, wspace=0.30)
    budgets = result["time_budgets_ms"]
    for model, color in zip(result["models"], COLORS):
        reports = [result["league"][str(t)]["models"][model["id"]] for t in budgets]
        for ax, key in zip(
            axes, ("ms_per_move", "nodes_per_move", "mean_completed_depth")
        ):
            ax.plot(
                budgets,
                [r[key] for r in reports],
                color=color,
                marker="o",
                label=model["label"],
            )
    for ax, label in zip(
        axes, ("Observed ms / move", "Nodes / move", "Mean completed depth")
    ):
        ax.set_xscale("log", base=4)
        ax.set_xticks(budgets, [f"{t:,}" for t in budgets])
        ax.set_xlabel("Budget (ms / move)")
        ax.set_ylabel(label)
        style_axis(ax)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "Observed search work in the round robin",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.supxlabel(
        "Means include cheap/forced moves. Positions differ across engines; node rate alone is not strength.\n1,600 ms enables existing deeper flip search and five-piece successor-table prebuilding.",
        fontsize=10,
        y=0.025,
    )
    save_figure(fig, "search-work")


def write_report(result):
    models = result["models"]
    names = {m["id"]: m["label"] for m in models}
    budgets = result["time_budgets_ms"]
    latest = result["league"]["1600"]["models"]
    high = max(m["score"] for m in latest.values())
    leaders = "、".join(names[m] for m, v in latest.items() if v["score"] == high)
    completion = json.loads((FOLDER / "completion.json").read_text())
    small_cell = result["cells"]["rr-structured139-vs-expanded499-t1600"]
    small_match = small_cell["b"]
    small_interval = {
        "estimate": small_match["score"],
        "paired_bootstrap_95_ci": [
            1 - bound
            for bound in reversed(small_cell["score"]["paired_bootstrap_95_ci"])
        ],
    }
    change = result["direct_changes"]["structured139-vs-expanded499"]["400->1600"]
    small_gain = {
        "estimate": -change["estimate"],
        "paired_bootstrap_95_ci": [
            -bound for bound in reversed(change["paired_bootstrap_95_ci"])
        ],
    }
    original_match = result["cells"]["rr-expanded499-vs-original-t1600"]
    anchor499 = result["anchor"]["expanded499"]
    lines = [
        "# 每步 1,600 ms 的延伸比較",
        "",
        f"新增 **{result['new_games']} 局**，與原來 1,312 局合併，共 **{result['games']:,} 局**。五個凍結版本沿用相同牌局與先後手配對，測試時間形成 **25 → 100 → 400 → 1,600 ms**、每級四倍的序列。所有預定對局均完成；以原始資料重新計算的舊有點估計、區間與差值，均與原報告一致。",
        "",
        f"**1,600 ms 循環賽的點估計最高為 {leaders}（{high*100:.1f}%），本輪繼續保留 499 作為研究基準。** 尚未確認一個全面最強的版本。每組直接對戰只有 16 副獨立牌局，交換先後手後為 32 局；這是探索性的小型延伸比較。",
        "",
        "本文的得分率為 `(勝 + 和 / 2) / 局數`；不是純勝率。方括號為整副牌局配對 bootstrap 的點態 95% 區間；所有差值以百分點表示，未做多重比較校正。",
        "",
        f"**最值得進一步確認的是 499 對 139 的長時間表現。** 400 ms 時雙方得分各 50%；1,600 ms 時，499 取得 {small_match['wins']} 勝／{small_match['draws']} 和／{small_match['losses']} 負，得分率 {interval(small_interval)}，配對時間增益為 {interval(small_gain, True)} 個百分點。這項區間尚未做多重比較校正，而且延伸實驗沿用先前的牌局，需要新牌確認。499 對同樣 1,600 ms 的原版得分率為 {interval(original_match['score'])}，仍不能確認其長時間優勢。",
        "",
        f"**499 的固定對手曲線在這一段沒有增加，但不能據此判定已到上限。** 對手維持原版 25 ms 時，499 在 400 與 1,600 ms 的得分率都是 {100 * anchor499['budgets']['400']['score']:.1f}%；400 → 1,600 ms 的配對增益為 {interval(anchor499['gains']['400->1600'], True)} 個百分點，仍容許可觀的正負差異。所有模型配對在 400 → 1,600 ms、25 → 1,600 ms 兩段的收益差區間都包含零，沒有確認較大容量獲得較大搜尋收益。",
        "",
        "![四個時間點的得分率](comparison.png)",
        "",
        "## 同時間循環賽",
        "",
        "每個版本在每個時間點對其餘四版各下 32 局，共 128 局；四個對手等權。",
        "",
        "| 版本 | 25 ms | 100 ms | 400 ms | 1,600 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        entries = [
            interval(result["league"][str(t)]["models"][model["id"]]["score_interval"])
            for t in budgets
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "雙方一起增加時間時，整個循環賽的平均得分必然維持 50%。下面量的是相對表現的變化；分數下降不能直接解讀為多想之後變弱。",
        "",
        "| 版本 | 400 → 1,600 ms | 25 → 1,600 ms |",
        "| --- | ---: | ---: |",
    ]
    for model in models:
        changes = result["league_changes"][model["id"]]
        lines.append(
            f"| {model['label']} | {interval(changes['400->1600'], True)} | {interval(changes['25->1600'], True)} |"
        )
    lines += [
        "",
        "## 固定對手：原版維持 25 ms",
        "",
        "只增加候選版本的時間，用來觀察搜尋收益。各模型在不同時間重複相同的 16 副牌；模型之間的牌局區塊不同。",
        "",
        "| 版本 | 25 ms | 100 ms | 400 ms | 1,600 ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        entries = [
            interval(result["anchor"][model["id"]]["budgets"][str(t)]["score_interval"])
            for t in budgets
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "| 版本 | 400 → 1,600 ms 收益 | 100 → 1,600 ms 收益 | 25 → 1,600 ms 收益 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in models:
        gains = result["anchor"][model["id"]]["gains"]
        entries = [
            interval(gains[k], True) for k in ("400->1600", "100->1600", "25->1600")
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "![固定對手的時間收益](time-gains.png)",
        "",
        "### 不同版本的搜尋收益差",
        "",
        "下表直接比較「A 的時間收益 − B 的時間收益」，保留各版本跨時間的配對關係。正值表示 A 的收益較大，負值表示 B 較大；區間包含零時，這次樣本仍無法分清。不能用『A 顯著、B 不顯著』代替這個差值檢驗。",
        "",
        "| A − B | 400 → 1,600 ms | 100 → 1,600 ms | 25 → 1,600 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pair, contrasts in result["anchor_gain_differences"].items():
        a, b = pair.split("-minus-")
        entries = [
            interval(contrasts[k], True) for k in ("400->1600", "100->1600", "25->1600")
        ]
        lines.append(f"| {names[a]} − {names[b]} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "這些比較無法單獨辨認參數數量的因果效果：版本同時有特徵、訓練資料、目標與架構差異。固定 25 ms 的對手也可能逐漸接近得分飽和；曲線變平不等於評估器達到能力上限。",
        "",
        "## 1,600 ms 直接互打",
        "",
        "| A | B | A 勝／和／負 | A 得分率與區間 |",
        "| --- | --- | ---: | ---: |",
    ]
    matching = sorted(
        (
            c
            for c in result["cells"].values()
            if c["config"]["family"] == "rr" and c["config"]["time_ms"] == 1600
        ),
        key=lambda c: c["config"]["block"],
    )
    for cell in matching:
        job, side = cell["config"], cell["a"]
        lines.append(
            f"| {names[job['a_id']]} | {names[job['b_id']]} | {side['wins']}／{side['draws']}／{side['losses']} | {interval(cell['score'])} |"
        )
    lines += [
        "",
        "![所有時間點的直接對戰](head-to-head.png)",
        "",
        "## 搜尋耗時與計算量",
        "",
        "下表為同時間循環賽的實際平均；包含便宜或強制走法，因此平均值可能低於設定預算。不同版本走到的局面不同，節點速率不能直接當成棋力。",
        "",
        "| 版本 | 預算 ms | 實際 ms／步 | 節點／步 | 平均完成深度 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for t in budgets:
            item = result["league"][str(t)]["models"][model["id"]]
            lines.append(
                f"| {model['label']} | {t:,} | {item['ms_per_move']:.1f} | {item['nodes_per_move']:,.0f} | {item['mean_completed_depth']:.2f} |"
            )
    lines += [
        "",
        "![實際搜尋工作量](search-work.png)",
        "",
        "## 方法與重現",
        "",
        "- 五版權重、原生 arena 執行檔、引擎程式、搜尋設定、種子及原有研究檔案均以 SHA-256 凍結。沒有重新訓練或挑選 checkpoint。",
        "- 同時間循環賽新增 10 組 × 32 局；固定原版 25 ms 新增 5 組 × 32 局，共 480 局。每組拆為八批四局，保留每副牌的兩局先後手配對。",
        "- 120 個小批次以事先固定種子打散；同時六批、每批兩個工作執行緒，總共十二個搜尋工作執行緒。每局仍使用原引擎既有的全新玩家與搜尋狀態。",
        "- bootstrap 使用 20,000 次整副牌局重抽樣；11 個獨立牌局區塊各自重抽，同一區塊跨時間及重用對局共用抽樣索引。原有所有點估計與區間必須完全一致。",
        "- 本次是在看過原有 25／100／400 ms 結果後加入的延伸實驗；沿用牌局有助比較時間差，並非新的獨立確認集。區間涵蓋牌局變異，不涵蓋重新訓練的種子變異。",
        "- 設定保持 tablebase 開啟、TT 18 bits、最大深度 64、id_stop 0.65、無節點上限；arena 使用直接搜尋模式。",
        "- **時間增加也跨過原有搜尋門檻。** 25 ms 的自動翻子最低深度為 5，100／400 ms 為 6，1,600 ms 為 7；1,600 ms 另啟用已完全翻開的五子局面之四子後繼殘局表預建。這反映實際預算設定的表現，不能歸因於純計算量增加。",
        "- 所有預定對局完成後才產生完整分析，沒有依勝負提前停止。並行執行仍有作業系統排程噪音。",
        "",
        f"本次延伸執行耗時 **{completion['wall_seconds_this_invocation'] / 60:.1f} 分鐘**。",
        "",
        "| 版本 | 可訓練參數 | 儲存係數 | 權重 |",
        "| --- | ---: | ---: | --- |",
    ]
    for model in models:
        path = model.get("path")
        link = f"[{path}](../../{path})" if path else "原版手寫評估"
        stored = "—" if model["id"] == "original" else model["stored_parameters"]
        lines.append(
            f"| {model['label']} | {model['trainable_parameters']} | {stored} | {link} |"
        )
    lines += [
        "",
        "139／499／1,948 為學習式結構化評估器；NN 1,057 是 64 → 16 → 1 ReLU 殘差網路（輸出倍率 0.5）。不能把所有版本都稱為相同架構的神經網路。",
        "",
        "### 重現指令",
        "",
        "```sh",
        "PYENV_VERSION=3.11.1 python training/tournament1600/run_long.py --verify-only",
        "PYENV_VERSION=3.11.1 python training/tournament1600/analyze_long.py",
        "MPLCONFIGDIR=/tmp/cdc-tournament1600-mpl PYENV_VERSION=3.11.1 python training/tournament1600/report.py",
        "PYENV_VERSION=3.11.1 python training/tournament1600/verify_report.py",
        "```",
        "",
        "原始 CSV、各批執行紀錄與凍結清單保存在本目錄。",
        "",
        "- [實驗計畫](experiment-plan.json)",
        "- [凍結輸入與排程](run.json)",
        "- [完成與原始 CSV 雜湊](completion.json)",
        "- [完整統計結果](results.json)",
        "- [獨立驗證](verification.json)",
        "- [原來三個時間點的報告](../tournament/README.md)",
        "",
    ]
    (FOLDER / "README.md").write_text("\n".join(lines))


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    if (
        result["time_budgets_ms"] != [25, 100, 400, 1600]
        or not result["prior_results_reproduced_exactly"]
    ):
        raise ValueError("expected the completed four-budget extension")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    curves(result)
    heatmaps(result)
    gain_plot(result)
    diagnostics(result)
    write_report(result)
    files = ["report.py", "README.md", "results.json", "completion.json", "run.json"]
    files += [
        f"{name}.{extension}"
        for name in ("comparison", "head-to-head", "time-gains", "search-work")
        for extension in ("png", "svg")
    ]
    save(
        FOLDER / "report-artifacts.json",
        {name: digest(FOLDER / name) for name in files},
    )
    print("Wrote README, four PNG/SVG figures, and artifact hashes")


if __name__ == "__main__":
    main()
