"""Render already-computed tournament results without changing the analysis."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from run import FOLDER, digest, save

COLORS = ["#68727e", "#2878b5", "#e47c24", "#9461ad", "#179d8e"]
SHORT_LABELS = ["Original", "139", "499", "NN 1,057", "1,948"]


def percent_interval(item):
    low, high = item["paired_bootstrap_95_ci"]
    return f"{100 * item['estimate']:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def delta_interval(item):
    low, high = item["paired_bootstrap_95_ci"]
    return f"{100 * item['estimate']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def save_figure(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(FOLDER / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def curves(result):
    budgets = result["time_budgets_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.77, wspace=0.22)
    for index, (model, color) in enumerate(zip(result["models"], COLORS)):
        model_id = model["id"]
        for family, ax in zip(("league", "anchor"), axes):
            reports = [
                result["league"][str(t)]["models"][model_id]
                if family == "league"
                else result["anchor"][model_id]["budgets"][str(t)]
                for t in budgets
            ]
            points = np.array([r["score"] * 100 for r in reports])
            bounds = (
                np.array(
                    [r["score_interval"]["paired_bootstrap_95_ci"] for r in reports]
                )
                * 100
            )
            # Small horizontal offsets separate overlapping intervals; tick labels remain the budgets.
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
                alpha=0.95,
            )
    for ax in axes:
        ax.set_xscale("log", base=4)
        ax.set_xticks(budgets, [str(t) for t in budgets])
        ax.set_xlim(19, 525)
        ax.set_xlabel("Candidate search budget (ms / move)")
        ax.set_ylabel("Score (%)")
        ax.axhline(50, color="#9298a0", linewidth=1, linestyle="--")
        ax.grid(axis="y", color="#dfe3e8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "Five frozen engines across a 16× search-time range",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.supxlabel(
        "Win = 1, draw = ½. Bars: pointwise 95% whole-deal bootstrap intervals; exploratory comparisons.",
        fontsize=10,
    )
    save_figure(fig, "comparison")


def heatmaps(result):
    models = result["models"]
    ids = [m["id"] for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), layout="constrained")
    palette = plt.get_cmap("RdBu").copy()
    palette.set_bad("#f0f2f5")
    for budget, ax in zip(result["time_budgets_ms"], axes):
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
        image = ax.imshow(
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
                    fontsize=9,
                    color="white"
                    if np.isfinite(value) and abs(value - 50) > 33
                    else "#26313d",
                )
        ax.set_xticks(range(len(ids)), SHORT_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(ids)), SHORT_LABELS)
        ax.set_xlabel("Opponent")
        ax.set_title(f"{budget} ms / move", fontweight="bold")
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Row engine")
    colorbar = fig.colorbar(
        image, ax=axes, orientation="horizontal", shrink=0.45, pad=0.04, aspect=40
    )
    colorbar.set_label("Row engine score (%)")
    fig.suptitle("Head-to-head results at equal time", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "Each cell: score; wins / draws / losses. 32 games from 16 paired deals; small cells have wide uncertainty.",
        fontsize=10,
    )
    save_figure(fig, "head-to-head")


def diagnostics(result):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.73, wspace=0.3)
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
        ax.set_xticks(budgets, [str(t) for t in budgets])
        ax.set_xlabel("Budget (ms / move)")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe3e8", linewidth=0.7)
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
        "Per-move means include cheap/forced moves. Positions differ across engines; node rate alone is not strength.",
        fontsize=10,
    )
    save_figure(fig, "search-work")


def write_report(result):
    models = result["models"]
    names = {m["id"]: m["label"] for m in models}
    budgets = result["time_budgets_ms"]
    lines = [
        "# 五個引擎版本的時間預算比較",
        "",
        f"完成 **{result['games']:,} 局、{result['batches']} 組對戰**。五個版本使用凍結權重，在每步 25、100、400 ms 互相對戰，並另與固定 25 ms 的原版比較。所有事先設定的對戰均完成，沒有依結果提早停止或更換模型。",
        "",
        "**三個時間點的循環賽點估計第一名分別為：25 ms 的 499 版、100 ms 的 1,948 版、400 ms 的 139 版。這輪沒有確認一個全面最強的版本，也沒有看到較大模型一致獲得較大的搜尋收益。** 400 ms 下，139 與 499 直接互打為 11 勝／10 和／11 負，得分率 50%；因此保留 499 作為研究基準，不把這次探索排序當成替換依據。",
        "",
        "這是小型探索比較：每個直接對戰條件只有 16 副獨立牌局（交換先後手，共 32 局）。表中得分率為 `(勝 + 和 / 2) / 局數`，不是純勝率；方括號為配對 bootstrap 的 95% 區間。點估計第一名不等於已確認的最強版本。",
        "",
        "![時間與得分率](comparison.png)",
        "",
        "## 同時間循環賽",
        "",
        "每個版本在每個時間點對其餘四個版本各下 32 局，共 128 局。平均得分對四個對手等權。",
        "",
        "| 版本 | 25 ms | 100 ms | 400 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in models:
        entries = [
            percent_interval(
                result["league"][str(t)]["models"][model["id"]]["score_interval"]
            )
            for t in budgets
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += ["", "各時間的點估計排序（同分不代表可分出高下）：", ""]
    for t in budgets:
        ranking = result["league"][str(t)]["ranking"]
        lines.append(
            f"- **{t} ms：** "
            + " → ".join(
                f"{names[m]} {result['league'][str(t)]['models'][m]['score'] * 100:.1f}%"
                for m in ranking
            )
        )
    lines += [
        "",
        "所有版本都增加時間時，循環賽分數的平均必然維持 50%。下表量的是相對排名的變化；不能把某版本得分下降直接解讀為它多想之後變弱。",
        "",
        "| 版本 | 25 → 100 ms | 100 → 400 ms | 25 → 400 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in models:
        entries = [
            delta_interval(result["league_changes"][model["id"]][key])
            for key in ("25->100", "100->400", "25->400")
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "單位：得分率百分點，附配對 95% 區間。",
        "",
        "## 固定對手下的搜尋收益",
        "",
        "對手一律是每步 25 ms 的原版，只有我方增加時間。每個版本的三個時間點沿用同一組牌局；不同版本使用不同的牌局組。原版對原版也實際下了對照局，沒有把其 25 ms 得分硬設成 50%。",
        "",
        "| 我方版本 | 25 ms | 100 ms | 400 ms | 25 → 400 ms 增益（百分點） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        item = result["anchor"][model["id"]]
        entries = [
            percent_interval(item["budgets"][str(t)]["score_interval"]) for t in budgets
        ]
        lines.append(
            f"| {model['label']} | "
            + " | ".join(entries)
            + f" | {delta_interval(item['gains']['25->400'])} |"
        )
    lines += [
        "",
        "固定對手下各時間區間的增益（得分率百分點；同組牌局配對比較）：",
        "",
        "| 我方版本 | 25 → 100 ms | 100 → 400 ms | 25 → 400 ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in models:
        gains = result["anchor"][model["id"]]["gains"]
        entries = [
            delta_interval(gains[key]) for key in ("25->100", "100->400", "25->400")
        ]
        lines.append(f"| {model['label']} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "## 如何解讀差異",
        "",
        "25 ms 的 499 版與 400 ms 的 139 版，循環賽平均得分率分別為 53.9% [47.3, 60.5] 與 53.5% [46.5, 60.5]。其點估計領先尚未建立對其他版本的明確優勢；400 ms 的 139 與 499 平均得分只差約 1.2 個百分點，直接對戰得分也恰好各半，不能據此宣稱兩者等效或分出最強。",
        "",
        "100 ms 的 1,948 版是值得複驗的訊號：循環賽平均得分率比 499 高 10.2 個百分點，逐項 95% 區間為 +0.4～+19.9。但同時間兩版的直接對戰，1,948 得分率為 59.4%，區間 46.9%～71.9%，仍跨過 50%；本輪也未校正多重比較，不能把這個單一訊號當作已確認的優勢。",
        "",
        "對固定 25 ms 原版，五版從 25 增至 400 ms 的得分增益點估計皆為正，介於 +6.25～+17.19 個百分點；139 的整段增益逐項區間高於零，其餘四版仍包含零。499 從 100 增至 400 ms 的增益為 +20.3 個百分點 [4.7, 35.9]。神經網路版從 100 至 400 ms 的點估計下降 6.25 個百分點，但區間為 −23.4～+10.9，不能據此判定更多時間有害。這些都是探索結果。",
        "",
        "## 直接對戰",
        "",
        "![三個時間的對戰矩陣](head-to-head.png)",
        "",
        "以下均從左方版本的角度計算；勝／和／負總數為 32。",
        "",
        "| 時間 | 左方 | 右方 | 勝／和／負 | 左方得分率 [95% 區間] |",
        "| ---: | --- | --- | --- | ---: |",
    ]
    for t in budgets:
        cells = sorted(
            (
                c
                for c in result["cells"].values()
                if c["config"]["family"] == "rr" and c["config"]["time_ms"] == t
            ),
            key=lambda c: c["config"]["block"],
        )
        for cell in cells:
            job, item = cell["config"], cell["a"]
            lines.append(
                f"| {t} ms | {names[job['a_id']]} | {names[job['b_id']]} | {item['wins']}／{item['draws']}／{item['losses']} | {percent_interval(cell['score'])} |"
            )
    lines += [
        "",
        "## 實際搜尋工作量",
        "",
        "![搜尋時間、節點與深度](search-work.png)",
        "",
        "搜尋保留現有 arena 的預設直接搜尋模式，未啟用 MoE 抽樣：殘局庫開啟、18-bit 置換表、最大深度 64、迭代加深停止比例 0.65。翻子搜尋門檻依各自預算自動設定：25 ms 為 5，100／400 ms 為 6。這衡量實際時間設定下的引擎行為，不能完全歸因於同一搜尋策略下的純計算量增加，也不是網頁 MoE 多樣模式的測試。",
        "",
        "時間包含評估器推論和殘局庫建構。便宜的走法、強制走法與迭代加深停止策略會使平均耗時低於每步預算；下表從循環賽的實際記錄彙總。",
        "",
        "| 時間 | 版本 | 勝／和／負 | 純勝率 | 實際 ms／步 | 節點／步 | 平均完成深度 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for t in budgets:
        for model in models:
            item = result["league"][str(t)]["models"][model["id"]]
            lines.append(
                f"| {t} | {model['label']} | {item['wins']}／{item['draws']}／{item['losses']} | {item['win_rate'] * 100:.1f}% | {item['ms_per_move']:.2f} | {item['nodes_per_move']:,.0f} | {item['mean_completed_depth']:.2f} |"
            )
    lines += [
        "",
        "## 模型與不確定性",
        "",
        "原版是既有手寫形式的評估器；139／499／1,948 參數版是結構化學習評估器，1,057 參數版才是 64 → 16 → 1 ReLU 殘差網路（scale 0.5）。499 與 1,948 的容量擴充使用查表項。五個版本的資料、訓練目標與特徵並非完全一致，因此本輪比較的是實際版本，無法單獨證明增加參數數量的因果效果。",
        "",
        "| 版本 | 學習模型的可訓練參數 | 凍結模型 |",
        "| --- | ---: | --- |",
    ]
    for model in models:
        path = model.get("path")
        link = f"[{Path(path).name}](../../{path})" if path else "原版（無外部學習模型）"
        parameters = f"{model['trainable_parameters']:,}" if path else "—"
        lines.append(f"| {model['label']} | {parameters} | {link} |")
    lines += [
        "",
        "1,948 參數版本在本輪開始前，依既有共同驗證損失從 seeds 53／71／89 選出 seed 89；其他版本沿用先前已指定的權重。所有模型 SHA-256、metadata、arena 二進位、引擎原始碼、分析程式與排程在第一局前凍結並於結束後核對。",
        "",
        "每組直接對手使用不同的 16 副牌，三個時間點重用該組牌局。循環賽共 160 副牌，原版固定對手的自我對照另用 16 副牌。學習版固定對手系列在 25 ms 重用循環賽結果，其餘 100／400 ms 另打，總計 1,312 局且沒有重複計數。種子範圍與所有模型的訓練／驗證牌局均不重疊。",
        "",
        "分析以整副牌的兩局為抽樣單位，進行 20,000 次分層 bootstrap；不同牌局組獨立抽樣，同組跨時間與重用的比賽共享抽樣索引。循環賽中雙方得分互補，保留相依性。所有區間為逐項 95% 區間，未做多重比較校正，且不涵蓋重新訓練種子的變異。三個有限時間點不能證明棋力上限。",
        "",
        "## 重現與檢查",
        "",
        "執行環境使用 Python 3.11.1、NumPy "
        + result["numpy_version"]
        + "、Matplotlib "
        + matplotlib.__version__
        + "。原生 CPU 搜尋以 6 組對戰 × 2 個 workers 並行，未使用 GPU。計時受系統排程影響，重跑比賽不保證逐局結果完全相同；保留原始 CSV 可精確重現分析。",
        "",
        "```sh",
        "PYENV_VERSION=3.11.1 python -m unittest discover -s training/tournament -v",
        "PYENV_VERSION=3.11.1 python training/tournament/run.py --verify-only",
        "PYENV_VERSION=3.11.1 python training/tournament/run.py",
        "PYENV_VERSION=3.11.1 python training/tournament/analyze.py",
        "PYENV_VERSION=3.11.1 MPLCONFIGDIR=/tmp/cdc-tournament-mpl python training/tournament/report.py",
        "```",
        "",
        "已完成的對戰只在輸入與 CSV 雜湊通過檢查後重用。未登記的輸出、缺少先後手、牌局重複、錯誤局數或權重改變均會明確失敗。五項測試涵蓋完整排程、固定對手重用、對戰方向、循環賽零和相依性及跨時間配對抽樣。",
        "",
        "原始證據：[事先固定的計畫](experiment-plan.json)、[凍結記錄](run.json)、[完成記錄與 CSV 雜湊](completion.json)、[完整數值結果](results.json)。",
        "",
    ]
    (FOLDER / "README.md").write_text("\n".join(lines))


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}
    )
    curves(result)
    heatmaps(result)
    diagnostics(result)
    write_report(result)
    paths = [FOLDER / "report.py", FOLDER / "results.json", FOLDER / "README.md"]
    paths += [
        FOLDER / f"{name}.{suffix}"
        for name in ("comparison", "head-to-head", "search-work")
        for suffix in ("png", "svg")
    ]
    save(FOLDER / "report-hashes.json", {p.name: digest(p) for p in paths})
    print("Saved report and three figures (PNG + SVG).")


if __name__ == "__main__":
    main()
