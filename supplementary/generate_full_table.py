import pandas as pd
import os

def format_bold(val, is_max, is_time=False):
    if is_time:
        s = f"{val:.2f}"
    else:
        s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if is_max else s

def generate_table():
    df = pd.read_csv("../data/processed/benchmark_results_detailed.csv")

    category_order = {
        "High Real-World Imbalance": ["ULB", "PaySim"],
        "Moderate Real-World Imbalance": ["IEEE-CIS", "BankSim"],
        "Synthetic Drift Benchmarks": [
            "Agrawal", "SEA",
            "Synth_Agrawal_0p005_var2", "Synth_Agrawal_0p01_var1", "Synth_Agrawal_0p02_var3",
            "Synth_Hyperplane_0p005_var2", "Synth_Hyperplane_0p01_var1", "Synth_Hyperplane_0p02_var3",
            "Synth_LED_0p005_var2", "Synth_LED_0p01_var1",
            "Synth_RandomTree_0p005_var2", "Synth_RandomTree_0p01_var1",
            "Synth_SEA_0p005_var2", "Synth_SEA_0p01_var1", "Synth_SEA_0p02_var3",
            "Synth_Waveform_0p02_var3",
        ],
    }

    model_order = [
        "Dynamic CS-ARF (Proposed)", "ARF (Standard)", "HAT", "OOB", "UOB",
        "CSARF-MCC (Aguiar)", "SMOTE-Window",
    ]

    lines = []
    lines.append("\\begin{longtable}{|l|l|c|c|c|c|c|}")
    lines.append("\\caption{Supplementary Table S1: Complete prequential performance across all 20 evaluated data streams and seven algorithms (G-Mean, Precision, Recall, $F_2$-Score, Time in seconds). Bold indicates the best value per metric within each dataset.} \\label{tab:supp_full_results} \\\\")
    lines.append("\\hline")
    lines.append("\\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{G-Mean} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F$_2$-Score} & \\textbf{Time (s)} \\\\ \\hline")
    lines.append("\\endfirsthead")
    lines.append("\\hline")
    lines.append("\\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{G-Mean} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F$_2$-Score} & \\textbf{Time (s)} \\\\ \\hline")
    lines.append("\\endhead")
    lines.append("\\hline \\multicolumn{7}{r}{\\textit{Continued on next page}} \\\\ \\hline")
    lines.append("\\endfoot")
    lines.append("\\hline")
    lines.append("\\endlastfoot")

    for category, datasets in category_order.items():
        lines.append(f"\\multicolumn{{7}}{{|l|}}{{\\textbf{{{category}}}}} \\\\ \\hline")
        for ds in datasets:
            subset = df[df["Dataset"] == ds].set_index("Model").loc[model_order].reset_index()
            max_gmean = subset["G-Mean"].max()
            max_prec = subset["Precision"].max()
            max_rec = subset["Recall"].max()
            max_f2 = subset["F2-Score"].max()

            n = len(subset)
            for i, row in subset.iterrows():
                gmean_str = format_bold(row["G-Mean"], row["G-Mean"] == max_gmean)
                prec_str = format_bold(row["Precision"], row["Precision"] == max_prec)
                rec_str = format_bold(row["Recall"], row["Recall"] == max_rec)
                f2_str = format_bold(row["F2-Score"], row["F2-Score"] == max_f2)
                time_str = format_bold(row["Time (s)"], False, is_time=True)
                alg = row["Model"]
                if i == 0:
                    lines.append(f"\\multirow{{{n}}}{{*}}{{{ds}}} & {alg} & {gmean_str} & {prec_str} & {rec_str} & {f2_str} & {time_str} \\\\ \\cline{{2-7}}")
                else:
                    end_line = "\\\\ \\hline" if i == n - 1 else "\\\\ \\cline{2-7}"
                    lines.append(f" & {alg} & {gmean_str} & {prec_str} & {rec_str} & {f2_str} & {time_str} {end_line}")

    lines.append("\\end{longtable}")

    os.makedirs(".", exist_ok=True)
    with open("full_results_table.tex", "w") as f:
        f.write("\n".join(lines))
    print("Table written to supplementary/full_results_table.tex")

if __name__ == "__main__":
    generate_table()
