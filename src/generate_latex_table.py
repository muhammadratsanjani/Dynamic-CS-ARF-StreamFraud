import pandas as pd
import os

def format_bold(val, is_max, is_time=False):
    if is_time:
        s = f"{val:.2f}"
    else:
        s = f"{val:.4f}"
    
    if is_max:
        return f"\\textbf{{{s}}}"
    return s

def generate_table():
    df = pd.read_csv("data/processed/benchmark_results_detailed.csv")
    
    # Filter only 6 representative streams for the main manuscript table to avoid layout breakage
    primary_datasets = ["Agrawal", "BankSim", "IEEE-CIS", "PaySim", "SEA", "ULB"]
    datasets = [d for d in df["Dataset"].unique() if d in primary_datasets]
    
    lines = []
    lines.append("\\begin{table*}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\caption{Detailed Performance Metrics across All Data Streams}")
    lines.append("    \\label{tab:detailed_results}")
    lines.append("    \\resizebox{\\textwidth}{!}{%")
    lines.append("    \\begin{tabular}{|l|l|c|c|c|c|c|}")
    lines.append("        \\hline")
    lines.append("        \\textbf{Dataset} & \\textbf{Algorithm} & \\textbf{G-Mean} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F$_2$-Score} & \\textbf{Time (s)} \\\\ \\hline")
    
    for ds in datasets:
        subset = df[df["Dataset"] == ds]
        # find max values
        max_gmean = subset["G-Mean"].max()
        max_prec = subset["Precision"].max()
        max_rec = subset["Recall"].max()
        max_f2 = subset["F2-Score"].max()
        
        n = len(subset)
        first = True
        for idx, row in subset.iterrows():
            gmean_str = format_bold(row["G-Mean"], row["G-Mean"] == max_gmean)
            prec_str = format_bold(row["Precision"], row["Precision"] == max_prec)
            rec_str = format_bold(row["Recall"], row["Recall"] == max_rec)
            f2_str = format_bold(row["F2-Score"], row["F2-Score"] == max_f2)
            time_str = format_bold(row["Time (s)"], False, is_time=True)
            
            alg = row["Model"]
            if first:
                lines.append(f"        \\multirow{{{n}}}{{*}}{{\\textbf{{{ds}}}}} & {alg} & {gmean_str} & {prec_str} & {rec_str} & {f2_str} & {time_str} \\\\ \\cline{{2-7}}")
                first = False
            else:
                end_line = "\\\\ \\hline" if idx == subset.index[-1] else "\\\\ \\cline{2-7}"
                lines.append(f"        & {alg} & {gmean_str} & {prec_str} & {rec_str} & {f2_str} & {time_str} {end_line}")
                
    lines.append("    \\end{tabular}%")
    lines.append("    }")
    lines.append("\\end{table*}")
    
    os.makedirs("figures", exist_ok=True)
    with open("figures/detailed_table.tex", "w") as f:
        f.write("\n".join(lines))
    print("Table written to figures/detailed_table.tex")

if __name__ == "__main__":
    generate_table()
