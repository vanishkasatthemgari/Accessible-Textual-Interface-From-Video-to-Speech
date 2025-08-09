import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Original OCR metric data
data = {
    "Language": ["French", "English", "Telugu", "Tamil", "Hindi"],
    "Accuracy (%)": [98.01, 95.10, 100.00, 92.92, 100.00],
    "Precision (%)": [97.58, 94.26, 100.00, 86.43, 99.60],
    "Recall (%)": [98.01, 95.10, 100.00, 92.92, 100.00]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate F1 Score
df["F1 Score (%)"] = 2 * (df["Precision (%)"] * df["Recall (%)"]) / (df["Precision (%)"] + df["Recall (%)"])

# ---- PART 1: Comparison Table & Grouped Bar Chart ----

print("\nFull OCR Metrics Comparison Table:\n")
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

# Grouped bar chart: All metrics across all languages
df_plot = df.set_index("Language")
df_plot.plot(kind="bar", figsize=(10, 6), color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
plt.title("OCR Metrics by Language")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---- PART 2: Individual Metric Tables & Bar Charts ----
metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
colors = {
    "Accuracy (%)": "#4CAF50",
    "Precision (%)": "#2196F3",
    "Recall (%)": "#FF9800",
    "F1 Score (%)": "#9C27B0"
}

for metric in metrics:
    print(f"\n{metric} Comparison Table:\n")
    sub_df = df[["Language", metric]]
    print(tabulate(sub_df, headers='keys', tablefmt='grid', showindex=False))

    # Plot metric-specific bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(sub_df["Language"], sub_df[metric], color=colors[metric])
    plt.title(f"{metric} Across Languages")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
