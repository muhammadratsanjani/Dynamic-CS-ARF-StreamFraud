import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv("data/raw/creditcard.csv")

# Convert Time from seconds to hours
df['Hour'] = df['Time'] / 3600.0

# Create a figure with 2 subplots (1 for transaction volume, 1 for fraud rate)
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot histogram of legitimate transactions
sns.histplot(df[df['Class'] == 0]['Hour'], bins=48, color='blue', alpha=0.5, label='Legitimate', ax=ax1)
ax1.set_xlabel('Time (Hours)')
ax1.set_ylabel('Number of Legitimate Transactions', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlim(0, 48)

# Create a second y-axis for the fraud rate
ax2 = ax1.twinx()

# Calculate hourly fraud rate
df['Hour_Bin'] = np.floor(df['Hour'])
fraud_rates = df.groupby('Hour_Bin')['Class'].mean() * 100 # percentage
hours = fraud_rates.index

# Plot fraud rate
ax2.plot(hours, fraud_rates.values, color='red', marker='o', linestyle='-', linewidth=2, label='Fraud Rate (%)')
ax2.set_ylabel('Fraud Rate (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and layout
plt.title('Distribution of Transactions and Fraud Rate over 48 Hours\n(Indicating Concept Drift)')
fig.tight_layout()

# Add legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.savefig("figures/data_distribution.pdf", format='pdf', bbox_inches='tight')
print("Plot saved to figures/data_distribution.pdf")
