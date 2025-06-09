import csv
import re
from collections import defaultdict

# Read raw data from file
with open("models/modelStats.txt", "r") as file:
    raw_data = file.read()

# Regex pattern to parse the line
pattern = re.compile(
    r"MODEL:(.*) , Train Loss: ([\d.]+), Train Accuracy: ([\d.]+), Validation Loss: ([\d.]+), Validation Accuracy: ([\d.]+)%"
)

# Store grouped models with their respective stats
models = defaultdict(list)

# Process the raw data and group by model base name
for match in pattern.finditer(raw_data):
    model, train_loss, train_acc, val_loss, val_acc = match.groups()

    # Remove trailing '-1', '-2', '-3' if present
    base_model = re.sub(r"-\d+$", "", model)

    # Store the model performance stats in the dictionary
    models[base_model].append(
        [
            float(train_loss),
            float(train_acc),
            float(val_loss),
            float(val_acc.strip("%")),  # Remove '%' from validation accuracy
        ]
    )

# Prepare the rows with averaged data
averaged_rows = []
for model, stats in models.items():
    # Calculate the averages for each stat
    avg_train_loss = sum(stat[0] for stat in stats) / len(stats)
    avg_train_acc = sum(stat[1] for stat in stats) / len(stats)
    avg_val_loss = sum(stat[2] for stat in stats) / len(stats)
    avg_val_acc = sum(stat[3] for stat in stats) / len(stats)

    # Append the averaged result for this model
    averaged_rows.append(
        [model, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
    )

# Write the averaged data to a CSV file
csv_filename = "averaged_model_performance.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Header row
    writer.writerow(
        [
            "Model",
            "Average Train Loss",
            "Average Train Accuracy",
            "Average Validation Loss",
            "Average Validation Accuracy",
        ]
    )

    # Write averaged data
    writer.writerows(averaged_rows)

print(f" Averaged CSV file saved as '{csv_filename}'")
