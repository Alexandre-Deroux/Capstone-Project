import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from capstone_project import analyse_aml_risk

# Load transactions from transactions.csv
df = pd.read_csv("transactions.csv")

# Function to extract the score from the output string
def extract_score(output):
    matches = re.findall(r"(\d{1,2}|100)/100", output)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"(\d{1,2})-(\d{1,2}|100)", output)
    valid_matches = [int(match[1]) for match in matches if int(match[1]) > 5]
    if valid_matches:
        return valid_matches[-1]
    matches = re.findall(r"\d{1,2}|100", output)
    valid_matches = [int(match) for match in matches if int(match) > 6]
    if valid_matches:
        return valid_matches[-1]
    return -1

# Apply the analyse_aml_risk function to each transaction with a progress bar
outputs = []
scores = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Test processing", unit="txn"):
    sender_name = row["sender"]
    sender_dict = {
        "Country": row["sender_country"],
        "Address": row["sender_address"]
    }
    sender_info = pd.DataFrame([sender_dict], index=[sender_name]).replace({"": None}).dropna(axis=1, how="all").T

    recipient_name = row["recipient"]
    recipient_dict = {
        "Country": row["recipient_country"],
        "Address": row["recipient_address"]
    }
    recipient_info = pd.DataFrame([recipient_dict], index=[recipient_name]).replace({"": None}).dropna(axis=1, how="all").T

    transaction_info = {
        "Amount": row["amount"],
        "Currency": row["currency"],
        "Reference": row["reference"],
        "Reason": row["reason"]
    }

    # Run AML analysis
    for k in range(5):
        risk_analysis_result = analyse_aml_risk(sender_info, recipient_info, transaction_info)
        if isinstance(risk_analysis_result, str) and len(risk_analysis_result.strip()) > 0:
            outputs.append(risk_analysis_result)
            break

    # Extract the score and determine if it is fraud
    score = extract_score(risk_analysis_result)
    scores.append(score)

# Store outputs and scores
df["output"] = outputs
df["score"] = scores

# Determine the best threshold
thresholds = np.linspace(0, 100, 101)
best_accuracy = 0
best_threshold = 0
for threshold in thresholds:
    predicted_frauds = np.where(df["score"] > threshold, "Fraud", "No Fraud")
    correct_predictions = sum(df["fraud"] == predicted_frauds)
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

# Store test results
test_results = np.where(df["score"] > best_threshold, "Fraud", "No Fraud")
df["predicted_fraud"] = test_results

# Save the results to a CSV file for analysis
df.set_index("sender", inplace=True)
df.to_csv("test_results.csv")

# Calculate the accuracy score
correct_predictions = sum(df["fraud"] == df["predicted_fraud"])
total_predictions = len(df)
accuracy = (correct_predictions / total_predictions) * 100

# Display results
print(f"\nAccuracy score: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")