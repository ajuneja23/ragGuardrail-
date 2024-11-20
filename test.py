from transformers import AutoTokenizer
from chroma_client import get_chroma_client
import matplotlib.pyplot as plt


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Example text
text = "nfidjvnfck"


client = get_chroma_client()
collection = client.get_collection("red_team_conversations")
if collection:
    print(collection.count())
similarity_scores = collection.query(
    query_texts=[text], n_results=1
)  # Changed 'queries' to 'query'
print(similarity_scores)
print(similarity_scores["distances"][0])
import pandas as pd

toxic_comments_df = pd.read_csv("train.csv")

print(toxic_comments_df.head())

toxic_comments_df = toxic_comments_df.drop(
    columns=["severe_toxic", "obscene", "threat", "insult", "identity_hate"]
)

print((toxic_comments_df["toxic"] >= 1).sum())
correct_predictions = 0
total_predictions = 0
predictions = []

for index, row in toxic_comments_df.iterrows():
    print("Row #:", index, "/", len(toxic_comments_df))
    if index > 1000:
        break
    comment_text = row["comment_text"]
    similarity_scores = collection.query(query_texts=[comment_text], n_results=1)
    print(similarity_scores["distances"])

    # Store the similarity score
    score = similarity_scores["distances"][0][0]

    # Initialize lists to store scores for positives and negatives
    if "positive_scores" not in locals():
        positive_scores = []
    if "negative_scores" not in locals():
        negative_scores = []

    if score < 1.1:
        predictions.append(1)  # Predicted toxic
        positive_scores.append(score)  # Store positive score
    else:
        predictions.append(0)  # Predicted non-toxic
        negative_scores.append(score)  # Store negative score

    total_predictions += 1
    if predictions[-1] == row["toxic"]:
        correct_predictions += 1

import matplotlib.pyplot as plt

# Create a bar chart for predictions
labels = ["Toxic", "Non-Toxic"]
counts = [sum(predictions), total_predictions - sum(predictions)]

plt.bar(labels, counts, color=["red", "green"])
plt.title("Toxicity Predictions")
plt.xlabel("Prediction Type")
plt.ylabel("Count")
plt.show()

# Print confusion matrix
true_positives = sum(
    1
    for i in range(total_predictions)
    if predictions[i] == 1 and toxic_comments_df.iloc[i]["toxic"] == 1
)
true_negatives = sum(
    1
    for i in range(total_predictions)
    if predictions[i] == 0 and toxic_comments_df.iloc[i]["toxic"] == 0
)
false_positives = sum(
    1
    for i in range(total_predictions)
    if predictions[i] == 1 and toxic_comments_df.iloc[i]["toxic"] == 0
)
false_negatives = sum(
    1
    for i in range(total_predictions)
    if predictions[i] == 0 and toxic_comments_df.iloc[i]["toxic"] == 1
)

print("Confusion Matrix:")
print(
    f"TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}"
)
