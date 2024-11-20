import json
from chroma_client import get_chroma_client
from chromadb import Client

# Load the data from red_team_attempts.jsonl
data = []
with open("red_team_attempts.jsonl", "r") as file:
    for line in file:
        line = line.strip()
        if line:
            try:
                line = line[:-1]
                if line[0] == "[":
                    line = line[1:]
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                pass  # print(f"Error decoding JSON: {e} - Line: {line}")

# Initialize ChromaDB client and create a single collection
client = get_chroma_client()
collection = client.create_collection("red_team_conversations")

convos = []
for index, convo in enumerate(data):
    if index > 500:
        break
    print(f"Processing convo {index + 1} out of {len(data)}")  # Progress printing
    if "transcript" in convo:
        transcript = convo["transcript"]
        turns = []  # Initialize turns for each transcript
        start_index = 0
        while start_index < len(transcript):
            next_index = transcript.find("Human:", start_index)
            assistant_index = transcript.find("Assistant:", start_index)

            # Determine the next index to process
            if next_index == -1 and assistant_index == -1:
                break  # No more turns found
            elif next_index == -1 or (
                assistant_index != -1 and assistant_index < next_index
            ):
                next_index = assistant_index

            # Get the next turn up to the next indicator
            next_turn = transcript[start_index:next_index].strip()
            if next_turn:
                turns.append(next_turn)

            # Move start_index to the end of the current turn
            start_index = (
                next_index + len("Human:")
                if transcript[next_index : next_index + len("Human:")] == "Human:"
                else next_index + len("Assistant:")
            )

            # Add the next_turn to the collection with a unique ID
            if next_turn:
                unique_id = f"{index}_{next_index}"  # Create a unique ID using index and next_index
                collection.add(
                    documents=[next_turn],
                    ids=[unique_id],  # Use the unique ID for the document
                )
