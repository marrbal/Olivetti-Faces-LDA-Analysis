from sklearn.datasets import fetch_olivetti_faces
import tomotopy as tp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import numpy as np

# Step 1: Fetch the Olivetti Faces Dataset
print("Fetching Olivetti Faces dataset...")
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)

# Convert numerical data to dummy textual documents for LDA
# Each image (64x64) is flattened to a single row of numbers and treated as a pseudo-document
images_as_text = [
    ' '.join(map(str, np.round(image * 255).astype(int))) for image in faces_data.data
]

# Step 2: Preprocess the documents
def preprocess_documents(docs):
    preprocessed_docs = []
    for doc in docs:
        # Remove non-alphabetic characters
        doc = re.sub(r'[^0-9\s]', '', doc)  # Keep numeric tokens since data is numerical
        # Split tokens and remove very short ones
        words = [word for word in doc.split() if len(word) > 1]  # Filtering short numeric values
        # Reconstruct the document if it has sufficient length
        if len(words) > 5:
            preprocessed_docs.append(' '.join(words))
    return preprocessed_docs

documents = preprocess_documents(images_as_text)

if not documents:
    raise ValueError("No valid documents found. Ensure the dataset has non-empty files with sufficient content.")

print(f"Total preprocessed documents: {len(documents)}")
print("First 5 preprocessed documents:", documents[:5])  # Debugging print statement

# Save documents to a single txt file
output_file = "olivetti_faces_documents.txt"
print(f"Saving documents to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(doc + '\n')
print(f"Documents saved to {output_file}")

# Step 3: Perform LDA using Tomotopy
# Initialize LDA model
K = 5  # Start with 5 topics; adjust as needed
mdl = tp.LDAModel(k=K)

# Add documents to the model
print("Adding documents to the LDA model...")
for doc in documents:
    mdl.add_doc(doc.split())

# Train the model
iterations = 1000  # Number of iterations
print(f"Training LDA model with {K} topics for {iterations} iterations...")
mdl.train(iterations)

# Display the topics
print("Displaying topics:")
for i in range(K):
    topic_words = mdl.get_topic_words(i, top_n=10)
    print(f"Topic {i}: {topic_words}")

print("LDA analysis completed.")
