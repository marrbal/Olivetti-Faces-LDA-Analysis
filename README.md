Olivetti Faces LDA Analysis

Key Steps:

1. Dataset Processing:
Fetches the Olivetti Faces dataset.
Converts grayscale facial images into numerical "documents."
Preprocesses these pseudo-text documents by filtering out noise.

2. Latent Dirichlet Allocation (LDA):
Initializes an LDA model with tomotopy.
Trains the model on the transformed dataset.
Extracts and displays dominant topics.

Output:

A text file (olivetti_faces_documents.txt) containing processed pseudo-text documents.
Topic distributions based on image data.
