from openai import OpenAI
import numpy as np

API_KEY = ''
client = OpenAI(api_key=API_KEY)


def get_embedding_semantic_similarity(sentence1, sentence2):
    vec1 = client.embeddings.create(input=[sentence1], model="text-embedding-3-large").data[0].embedding
    vec2 = client.embeddings.create(input=[sentence2], model="text-embedding-3-large").data[0].embedding

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an NLP assistant, skilled in evaluating how similar given two sentences are."},
    {"role": "user", "content": "Provide a semantic similarity score for given sentences A and B. Semantic similarity score is between -1 and 1 where 1 means they are perfectly similar and -1 mean they are opposite while 0 means their meaning are uncorrelated. Sentence A=(The cat sat on the mat.) Sentence B=(The feline rested on the rug.)"}
  ]
)

print(completion.choices[0].message)

# # Example usage
sentence1 = "enes went to the grocery store and bought some oranges"
sentence2 = "oranges went to the grocery store and bought some enes"
similarity_score = get_embedding_semantic_similarity(sentence1, sentence2)
print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print("Semantic Similarity:", similarity_score)

