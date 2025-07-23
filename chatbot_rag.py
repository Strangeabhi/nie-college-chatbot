import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_filler(user_input):
    user_input = user_input.lower().strip()
    filler_patterns = [
        r"^and\s+", r"^also\s+", r"^what about\s+", r"^tell me about\s+",
        r"^do you know about\s+", r"^can you tell me about\s+", r"^i want to know about\s+"
    ]
    for pattern in filler_patterns:
        user_input = re.sub(pattern, '', user_input)
    return user_input.strip()

FAQ_PATH = 'faq_data.json'
EMBEDDINGS_CACHE = 'faq_embeddings.npy'
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.75

class RAGChatbot:
    def __init__(self, faq_path=FAQ_PATH, model_name=MODEL_NAME, cache_path=EMBEDDINGS_CACHE):
        self.faq_path = faq_path
        self.model_name = model_name
        self.cache_path = cache_path
        self.model = SentenceTransformer(self.model_name)
        self.faq_data = self._load_faq()
        self.questions = [item['question'] for item in self.faq_data]
        self.answers = [item['answer'] for item in self.faq_data]
        self.embeddings = self._load_or_create_embeddings()

    def _load_faq(self):
        with open(self.faq_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_or_create_embeddings(self):
        if os.path.exists(self.cache_path):
            return np.load(self.cache_path)
        embeddings = self.model.encode(self.questions, show_progress_bar=True)
        np.save(self.cache_path, embeddings)
        return embeddings

    def get_response(self, user_query):
        cleaned_query = clean_filler(user_query)
        query_emb = self.model.encode([cleaned_query])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        if best_score >= SIMILARITY_THRESHOLD:
            return self.answers[best_idx], float(best_score)
        return "Sorry, I didn’t understand that. Can you rephrase?", float(best_score)

# Singleton instance for API use
chatbot = RAGChatbot() 