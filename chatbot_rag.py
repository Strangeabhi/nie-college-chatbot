import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import traceback

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
        self.questions = [qa['question'] for cat in self.faq_data for qa in cat['questions']]
        self.answers = [qa['answer'] for cat in self.faq_data for qa in cat['questions']]
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
        try:
            cleaned_query = clean_filler(user_query)
            # Check if query is about cutoffs but too generic
            cutoff_keywords = ['cutoff', 'cut off', 'cut-off', 'rank', 'ranks']
            branch_keywords = ['cse', 'ece', 'eee', 'me', 'mechanical', 'civil', 'ci', 'ise', 'computer science', 'electronics', 'electrical', 'mechanical engineering', 'civil engineering']
            # Check if user is asking for a specific branch cutoff
            if any(keyword in cleaned_query.lower() for keyword in cutoff_keywords):
                # Check if a specific branch is mentioned
                if any(branch in cleaned_query.lower() for branch in branch_keywords):
                    # User asked for specific branch cutoff - provide comprehensive answer
                    if 'cse' in cleaned_query.lower():
                        return "CSE (Computer Science & Engineering) cutoffs for NIE:\n\nKCET:\n- College code (E178): 8726\n\nCOMEDK:\n- College code (E085): 10182", 0.0
                    elif 'ci' in cleaned_query.lower():
                        return "CI (Computer Science & Engineering – AI & ML) cutoffs for NIE:\n\nKCET:\n- College code (E178): 11300\n\nCOMEDK:\n- College code (E085): 12789", 0.0
                    elif 'ece' in cleaned_query.lower():
                        return "ECE (Electronics & Communication Engineering) cutoffs for NIE:\n\nKCET:\n- College code (E022) - Aided: 95447\n- College code (E056) - Unaided: 48525\n\nCOMEDK:\n- College code (E142): 29308", 0.0
                    elif 'eee' in cleaned_query.lower():
                        return "EEE (Electrical & Electronics Engineering) cutoffs for NIE:\n\nKCET:\n- College code (E056) - Unaided: 35887\n\nCOMEDK:\n- College code (E142): 101747", 0.0
                    elif any(mech in cleaned_query.lower() for mech in ['me', 'mechanical']):
                        return "ME (Mechanical Engineering) cutoffs for NIE:\n\nKCET:\n- College code (E022) - Aided: 42543\n- College code (E056) - Unaided: 61681\n\nCOMEDK:\n- College code (E142): 95259", 0.0
                    elif 'civil' in cleaned_query.lower():
                        return "Civil Engineering cutoffs for NIE:\n\nKCET:\n- College code (E022) - Aided: 95447\n- College code (E056) - Unaided: 115835\n\nCOMEDK:\n- College code (E142): 80212", 0.0
                    elif 'ise' in cleaned_query.lower():
                        return "ISE (Information Science & Engineering) cutoffs for NIE:\n\nNote: ISE cutoffs are typically similar to CSE. For the most accurate information, please check the official KCET/COMEDK websites or contact the college directly.", 0.0
                    else:
                        # Let normal similarity search handle it
                        pass
                else:
                    # Generic cutoff question - show guidance
                    return "I can help you with cutoffs! Please specify which type:\n- KCET cutoffs\n- COMEDK cutoffs\n\nYou can also ask about specific branches like 'CSE cutoff' or 'ECE KCET rank'.", 0.0
            query_emb = self.model.encode([cleaned_query])
            sims = cosine_similarity(query_emb, self.embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            if best_score >= SIMILARITY_THRESHOLD:
                return self.answers[best_idx], float(best_score)
            return "Sorry, I didn't understand that. Can you rephrase?", float(best_score)
        except Exception as e:
            print('--- Exception in chatbot logic ---')
            print(traceback.format_exc())
            return "Sorry, I couldn't process your request due to a technical issue. Please try again later.", 0.0

# Singleton instance for API use
chatbot = RAGChatbot() 