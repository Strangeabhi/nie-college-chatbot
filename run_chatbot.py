from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import random
import time
import os
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

class NIEAdvancedChatbot:
    def __init__(self, faq_file='faq_data.json', embeddings_file='faq_embeddings.npy'):
        self.faq_file = faq_file
        self.embeddings_file = embeddings_file
        self.load_faq_data()
        self.load_embeddings()
        # Lazy init embedding model to avoid slow/failed cold starts on Railway
        self.embeddings_model = None
        self.conversation_memory = {}
        self.response_templates = {
            "cutoff": [
                "Based on the latest data, the cutoffs for {} are:\n{}",
                "Here are the cutoff ranks for {} at NIE:\n{}",
                "For {} admissions, you'll need these ranks:\n{}",
            ],
            "general": [
                "Great question! {}",
                "I'd be happy to help with that. {}",
                "Here's what you need to know: {}",
                "That's a common question! {}"
            ],
            "placements": [
                "NIE has excellent placement records! {}",
                "Placements at NIE are quite impressive: {}",
                "Here's the placement data you're looking for: {}",
            ],
            "hostel": [
                "Regarding hostel facilities at NIE: {}",
                "Here's what you need to know about hostels: {}",
                "NIE hostel information: {}",
            ]
        }
        logger.info("NIE Advanced Chatbot initialized ✅")

    def load_faq_data(self):
        with open(self.faq_file, 'r', encoding='utf-8') as f:
            self.faq_data = json.load(f)
        self.questions = []
        self.answers = []
        self.categories = []
        for category_data in self.faq_data:
            cat_name = category_data['category']
            for qa in category_data['questions']:
                self.questions.append(qa['question'])
                self.answers.append(qa['answer'])
                self.categories.append(cat_name)
        logger.info(f"Loaded {len(self.questions)} questions from {len(self.faq_data)} categories")

    def load_embeddings(self):
        try:
            self.embeddings = np.load(self.embeddings_file)
            logger.info("Loaded precomputed embeddings ✅")
        except FileNotFoundError:
            logger.error("Embeddings not found! Generate them locally first.")
            raise

    def get_encoder(self):
        if self.embeddings_model is None:
            try:
                model_name = os.getenv('SENTENCE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
                cache_dir = os.getenv('SENTENCE_CACHE_DIR', None)
                self.embeddings_model = SentenceTransformer(model_name, cache_folder=cache_dir) if cache_dir else SentenceTransformer(model_name)
                logger.info(f"SentenceTransformer model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embeddings_model = False  # sentinel meaning unavailable
        return self.embeddings_model

    def get_response_variation(self, answer: str) -> str:
        if random.random() > 0.3:
            return answer
        response_type = "general"
        if "cutoff" in answer.lower() or "rank" in answer.lower():
            response_type = "cutoff"
        elif "placement" in answer.lower():
            response_type = "placements"
        elif "hostel" in answer.lower():
            response_type = "hostel"
        templates = self.response_templates.get(response_type, self.response_templates["general"])
        template = random.choice(templates)
        if "{}" in template:
            return template.format(answer)
        else:
            return template + " " + answer

    def search_nie_website_fallback(self, query: str) -> str:
        if any(word in query.lower() for word in ['admission', 'cutoff', 'rank']):
            return "I don't have the latest admission details. Check https://nie.ac.in/admissions/ or contact admissions."
        elif any(word in query.lower() for word in ['placement', 'package', 'job']):
            return "I don't have the most recent placement statistics. Check https://nie.ac.in/placements/ or contact the placement cell."
        elif any(word in query.lower() for word in ['hostel', 'accommodation']):
            return "I don't have current hostel details. Check https://nie.ac.in/facilities/ or contact the hostel office."
        else:
            return f"I'm not sure about '{query}'. Check https://nie.ac.in/ or contact the relevant department."

    def get_cutoff_data(self, exam_type: str, user_history: list):
        if 'comedk' in exam_type:
            comedk_indices = [i for i, q in enumerate(self.questions) if 'comedk' in q.lower() and 'cutoff' in q.lower()]
            if comedk_indices:
                encoder = self.get_encoder()
                if encoder is False:
                    return "COMEDK cutoffs information is available, but the encoder failed to load on the server. Please retry in a minute.", 0.5
                query_emb = encoder.encode([exam_type])
                comedk_embeddings = self.embeddings[comedk_indices]
                similarities = cosine_similarity(query_emb, comedk_embeddings)[0]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                if best_score > 0.5:
                    answer = self.answers[comedk_indices[best_idx]]
                    return answer, float(best_score)
            return """COMEDK cutoffs (last year):
- CSE (E085): 10182
- CI (E085): 12789  
- ECE (E142): 29308
- EEE (E142): 101747
- ME (E142): 95259
- Civil (E142): 80212""", 0.8
        elif 'kcet' in exam_type:
            kcet_indices = [i for i, q in enumerate(self.questions) if 'kcet' in q.lower() and 'cutoff' in q.lower()]
            if kcet_indices:
                encoder = self.get_encoder()
                if encoder is False:
                    return "KCET cutoffs information is available, but the encoder failed to load on the server. Please retry in a minute.", 0.5
                query_emb = encoder.encode([exam_type])
                kcet_embeddings = self.embeddings[kcet_indices]
                similarities = cosine_similarity(query_emb, kcet_embeddings)[0]
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]
                if best_score > 0.5:
                    answer = self.answers[kcet_indices[best_idx]]
                    return answer, float(best_score)
            return """KCET cutoffs (last year):
- CSE (E178): 8726
- CI (E178): 11300
- ECE (Aided): 95447
- ME (Aided): 42543
- Civil (Aided): 95447
- EEE (Unaided): 35887
- ECE (Unaided): 48525
- ME (Unaided): 61681
- Civil (Unaided): 115835""", 0.8
        else:
            return "I can help with KCET and COMEDK cutoffs. Which exam?", 0.6

    def get_response(self, user_query: str, user_id: str = "default"):
        cleaned_query = user_query.lower().strip()
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        user_history = self.conversation_memory[user_id]

        if len(user_history) > 0 and any(word in user_history[-1].lower() for word in ['cutoff', 'cut off', 'ranks', 'rank']):
            if any(exam in cleaned_query for exam in ['kcet', 'comedk']):
                return self.get_cutoff_data(cleaned_query, user_history)

        generic_cutoff_phrases = ['tell me about cutoffs','tell me about cutoff','tell me about ranks','what are cutoffs','what are cutoff','what are ranks','cutoffs','cutoff','ranks','rank']
        if any(phrase in cleaned_query for phrase in generic_cutoff_phrases) and not any(specific in cleaned_query for specific in ['kcet','comedk','cse','ece','eee','me','civil']):
            user_history.append(user_query)
            return "Please specify which exam - KCET or COMEDK?", 0.8

        candidate_indices = list(range(len(self.questions)))
        if 'placement' in cleaned_query:
            preferred_indices = [i for i, q in enumerate(self.questions) if any(k in q.lower() for k in ['which companies visit','companies visit','recruiters','placement statistics','highest package','average package','package'])]
            avoid_indices = [i for i, a in enumerate(self.answers) if 'training in technical skills' in a.lower()]
            narrowed = [i for i in preferred_indices if i not in avoid_indices]
            if narrowed:
                candidate_indices = narrowed

        encoder = self.get_encoder()
        if encoder is False:
            # Fallback: simple keyword overlap search across questions when encoder unavailable
            tokens = set(cleaned_query.split())
            best_idx = 0
            best_score = 0.0
            for i, q in enumerate(self.questions):
                overlap = tokens.intersection(set(q.lower().split()))
                score = len(overlap) / (len(tokens) + 1e-6)
                if score > best_score:
                    best_idx = i
                    best_score = score
            if best_score > 0:
                answer = self.answers[best_idx]
                varied_answer = self.get_response_variation(answer)
                user_history.append(user_query)
                return varied_answer, float(min(0.49, best_score))
            fallback_response = self.search_nie_website_fallback(cleaned_query)
            user_history.append(user_query)
            return fallback_response, 0.3
        query_emb = encoder.encode([cleaned_query])
        if len(candidate_indices) != len(self.questions):
            subset_embs = self.embeddings[candidate_indices]
            similarities = cosine_similarity(query_emb, subset_embs)[0]
            local_best = int(np.argmax(similarities))
            best_idx = candidate_indices[local_best]
            best_score = similarities[local_best]
        else:
            similarities = cosine_similarity(query_emb, self.embeddings)[0]
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]

        if best_score >= 0.2:
            answer = self.answers[best_idx]
            varied_answer = self.get_response_variation(answer)
            user_history.append(user_query)
            return varied_answer, float(best_score)
        else:
            fallback_response = self.search_nie_website_fallback(cleaned_query)
            user_history.append(user_query)
            return fallback_response, 0.3

class FallbackChatbot:
    def __init__(self, error: str):
        self.error = error
        self.questions = []
        self.faq_data = []
    def get_response(self, user_query: str, user_id: str = "default"):
        return ("Service is starting up or temporarily unavailable. Please retry in a minute.", 0.1)

try:
    chatbot = NIEAdvancedChatbot()
except Exception as init_err:
    logger.error(f"Chatbot failed to initialize: {init_err}")
    chatbot = FallbackChatbot(str(init_err))

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')
        if not user_message:
            return jsonify({"response": "Please provide a message."}), 400
        start_time = time.time()
        bot_response, confidence = chatbot.get_response(user_message, user_id)
        response_time = time.time() - start_time
        logger.info(f"Query: {user_message} | Confidence: {confidence:.2f} | Time: {response_time:.2f}s")
        return jsonify({'response': bot_response,'confidence': confidence,'response_time': response_time})
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({'response': 'Error processing request.'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy","questions": len(chatbot.questions),"categories": len(chatbot.faq_data)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
