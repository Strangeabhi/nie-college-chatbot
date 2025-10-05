"""
NIE Advanced Chatbot - Single File Solution
Everything you need in one place!
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import numpy as np
import traceback
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Core imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class NIEAdvancedChatbot:
    def __init__(self, faq_file='faq_data.json', embeddings_file='faq_embeddings.npy'):
        """Initialize the advanced chatbot"""
        self.faq_file = faq_file
        self.embeddings_file = embeddings_file
        
        # Load FAQ data
        self.load_faq_data()
        
        # Initialize embeddings
        self.setup_embeddings()
        
        # Conversation memory
        self.conversation_memory = {}
        
        # Response templates for dynamic answers
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
        
        logger.info("NIE Advanced Chatbot initialized successfully")

    def load_faq_data(self):
        """Load FAQ data"""
        try:
            with open(self.faq_file, 'r', encoding='utf-8') as f:
                self.faq_data = json.load(f)
            
            # Flatten questions and answers
            self.questions = []
            self.answers = []
            self.categories = []
            
            for category_data in self.faq_data:
                category_name = category_data['category']
                for qa in category_data['questions']:
                    self.questions.append(qa['question'])
                    self.answers.append(qa['answer'])
                    self.categories.append(category_name)
            
            logger.info(f"Loaded {len(self.questions)} questions from {len(self.faq_data)} categories")
            
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            raise

    def setup_embeddings(self):
        """Setup embeddings"""
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            try:
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded existing embeddings")
            except FileNotFoundError:
                logger.info("Generating new embeddings...")
                self.embeddings = self.embeddings_model.encode(self.questions)
                np.save(self.embeddings_file, self.embeddings)
                logger.info("Embeddings generated and saved")
            
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise

    def get_response_variation(self, answer: str) -> str:
        """Generate varied response - but only sometimes"""
        try:
            # Only add variation 30% of the time to avoid being too repetitive
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
                
        except Exception as e:
            return answer

    def search_nie_website_fallback(self, query: str) -> str:
        """Simple fallback to NIE website"""
        if any(word in query.lower() for word in ['admission', 'cutoff', 'rank']):
            return "I don't have the latest admission details in my database. For the most current information, I'd recommend checking the official NIE website at https://nie.ac.in/admissions/ or contacting the admissions office directly."
        elif any(word in query.lower() for word in ['placement', 'package', 'job']):
            return "I don't have the most recent placement statistics. For detailed information about companies and packages, check the official NIE website at https://nie.ac.in/placements/ or contact the placement cell."
        elif any(word in query.lower() for word in ['hostel', 'accommodation']):
            return "I don't have current hostel details. For the latest information about facilities and availability, visit https://nie.ac.in/facilities/ or contact the hostel office."
        else:
            return f"I'm not sure about '{query}' specifically. You might find more detailed information on the official NIE website at https://nie.ac.in/ or by contacting the relevant department directly."

    def get_cutoff_data(self, exam_type: str, user_history: List[str]) -> Tuple[str, float]:
        """Get specific cutoff data based on exam type"""
        try:
            if 'comedk' in exam_type:
                # Search for COMEDK cutoff data
                comedk_questions = [q for q in self.questions if 'comedk' in q.lower() and 'cutoff' in q.lower()]
                if comedk_questions:
                    # Find the best match
                    query_emb = self.embeddings_model.encode([exam_type])
                    comedk_indices = [i for i, q in enumerate(self.questions) if 'comedk' in q.lower() and 'cutoff' in q.lower()]
                    comedk_embeddings = self.embeddings[comedk_indices]
                    similarities = cosine_similarity(query_emb, comedk_embeddings)[0]
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]
                    
                    if best_score > 0.5:
                        answer = self.answers[comedk_indices[best_idx]]
                        return answer, float(best_score)
                
                # Fallback COMEDK data
                return """COMEDK cutoffs for NIE (last available year):
- CSE (E085): 10182
- CI (E085): 12789  
- ECE (E142): 29308
- EEE (E142): 101747
- ME (E142): 95259
- Civil Engineering (E142): 80212

Note: These are last year's figures and may change for current admissions.""", 0.8
            
            elif 'kcet' in exam_type:
                # Search for KCET cutoff data
                kcet_questions = [q for q in self.questions if 'kcet' in q.lower() and 'cutoff' in q.lower()]
                if kcet_questions:
                    query_emb = self.embeddings_model.encode([exam_type])
                    kcet_indices = [i for i, q in enumerate(self.questions) if 'kcet' in q.lower() and 'cutoff' in q.lower()]
                    kcet_embeddings = self.embeddings[kcet_indices]
                    similarities = cosine_similarity(query_emb, kcet_embeddings)[0]
                    best_idx = np.argmax(similarities)
                    best_score = similarities[best_idx]
                    
                    if best_score > 0.5:
                        answer = self.answers[kcet_indices[best_idx]]
                        return answer, float(best_score)
                
                # Fallback KCET data
                return """KCET cutoffs for NIE (last available year):
College code (E178):
- CSE: 8726
- CI: 11300

College code (E022):
- ECE (Aided): 95447
- ME (Aided): 42543
- Civil Engineering (Aided): 95447

College code (E056):
- EEE (Unaided): 35887
- ECE (Unaided): 48525
- ME (Unaided): 61681
- Civil Engineering (Unaided): 115835

Note: These are last year's figures and may change for current admissions.""", 0.8
            
            else:
                return "I can help with both KCET and COMEDK cutoffs. Which exam would you like to know about?", 0.6
                
        except Exception as e:
            logger.error(f"Error getting cutoff data: {e}")
            return "I'm having trouble retrieving the cutoff data. Please try asking about KCET or COMEDK cutoffs specifically.", 0.3

    def get_response(self, user_query: str, user_id: str = "default") -> Tuple[str, float]:
        """Get chatbot response with conversation memory"""
        try:
            cleaned_query = user_query.lower().strip()
            
            # Get user's conversation history
            if user_id not in self.conversation_memory:
                self.conversation_memory[user_id] = []
            
            user_history = self.conversation_memory[user_id]
            
            # Let FAQ search handle all responses - no hardcoded overrides
            
            # Check if user was asking about cutoffs in previous message
            if len(user_history) > 0 and any(word in user_history[-1].lower() for word in ['cutoff', 'cut off', 'ranks', 'rank']):
                # User was asking about cutoffs, now specifying exam type
                if any(exam in cleaned_query for exam in ['kcet', 'comedk']):
                    return self.get_cutoff_data(cleaned_query, user_history)
            
            # Special handling for generic cutoff queries - check this BEFORE semantic search
            generic_cutoff_phrases = [
                'tell me about cutoffs', 'tell me about cutoff', 'tell me about ranks',
                'what are cutoffs', 'what are cutoff', 'what are ranks',
                'cutoffs', 'cutoff', 'ranks', 'rank'
            ]
            
            if any(phrase in cleaned_query for phrase in generic_cutoff_phrases) and \
               not any(specific in cleaned_query for specific in ['kcet', 'comedk', 'cse', 'ece', 'eee', 'me', 'civil']):
                # Store this in memory
                user_history.append(user_query)
                return "I'd be happy to help with cutoff information! Could you please specify which exam you're interested in - KCET or COMEDK? This will help me provide you with the most accurate and relevant cutoff data.", 0.8
            
            # Perform semantic search with light intent routing for placements
            candidate_indices = list(range(len(self.questions)))
            if 'placement' in cleaned_query or 'placements' in cleaned_query:
                preferred_indices = [
                    i for i, q in enumerate(self.questions)
                    if any(k in q.lower() for k in [
                        'which companies visit', 'companies visit', 'recruiters', 'placement statistics',
                        'how are placements', 'highest package', 'average package', 'package'
                    ])
                ]
                avoid_indices = [
                    i for i, a in enumerate(self.answers)
                    if 'training in technical skills' in a.lower()
                ]
                # Build candidate set: prefer detailed answers, avoid generic training-only responses
                narrowed = [i for i in preferred_indices if i not in avoid_indices]
                if not narrowed:
                    narrowed = [i for i, q in enumerate(self.questions) if 'placement' in q.lower() and i not in avoid_indices]
                if narrowed:
                    candidate_indices = narrowed

            query_emb = self.embeddings_model.encode([cleaned_query])
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
            
            # High confidence - return FAQ answer with variation
            if best_score >= 0.5:
                answer = self.answers[best_idx]
                varied_answer = self.get_response_variation(answer)
                # Store conversation
                user_history.append(user_query)
                return varied_answer, float(best_score)
            
            # Medium confidence - still use FAQ answer
            elif best_score >= 0.2:
                answer = self.answers[best_idx]
                varied_answer = self.get_response_variation(answer)
                # Store conversation
                user_history.append(user_query)
                return varied_answer, float(best_score)
            
            # Low confidence - generic fallback
            else:
                fallback_response = self.search_nie_website_fallback(cleaned_query)
                # Store conversation
                user_history.append(user_query)
                return fallback_response, 0.3
                
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return "Sorry, I encountered a technical issue while processing your request. Please try again in a moment.", 0.0

# Initialize chatbot
chatbot = NIEAdvancedChatbot()

@app.route('/')
def index():
    """Serve the original NIE-themed chatbot interface"""
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')
        
        if not user_message:
            return jsonify({"response": "Please provide a message."}), 400
        
        start_time = time.time()
        bot_response, confidence = chatbot.get_response(user_message, user_id)
        response_time = time.time() - start_time
        
        logger.info(f"Query: {user_message}")
        logger.info(f"Confidence: {confidence:.2f}")
        logger.info(f"Response time: {response_time:.2f}s")
        
        return jsonify({
            'response': bot_response,
            'confidence': confidence,
            'response_time': response_time
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "questions": len(chatbot.questions),
        "categories": len(chatbot.faq_data)
    })

# Using original NIE-themed frontend from index.html

if __name__ == '__main__':
    print("🚀 Starting NIE Advanced Chatbot...")
    print("💬 Dynamic responses and external fallback")
    print("⚡ Fast performance with caching")
    print("\nAccess the chatbot at: http://localhost:5000")
    print("Health check at: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
