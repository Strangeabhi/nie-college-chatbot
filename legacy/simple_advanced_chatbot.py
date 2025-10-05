"""
Simplified Advanced RAG Chatbot without complex LangChain dependencies
Focuses on the core advanced features that work reliably
"""

import json
import numpy as np
import traceback
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# Core imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAdvancedRAGChatbot:
    def __init__(self, faq_file='faq_data.json', embeddings_file='faq_embeddings.npy'):
        """Initialize the simplified advanced RAG chatbot"""
        self.faq_file = faq_file
        self.embeddings_file = embeddings_file
        
        # Load FAQ data
        self.load_faq_data()
        
        # Initialize embeddings
        self.setup_embeddings()
        
        # Response variations for dynamic answers
        self.response_templates = self.load_response_templates()
        
        logger.info("Simple Advanced RAG Chatbot initialized successfully")

    def load_faq_data(self):
        """Load and process FAQ data"""
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
            # Initialize embeddings model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to load existing embeddings
            try:
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded existing embeddings from {self.embeddings_file}")
            except FileNotFoundError:
                logger.info("Generating new embeddings...")
                self.generate_embeddings()
            
            logger.info("Embeddings setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise

    def generate_embeddings(self):
        """Generate embeddings for FAQ questions"""
        try:
            logger.info("Generating embeddings for FAQ questions...")
            self.embeddings = self.embeddings_model.encode(self.questions)
            
            # Save embeddings
            np.save(self.embeddings_file, self.embeddings)
            logger.info(f"Embeddings saved to {self.embeddings_file}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def load_response_templates(self) -> Dict[str, List[str]]:
        """Load response variation templates"""
        return {
            "cutoff": [
                "Based on the latest data, the cutoffs for {} are:\n{}",
                "Here are the cutoff ranks for {} at NIE:\n{}",
                "For {} admissions, you'll need these ranks:\n{}",
                "The competitive cutoff ranks for {} are:\n{}"
            ],
            "general": [
                "Great question! {}",
                "I'd be happy to help with that. {}",
                "Here's what you need to know: {}",
                "Let me provide you with the details: {}",
                "That's a common question! {}"
            ],
            "placements": [
                "NIE has excellent placement records! {}",
                "Placements at NIE are quite impressive: {}",
                "Here's the placement data you're looking for: {}",
                "NIE's placement statistics show: {}"
            ],
            "hostel": [
                "Regarding hostel facilities at NIE: {}",
                "Here's what you need to know about hostels: {}",
                "NIE hostel information: {}",
                "For accommodation at NIE: {}"
            ]
        }

    def get_response_variation(self, answer: str, category: str) -> str:
        """Generate a varied response using templates"""
        try:
            # Determine response type
            response_type = "general"
            if "cutoff" in answer.lower() or "rank" in answer.lower():
                response_type = "cutoff"
            elif "placement" in answer.lower() or "package" in answer.lower():
                response_type = "placements"
            elif "hostel" in answer.lower() or "accommodation" in answer.lower():
                response_type = "hostel"
            
            # Get random template
            templates = self.response_templates.get(response_type, self.response_templates["general"])
            template = random.choice(templates)
            
            # Apply template
            if "{}" in template:
                return template.format(answer)
            else:
                return template + " " + answer
                
        except Exception as e:
            logger.warning(f"Error generating response variation: {e}")
            return answer

    def search_nie_website(self, query: str) -> str:
        """Simple NIE website search fallback"""
        try:
            # Simple fallback responses based on query keywords
            if any(word in query.lower() for word in ['admission', 'cutoff', 'rank']):
                return "For the most current admission information, please visit the official NIE website at https://nie.ac.in/admissions/ or contact the admissions office directly."
            
            elif any(word in query.lower() for word in ['placement', 'package', 'job']):
                return "For detailed placement statistics and company information, check the official NIE website at https://nie.ac.in/placements/ or contact the placement cell."
            
            elif any(word in query.lower() for word in ['hostel', 'accommodation']):
                return "For current hostel information, facilities, and availability, please visit https://nie.ac.in/facilities/ or contact the hostel office."
            
            else:
                return f"For information about '{query}', please visit the official NIE website at https://nie.ac.in/ or contact the relevant department directly."
                
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return "I couldn't find specific information for your query. Please visit the official NIE website at https://nie.ac.in/ for the most current information."

    def get_response(self, user_query: str, user_id: str = "default") -> Tuple[str, float]:
        """Get response using simplified advanced RAG"""
        try:
            # Clean user query
            cleaned_query = self.clean_filler(user_query)
            
            # Special handling for cutoff queries
            if self.is_generic_cutoff_query(cleaned_query):
                return "I'd be happy to help with cutoff information! Could you please specify which exam you're interested in - KCET or COMEDK? This will help me provide you with the most accurate and relevant cutoff data.", 0.8
            
            # Perform semantic search
            query_emb = self.embeddings_model.encode([cleaned_query])
            similarities = cosine_similarity(query_emb, self.embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # High confidence - return FAQ answer with variation
            if best_score >= 0.7:
                answer = self.answers[best_idx]
                category = self.categories[best_idx]
                
                # Generate varied response
                varied_answer = self.get_response_variation(answer, category)
                
                return varied_answer, float(best_score)
            
            # Medium confidence - try fallback
            elif best_score >= 0.4:
                fallback_response = self.search_nie_website(cleaned_query)
                return fallback_response, 0.6
            
            # Low confidence - generic fallback
            else:
                fallback_response = self.search_nie_website(cleaned_query)
                return fallback_response, 0.3
                
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            logger.error(traceback.format_exc())
            return "Sorry, I encountered a technical issue while processing your request. Please try again in a moment.", 0.0

    def clean_filler(self, text: str) -> str:
        """Clean filler words from user input"""
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally']
        cleaned = text.lower()
        for filler in filler_words:
            cleaned = cleaned.replace(filler, '')
        return cleaned.strip()

    def is_generic_cutoff_query(self, query: str) -> bool:
        """Check if query is a generic cutoff question"""
        generic_cutoff_words = ['cutoff', 'cut off', 'cut-offs', 'ranks', 'rank']
        return any(word in query.lower() for word in generic_cutoff_words) and \
               not any(specific in query.lower() for specific in ['kcet', 'comedk', 'cse', 'ece', 'eee', 'me', 'civil'])

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "total_questions": len(self.questions),
            "total_categories": len(self.faq_data),
            "embeddings_shape": self.embeddings.shape,
            "model_name": "all-MiniLM-L6-v2"
        }


# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = SimpleAdvancedRAGChatbot()
    return chatbot

def get_chatbot():
    """Get the chatbot instance"""
    return chatbot or initialize_chatbot()

if __name__ == "__main__":
    # Test the simplified advanced chatbot
    bot = initialize_chatbot()
    
    print("Simple Advanced RAG Chatbot initialized!")
    print(f"Loaded {len(bot.questions)} questions from {len(bot.faq_data)} categories")
    
    # Test queries
    test_queries = [
        "What are the CSE cutoffs?",
        "Tell me about placements",
        "Hostel facilities",
        "What is NIE?"
    ]
    
    for query in test_queries:
        response, confidence = bot.get_response(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response[:100]}...")
        print(f"Confidence: {confidence:.2f}")
