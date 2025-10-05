"""
Advanced RAG Chatbot with LangChain, Memory, and MLOps
Upgraded version of the original chatbot_rag.py
"""

import json
import numpy as np
import traceback
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# LangChain imports (updated for newer versions)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

# MLOps imports
import mlflow
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil

# Fallback system import
from fallback_system import get_fallback_response

# Performance cache import
from performance_cache import get_cache, cached_query_response, cached_fallback_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('chatbot_requests_total', 'Total number of requests')
RESPONSE_TIME = Histogram('chatbot_response_time_seconds', 'Response time in seconds')
CONFIDENCE_SCORE = Histogram('chatbot_confidence_score', 'Confidence score distribution')
ACTIVE_CONVERSATIONS = Gauge('chatbot_active_conversations', 'Number of active conversations')

class AdvancedRAGChatbot:
    def __init__(self, faq_file='faq_data.json', embeddings_file='faq_embeddings.npy'):
        """Initialize the advanced RAG chatbot with LangChain"""
        self.faq_file = faq_file
        self.embeddings_file = embeddings_file
        
        # Load FAQ data
        self.load_faq_data()
        
        # Initialize embeddings and vector store
        self.setup_embeddings()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize MLOps tracking
        self.setup_mlops()
        
        # Initialize performance cache
        self.cache = get_cache()
        
        # Response variations for dynamic answers
        self.response_templates = self.load_response_templates()
        
        logger.info("Advanced RAG Chatbot initialized successfully")

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
        """Setup embeddings and vector store"""
        try:
            # Initialize embeddings
            self.embeddings_model = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Try to load existing embeddings
            try:
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded existing embeddings from {self.embeddings_file}")
            except FileNotFoundError:
                logger.info("Generating new embeddings...")
                self.generate_embeddings()
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_texts(
                texts=self.questions,
                embedding=self.embeddings_model
            )
            
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise

    def generate_embeddings(self):
        """Generate embeddings for FAQ questions"""
        try:
            logger.info("Generating embeddings for FAQ questions...")
            self.embeddings = self.embeddings_model.embed_documents(self.questions)
            self.embeddings = np.array(self.embeddings)
            
            # Save embeddings
            np.save(self.embeddings_file, self.embeddings)
            logger.info(f"Embeddings saved to {self.embeddings_file}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def setup_mlops(self):
        """Setup MLOps tracking"""
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("nie_chatbot")
            
            # Start Prometheus metrics server
            start_http_server(8000)
            
            logger.info("MLOps tracking initialized")
            
        except Exception as e:
            logger.warning(f"MLOps setup failed: {e}")

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

    @cached_query_response(get_cache())
    def get_response(self, user_query: str, user_id: str = "default") -> Tuple[str, float]:
        """
        Get response using advanced RAG with memory and variations
        """
        start_time = datetime.now()
        
        try:
            # Update metrics
            REQUEST_COUNT.inc()
            ACTIVE_CONVERSATIONS.inc()
            
            # Clean user query
            cleaned_query = self.clean_filler(user_query)
            
            # Special handling for cutoff queries
            if self.is_generic_cutoff_query(cleaned_query):
                return "I'd be happy to help with cutoff information! Could you please specify which exam you're interested in - KCET or COMEDK? This will help me provide you with the most accurate and relevant cutoff data.", 0.8
            
            # Perform hybrid search
            search_results = self.hybrid_search(cleaned_query, k=3)
            
            if not search_results:
                return "Sorry, I didn't find relevant information for your query. Could you rephrase your question or try asking about admissions, placements, hostels, or other NIE-related topics?", 0.0
            
            # Get best match
            best_match = search_results[0]
            confidence = best_match['score']
            
            # Update confidence metrics
            CONFIDENCE_SCORE.observe(confidence)
            
            if confidence >= 0.7:
                # Get the answer and apply variation
                answer_index = best_match['index']
                base_answer = self.answers[answer_index]
                category = self.categories[answer_index]
                
                # Generate varied response
                varied_answer = self.get_response_variation(base_answer, category)
                
                # Log to MLflow
                self.log_interaction(user_query, varied_answer, confidence, user_id)
                
                return varied_answer, float(confidence)
            else:
                # Low confidence - try external fallback first
                logger.info(f"Low confidence ({confidence:.2f}), trying external fallback...")
                try:
                    # Check cache first
                    cached_fallback = self.cache.get_cached_fallback(cleaned_query)
                    if cached_fallback:
                        logger.info("Using cached fallback response")
                        return cached_fallback
                    
                    fallback_response, fallback_confidence = get_fallback_response(cleaned_query)
                    if fallback_confidence > 0.4:  # If fallback has decent confidence
                        # Cache the fallback response
                        self.cache.cache_fallback_response(cleaned_query, fallback_response, fallback_confidence)
                        return fallback_response, float(fallback_confidence)
                except Exception as e:
                    logger.warning(f"External fallback failed: {e}")
                
                # If external fallback fails, use intelligent local fallback
                fallback_response = self.get_intelligent_fallback(cleaned_query, search_results)
                return fallback_response, float(confidence)
                
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            logger.error(traceback.format_exc())
            return "Sorry, I encountered a technical issue while processing your request. Please try again in a moment.", 0.0
        
        finally:
            # Update response time metrics
            response_time = (datetime.now() - start_time).total_seconds()
            RESPONSE_TIME.observe(response_time)
            ACTIVE_CONVERSATIONS.dec()

    def hybrid_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword matching"""
        try:
            # Semantic search using FAISS
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
            
            # Convert to our format
            results = []
            for i, (doc, score) in enumerate(semantic_results):
                # Find the index of this question in our list
                try:
                    question_index = self.questions.index(doc.page_content)
                    results.append({
                        'question': doc.page_content,
                        'answer': self.answers[question_index],
                        'category': self.categories[question_index],
                        'score': 1 - score,  # Convert distance to similarity
                        'index': question_index
                    })
                except ValueError:
                    continue
            
            # Sort by score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid_search: {e}")
            return []

    def get_intelligent_fallback(self, query: str, search_results: List[Dict]) -> str:
        """Generate intelligent fallback responses"""
        try:
            if search_results:
                best_result = search_results[0]
                if best_result['score'] > 0.5:
                    return f"I found some related information, but I'm not completely sure if this answers your question. {best_result['answer']}\n\nIf this doesn't help, could you rephrase your question?"
            
            # Analyze query for suggestions
            suggestions = []
            if any(word in query.lower() for word in ['cutoff', 'rank', 'admission']):
                suggestions.append("admission cutoffs")
            if any(word in query.lower() for word in ['placement', 'package', 'job']):
                suggestions.append("placement information")
            if any(word in query.lower() for word in ['hostel', 'accommodation', 'room']):
                suggestions.append("hostel facilities")
            
            if suggestions:
                return f"I'm not sure about your specific question, but you might be interested in {', '.join(suggestions)}. Could you ask more specifically about any of these topics?"
            
            return "I'm sorry, I couldn't find relevant information for your query. You can ask me about NIE admissions, placements, hostels, courses, facilities, or any other college-related topics!"
            
        except Exception as e:
            logger.error(f"Error in get_intelligent_fallback: {e}")
            return "Sorry, I couldn't process your request. Please try asking about NIE admissions, placements, or other college topics."

    def log_interaction(self, query: str, answer: str, confidence: float, user_id: str):
        """Log interaction to MLflow for MLOps"""
        try:
            with mlflow.start_run():
                mlflow.log_param("user_query", query)
                mlflow.log_param("user_id", user_id)
                mlflow.log_metric("confidence_score", confidence)
                mlflow.log_metric("response_length", len(answer))
                
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")

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

    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user"""
        # This would be implemented with a proper database
        # For now, return empty list
        return []

    def retrain_model(self):
        """Retrain the model when FAQ data changes"""
        try:
            logger.info("Starting model retraining...")
            
            # Reload FAQ data
            self.load_faq_data()
            
            # Regenerate embeddings
            self.generate_embeddings()
            
            # Recreate vector store
            self.setup_embeddings()
            
            logger.info("Model retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "total_questions": len(self.questions),
            "total_categories": len(self.faq_data),
            "embeddings_shape": self.embeddings.shape,
            "active_conversations": ACTIVE_CONVERSATIONS._value._value,
            "total_requests": REQUEST_COUNT._value._value
        }


# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = AdvancedRAGChatbot()
    return chatbot

def get_chatbot():
    """Get the chatbot instance"""
    return chatbot or initialize_chatbot()

if __name__ == "__main__":
    # Test the advanced chatbot
    bot = initialize_chatbot()
    
    print("Advanced RAG Chatbot initialized!")
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
        print(f"Response: {response}")
        print(f"Confidence: {confidence:.2f}")
