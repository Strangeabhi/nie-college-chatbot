import os
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'chatbot_db'
COLLECTION_NAME = 'chat_logs'

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
chat_logs = db[COLLECTION_NAME]

def log_chat(user_message, bot_response, similarity=None):
    entry = {
        'user_message': user_message,
        'bot_response': bot_response,
        'similarity': similarity,
        'timestamp': datetime.utcnow()
    }
    chat_logs.insert_one(entry) 