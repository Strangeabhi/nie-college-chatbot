from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_rag import chatbot
from database import log_chat

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please provide a message.'}), 400
    bot_response, similarity = chatbot.get_response(user_message)
    log_chat(user_message, bot_response, similarity)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
