from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_rag import chatbot
from database import log_chat
import traceback

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error with traceback
    print('--- Exception in API ---')
    print(traceback.format_exc())
    return jsonify({'response': 'Sorry, something went wrong. Please try again later.'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'response': 'Please provide a message.'}), 400
        bot_response, similarity = chatbot.get_response(user_message)
        log_chat(user_message, bot_response, similarity)
        # Debug prints
        print(f"User Question: {user_message}")
        print(f"Selected Answer: {bot_response}")
        print(f"Confidence Score: {similarity:.2f}")
        if similarity < 0.7:
            print("⚠️ Low confidence, maybe fallback needed!")
        return jsonify({'response': bot_response})
    except Exception as e:
        print('--- Exception in /api/chat endpoint ---')
        print(traceback.format_exc())
        return jsonify({'response': 'Sorry, I could not process your request due to a technical issue.'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
