"""
Simplified Advanced API for NIE Chatbot
Uses the simplified advanced chatbot without complex dependencies
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import traceback
import json
import time
from datetime import datetime
import logging

# Import our simplified advanced components
from simple_advanced_chatbot import get_chatbot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global instances
chatbot = None

def initialize_services():
    """Initialize chatbot services"""
    global chatbot
    
    if chatbot is None:
        chatbot = get_chatbot()
        logger.info("Simple advanced chatbot initialized")

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error('--- Exception in API ---')
    logger.error(traceback.format_exc())
    return jsonify({
        "response": "Sorry, something went wrong. Please try again later.",
        "error": "internal_server_error"
    }), 500

@app.route('/')
def index():
    """Serve the enhanced chatbot interface"""
    return render_template_string(SIMPLE_HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({
                "response": "Please provide a message.",
                "error": "empty_message"
            }), 400
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Get response from simplified advanced chatbot
        bot_response, confidence = chatbot.get_response(user_message, user_id)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Enhanced logging
        logger.info(f"User Question: {user_message}")
        logger.info(f"Selected Answer: {bot_response[:100]}...")
        logger.info(f"Confidence Score: {confidence:.2f}")
        logger.info(f"Response Time: {response_time:.2f}s")
        
        # Warning for low confidence and fallback usage
        if confidence < 0.7:
            if confidence >= 0.4:
                logger.warning("⚠️ Using external fallback (NIE website)")
            else:
                logger.warning("⚠️ Low confidence, using local fallback")
        
        # Return response with metadata
        return jsonify({
            'response': bot_response,
            'confidence': confidence,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error('--- Exception in /api/chat endpoint ---')
        logger.error(traceback.format_exc())
        return jsonify({
            'response': 'Sorry, I could not process your request due to a technical issue.',
            'error': 'processing_error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        chatbot_metrics = chatbot.get_performance_metrics()
        
        return jsonify({
            "status": "healthy",
            "chatbot": chatbot_metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error('--- Exception in health check ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Simplified HTML template
SIMPLE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIE Advanced Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.4;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e1e5e9;
        }
        
        .message-meta {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e5e9;
        }
        
        .chat-input-wrapper {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            color: #666;
            font-style: italic;
        }
        
        .confidence-indicator {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .confidence-high { color: #4CAF50; }
        .confidence-medium { color: #FF9800; }
        .confidence-low { color: #F44336; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>NIE Advanced Chatbot</h1>
            <p>Powered by Advanced RAG with Dynamic Responses</p>
            <div class="status-indicator" title="System Status: Online"></div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm the NIE Advanced Chatbot with enhanced capabilities. I can help you with admissions, placements, hostels, courses, and more. Ask me anything about NIE!<br><br>
                    <strong>New Features:</strong><br>
                    • 🔄 Dynamic responses - different answers every time<br>
                    • 🌐 External fallback to NIE website<br>
                    • 🎯 Smart query understanding<br>
                    • ⚡ Fast performance with caching
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            Bot is typing...
        </div>
        
        <div class="chat-input-container">
            <div class="chat-input-wrapper">
                <input type="text" id="chatInput" class="chat-input" placeholder="Ask me anything about NIE..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" class="send-button" id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        let currentConversation = [];
        let currentUserId = 'user_' + Math.random().toString(36).substr(2, 9);
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function addMessage(message, isUser = false, confidence = null) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // Convert line breaks to HTML
            contentDiv.innerHTML = message.replace(/\\n/g, '<br>');
            
            if (!isUser && confidence !== null) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                
                const confidenceClass = confidence >= 0.8 ? 'confidence-high' : 
                                      confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
                
                metaDiv.innerHTML = `
                    <div class="confidence-indicator ${confidenceClass}">
                        Confidence: ${(confidence * 100).toFixed(1)}%
                    </div>
                `;
                
                contentDiv.appendChild(metaDiv);
            }
            
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return messageDiv;
        }
        
        function showTyping() {
            document.getElementById('typingIndicator').style.display = 'block';
        }
        
        function hideTyping() {
            document.getElementById('typingIndicator').style.display = 'none';
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const sendButton = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Disable input and show typing
            input.disabled = true;
            sendButton.disabled = true;
            showTyping();
            
            // Add user message
            addMessage(message, true);
            currentConversation.push({role: 'user', content: message});
            
            // Clear input
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        user_id: currentUserId
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response, false, data.confidence);
                    currentConversation.push({
                        role: 'bot', 
                        content: data.response,
                        confidence: data.confidence
                    });
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, I encountered a network error. Please check your connection and try again.', false);
            } finally {
                // Re-enable input
                input.disabled = false;
                sendButton.disabled = false;
                hideTyping();
                input.focus();
            }
        }
        
        // Focus input on load
        document.getElementById('chatInput').focus();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    # Initialize services
    initialize_services()
    
    print("🚀 Starting NIE Simple Advanced Chatbot API...")
    print("💬 Enhanced UI with dynamic responses")
    print("🌐 External fallback to NIE website")
    print("⚡ Fast performance with caching")
    print("\nAccess the chatbot at: http://localhost:5000")
    print("Health check at: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
