"""
Advanced API for NIE Chatbot with LangChain, Memory, and MLOps
Upgraded version of the original api.py
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import traceback
import json
import time
from datetime import datetime
import logging

# Import our advanced components
from advanced_chatbot_rag import get_chatbot
from mlops_monitor import get_monitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global instances
chatbot = None
monitor = None

def initialize_services():
    """Initialize chatbot and monitoring services"""
    global chatbot, monitor
    
    if chatbot is None:
        chatbot = get_chatbot()
        logger.info("Advanced chatbot initialized")
    
    if monitor is None:
        monitor = get_monitor()
        logger.info("MLOps monitor initialized")

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
    return render_template_string(ENHANCED_HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with MLOps tracking"""
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
        
        # Get response from advanced chatbot
        bot_response, confidence = chatbot.get_response(user_message, user_id)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log interaction for MLOps
        monitor.log_interaction(
            user_query=user_message,
            bot_response=bot_response,
            confidence=confidence,
            response_time=response_time,
            user_id=user_id
        )
        
        # Enhanced logging with MLOps data
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

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Endpoint for user feedback collection"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        bot_response = data.get('response', '')
        feedback_score = data.get('score', 0)  # 1-5 scale
        user_id = data.get('user_id', 'anonymous')
        
        # Validate feedback score
        if not (1 <= feedback_score <= 5):
            return jsonify({
                "message": "Feedback score must be between 1 and 5",
                "error": "invalid_score"
            }), 400
        
        # Log feedback for MLOps
        monitor.log_feedback(
            user_query=user_query,
            bot_response=bot_response,
            feedback_score=feedback_score,
            user_id=user_id
        )
        
        logger.info(f"Feedback received: {feedback_score}/5 for query: {user_query[:50]}...")
        
        return jsonify({
            "message": "Thank you for your feedback!",
            "status": "success"
        })
        
    except Exception as e:
        logger.error('--- Exception in /api/feedback endpoint ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Failed to process feedback",
            "error": "feedback_error"
        }), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Get chatbot analytics and performance metrics"""
    try:
        analytics_report = monitor.get_analytics_report()
        
        return jsonify({
            "analytics": analytics_report,
            "status": "success"
        })
        
    except Exception as e:
        logger.error('--- Exception in /api/analytics endpoint ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Failed to retrieve analytics",
            "error": "analytics_error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        chatbot_metrics = chatbot.get_performance_metrics()
        monitor_metrics = monitor.get_performance_metrics()
        
        return jsonify({
            "status": "healthy",
            "chatbot": chatbot_metrics,
            "monitor": monitor_metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error('--- Exception in health check ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Manually trigger model retraining"""
    try:
        success = monitor.trigger_retrain()
        
        if success:
            return jsonify({
                "message": "Model retraining completed successfully",
                "status": "success"
            })
        else:
            return jsonify({
                "message": "Model retraining failed",
                "status": "error"
            }), 500
            
    except Exception as e:
        logger.error('--- Exception in model retraining ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Failed to retrain model",
            "error": "retrain_error"
        }), 500

@app.route('/api/export', methods=['GET'])
def export_analytics():
    """Export analytics data"""
    try:
        filename = monitor.export_analytics()
        
        return jsonify({
            "message": f"Analytics exported to {filename}",
            "filename": filename,
            "status": "success"
        })
        
    except Exception as e:
        logger.error('--- Exception in analytics export ---')
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "Failed to export analytics",
            "error": "export_error"
        }), 500

# Enhanced HTML template with feedback system and analytics
ENHANCED_HTML_TEMPLATE = """
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
        
        .feedback-buttons {
            margin-top: 8px;
            display: flex;
            gap: 5px;
        }
        
        .feedback-btn {
            background: #f0f0f0;
            border: none;
            padding: 4px 8px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .feedback-btn:hover {
            background: #e0e0e0;
        }
        
        .feedback-btn.selected {
            background: #4CAF50;
            color: white;
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
        
        .analytics-toggle {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .analytics-panel {
            position: absolute;
            top: 0;
            left: -300px;
            width: 300px;
            height: 100%;
            background: white;
            border-right: 1px solid #e1e5e9;
            transition: left 0.3s;
            z-index: 1000;
            overflow-y: auto;
            padding: 20px;
        }
        
        .analytics-panel.open {
            left: 0;
        }
        
        .analytics-content h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .metric-item {
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .metric-label {
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-value {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <button class="analytics-toggle" onclick="toggleAnalytics()">📊 Analytics</button>
            <h1>NIE Advanced Chatbot</h1>
            <p>Powered by LangChain & MLOps</p>
            <div class="status-indicator" title="System Status: Online"></div>
        </div>
        
        <div class="analytics-panel" id="analyticsPanel">
            <div class="analytics-content">
                <h3>Performance Metrics</h3>
                <div id="analyticsContent">
                    <p>Loading analytics...</p>
                </div>
            </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm the NIE Advanced Chatbot with enhanced capabilities. I can help you with admissions, placements, hostels, courses, and more. Ask me anything about NIE!
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
        
        function addMessage(message, isUser = false, confidence = null, messageId = null) {
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
                
                // Add feedback buttons for bot messages
                const feedbackDiv = document.createElement('div');
                feedbackDiv.className = 'feedback-buttons';
                feedbackDiv.innerHTML = `
                    <button class="feedback-btn" onclick="submitFeedback(${messageId}, 5)">👍</button>
                    <button class="feedback-btn" onclick="submitFeedback(${messageId}, 4)">😊</button>
                    <button class="feedback-btn" onclick="submitFeedback(${messageId}, 3)">😐</button>
                    <button class="feedback-btn" onclick="submitFeedback(${messageId}, 2)">😞</button>
                    <button class="feedback-btn" onclick="submitFeedback(${messageId}, 1)">👎</button>
                `;
                
                contentDiv.appendChild(metaDiv);
                contentDiv.appendChild(feedbackDiv);
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
                    const messageId = Date.now();
                    addMessage(data.response, false, data.confidence, messageId);
                    currentConversation.push({
                        role: 'bot', 
                        content: data.response,
                        confidence: data.confidence,
                        messageId: messageId
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
        
        async function submitFeedback(messageId, score) {
            try {
                // Find the message in conversation
                const message = currentConversation.find(msg => msg.messageId === messageId);
                if (!message) return;
                
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: currentConversation[currentConversation.indexOf(message) - 1]?.content || '',
                        response: message.content,
                        score: score,
                        user_id: currentUserId
                    })
                });
                
                // Visual feedback
                const buttons = document.querySelectorAll(`button[onclick="submitFeedback(${messageId}, "]`);
                buttons.forEach(btn => btn.classList.remove('selected'));
                event.target.classList.add('selected');
                
            } catch (error) {
                console.error('Feedback error:', error);
            }
        }
        
        function toggleAnalytics() {
            const panel = document.getElementById('analyticsPanel');
            panel.classList.toggle('open');
            
            if (panel.classList.contains('open')) {
                loadAnalytics();
            }
        }
        
        async function loadAnalytics() {
            try {
                const response = await fetch('/api/analytics');
                const data = await response.json();
                
                if (response.ok) {
                    const metrics = data.analytics.performance_metrics;
                    const content = `
                        <div class="metric-item">
                            <div class="metric-label">Total Interactions</div>
                            <div class="metric-value">${metrics.total_interactions}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Average Confidence</div>
                            <div class="metric-value">${(metrics.avg_confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Success Rate</div>
                            <div class="metric-value">${(metrics.success_rate * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Response Time</div>
                            <div class="metric-value">${metrics.avg_response_time.toFixed(2)}s</div>
                        </div>
                    `;
                    
                    document.getElementById('analyticsContent').innerHTML = content;
                }
            } catch (error) {
                console.error('Analytics error:', error);
                document.getElementById('analyticsContent').innerHTML = '<p>Failed to load analytics</p>';
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
    
    print("🚀 Starting NIE Advanced Chatbot API...")
    print("📊 MLOps monitoring enabled")
    print("🔗 LangChain integration active")
    print("💬 Enhanced UI with feedback system")
    print("📈 Real-time analytics available")
    print("\nAccess the chatbot at: http://localhost:5000")
    print("Analytics dashboard at: http://localhost:5000/api/analytics")
    print("Health check at: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
