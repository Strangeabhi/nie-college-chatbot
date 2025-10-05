# NIE Mysuru Advanced FAQ Chatbot 🤖🎓

An intelligent, AI-powered FAQ chatbot built for The National Institute of Engineering (NIE), Mysuru.
Now featuring **LangChain integration**, **MLOps monitoring**, **dynamic response variations**, and **advanced RAG capabilities** with comprehensive coverage of 427+ questions across all aspects of college life.

> 💡 Originally developed as part of the IIT Chatbot Project. Now upgraded with enterprise-grade features!

---

## 🚀 Features

### 🆕 **Advanced Features (NEW!)**
* 🔗 **LangChain Integration** — enterprise-grade RAG with conversation memory and advanced retrieval
* 🔄 **Dynamic Response Variations** — different answers for the same question every time
* 📊 **MLOps Monitoring** — real-time performance tracking, auto-retraining, and analytics
* 🎯 **Hybrid Search** — combines semantic and keyword matching for better accuracy
* 💾 **Conversation Memory** — remembers context across multiple questions
* 📈 **User Feedback System** — collects and learns from user satisfaction scores
* 🚨 **Auto-Retraining** — automatically improves when performance drops
* 📱 **Enhanced UI** — modern interface with analytics dashboard and feedback buttons

### 🔥 **Core Features**
* 🎨 **NIE-branded UI** — modern, mobile-friendly design with campus theme (navy/red/white)
* 🧠 **AI-powered semantic search** — understands intent using MiniLM (SentenceTransformers)
* 💬 **Follow-up friendly** — smart query cleaner handles chained queries like "and hostel?"
* 🎯 **Smart cutoff handling** — intelligent responses for specific branch cutoffs (CSE, ECE, etc.)
* 🛡️ **Robust error handling** — graceful fallbacks and user-friendly error messages
* 📝 **Easy FAQ editing** — update `faq_data.json`, restart server — done!
* 📟 **Chat logging** — MongoDB stores all user-bot chats for analytics
* 🌐 **Deployable anywhere** — designed for platforms like Render, Railway, etc.
* 📊 **Professional formatting** — line breaks and structured responses for better readability

---

## 🧠 How It Works

### 🆕 **Advanced Architecture (NEW!)**
1. **LangChain RAG:** Uses FAISS vector store with conversation memory for context-aware responses
2. **Hybrid Search:** Combines semantic similarity with keyword matching for better retrieval
3. **Response Variations:** Generates different answers using dynamic templates and context
4. **MLOps Pipeline:** Continuously monitors performance, collects feedback, and auto-retrains
5. **Memory System:** Remembers conversation history for follow-up questions
6. **Feedback Loop:** Learns from user ratings to improve future responses

### 🔥 **Core Process**
1. **Embeds FAQs:** At startup, it encodes all FAQ questions into vector embeddings.
2. **Semantic Match:** User query is cleaned and compared via cosine similarity (> 0.75).
3. **Smart Routing:** Handles specific queries (cutoffs, fees, documents) with targeted responses.
4. **Follow-up Support:** Recognizes and handles chained or partial follow-up queries.
5. **Error Handling:** Graceful fallbacks with user-friendly messages for unexpected issues.
6. **Logs Chats:** Every interaction is stored in MongoDB (`database.py`) for future insights.

---

## 📚 Data Coverage

The chatbot covers **427+ questions** across comprehensive categories:

* **General Information** — college overview, contact, rankings, accreditation
* **Branches & Courses** — CSE, CI (AI & ML), ISE, ECE, EEE, Mechanical, Civil
* **Admissions & Eligibility** — KCET, COMEDK, management quota, required documents
* **Cutoffs & Seats** — detailed seat matrix, quota-wise cutoffs with college codes
* **Fees & Financial Aid** — quota-wise fees, scholarships, payment details
* **Hostels & Accommodation** — boys/girls hostels, facilities, transportation
* **Placements & Career** — packages, recruiters, branch-wise statistics
* **Campus Facilities** — library, canteen, sports, ALPHA Lab, innovation centers
* **Transport & Connectivity** — bus routes, distances, parking
* **Clubs & Fests** — IEEE, TECHNIEKS, cultural events, competitions
* **Student Life** — demographics, attendance, dress code, academic calendar
* **Scholarships & Financial Aid** — various scholarship schemes and eligibility
* **Anti-Ragging & Safety** — campus safety, hostel security, reporting procedures

---

## 🛠️ Setup & Local Testing

### 🆕 **Advanced Setup (Recommended)**
1. Install enhanced dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Start the complete advanced system:

   ```bash
   python start_advanced_chatbot.py
   ```
3. Access the enhanced chatbot at: http://localhost:5000
4. Check analytics dashboard for performance metrics
5. Enjoy dynamic responses and conversation memory!

### 🔥 **Basic Setup (Original)**
1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Start MongoDB (locally or Atlas)
3. Run Flask API:

   ```bash
   python api.py
   ```
4. Open `index.html` in a browser and chat away!

### 🧪 **Testing Commands**
```bash
# Test system components
python start_advanced_chatbot.py test

# Check system status
python start_advanced_chatbot.py status

# Run MLOps monitor only
python mlops_monitor.py

# Test accuracy
python test_accuracy.py
```

---

## 🌍 Deploy on Render (or any other cloud)

1. Push the project to GitHub.
2. Go to [render.com](https://render.com/) and create a new Web Service.
3. Set:

   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python api.py`
4. In `api.py`, make sure this exists:

   ```python
   import os
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port)
   ```
5. Serve frontend via `/static` or configure Flask routes for static files.
6. Share your public Render URL — no installation needed by users!

---

## 📁 File Structure

### 🆕 **Advanced Files (NEW!)**
| File / Folder                | Purpose                                    |
| ---------------------------- | ------------------------------------------ |
| `advanced_api.py`            | Enhanced Flask API with LangChain & MLOps |
| `advanced_chatbot_rag.py`    | LangChain-powered RAG with memory         |
| `mlops_monitor.py`           | MLOps monitoring and analytics system     |
| `start_advanced_chatbot.py`  | Complete system startup and management    |

### 🔥 **Core Files**
| File / Folder      | Purpose                            |
| ------------------ | ---------------------------------- |
| `api.py`           | Flask backend with REST API        |
| `chatbot_rag.py`   | Core AI chatbot logic using MiniLM |
| `database.py`      | MongoDB connection and logging     |
| `faq_data.json`    | Editable FAQ question-answer pairs |
| `index.html`       | Responsive NIE-themed frontend     |
| `requirements.txt` | Python dependencies                |
| `test_accuracy.py` | Accuracy testing and evaluation    |

---

## 🧹 FAQ Data Format

All questions and answers are stored in a nested category structure:

```json
[
  {
    "category": "Branches and Courses",
    "questions": [
      {
        "question": "What branches are offered at NIE?",
        "answer": "NIE offers undergraduate programs in:\n- Computer Science Engineering (CSE)\n- Computer Science & Engineering – AI & ML (CI)\n- Information Science Engineering (ISE)\n- Electronics and Communication Engineering (ECE)\n- Electrical and Electronics Engineering (EEE)\n- Mechanical Engineering\n- Civil Engineering."
      }
    ]
  }
]
```

**Key Features:**
- **Nested categories** for better organization
- **Line breaks** (`\n`) for readable formatting
- **427+ questions** with comprehensive coverage
- **Smart responses** for specific query types

Want to update the bot's answers? Just edit this file and restart the server.

---

## 🎯 Smart Features

### 🆕 **Advanced Intelligence (NEW!)**
* **Dynamic Responses:** Different answers for the same question every time you ask
* **Conversation Memory:** "Tell me more about CSE" works perfectly after asking about CSE
* **Hybrid Search:** Combines AI understanding with keyword matching for better results
* **MLOps Analytics:** Real-time performance monitoring and automatic improvements
* **User Feedback Learning:** Gets smarter based on user satisfaction ratings
* **Auto-Retraining:** Automatically updates when performance drops or FAQ changes

### 🔥 **Core Intelligence**
* **Cutoff Intelligence:** Ask "CSE cutoff" and get comprehensive KCET + COMEDK data
* **Branch-Specific Responses:** Detailed answers for each engineering branch
* **Quota-Wise Information:** Separate data for aided, unaided, management, COMEDK
* **Professional Formatting:** Clean, readable responses with proper line breaks
* **Comprehensive Coverage:** From admission to placement, hostels to safety
* **Error Resilience:** Backend exception handling prevents crashes and provides helpful fallback messages

---

## 🆕 Advanced Features Deep Dive

### 🔗 **LangChain Integration**
- **FAISS Vector Store:** Faster and more accurate similarity search
- **Conversation Memory:** Maintains context across multiple questions
- **Advanced Retrieval:** Reranking and filtering for better results
- **Template System:** Dynamic response generation with variations

### 📊 **MLOps Pipeline**
- **Real-time Monitoring:** Tracks confidence, response time, and accuracy
- **Performance Analytics:** Detailed metrics and insights dashboard
- **Auto-Retraining:** Automatically improves when performance drops
- **Feedback Collection:** User rating system for continuous learning
- **Export Capabilities:** Analytics data export for further analysis

### 🎯 **Response Variations**
The chatbot now provides different answers for the same question:
```
Question: "What are CSE cutoffs?"
Answer 1: "KCET cutoffs for CSE at NIE are quite competitive! For the last available year..."
Answer 2: "CSE cutoffs at NIE vary by exam type: KCET: 8726 rank, COMEDK: 10182 rank..."
Answer 3: "Great question! CSE is one of NIE's most sought-after branches. Cutoff ranks..."
```

### 💾 **Memory System**
- Remembers conversation context
- Follow-up questions work naturally
- Personalized responses based on history
- Session-based user tracking

## 📱 Mobile Friendly

Built with responsive HTML/CSS. Works great on desktops, tablets, and phones. The advanced UI includes analytics dashboard and feedback buttons for better user experience.

---

## ✅ Requirements

### 🆕 **Advanced Requirements**
* Python 3.8+
* LangChain & FAISS for advanced RAG
* MLflow for MLOps tracking
* Prometheus for metrics collection
* Rich for enhanced CLI experience

### 🔥 **Core Requirements**
* Python 3.7+
* MongoDB (local or cloud via Atlas)
* See `requirements.txt` for all dependencies

---

## 🙌 Credits

* Developed by **Abhishek C Patil** (NIE Mysuru, 2027)
* Created using publicly available FAQ data and the official NIE website
* UI theme inspired by [nie.ac.in](https://nie.ac.in)
* Built as part of the IIT Chatbot Challenge

---

Have fun chatting, and feel free to build on top of this! 🚠
