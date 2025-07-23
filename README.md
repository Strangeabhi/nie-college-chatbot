# NIE Mysuru FAQ Chatbot 🤖🎓

An intelligent, AI-powered FAQ chatbot built for The National Institute of Engineering (NIE), Mysuru.
Perfect for answering student queries 24/7 with a beautiful UI, semantic search, and follow-up handling.

> 💡 Originally developed as part of the IIT Chatbot Project.

---

## 🚀 Features

* 🎨 **NIE-branded UI** — modern, mobile-friendly design with campus theme (navy/red/white)
* 🧠 **AI-powered semantic search** — understands intent using MiniLM (SentenceTransformers)
* 💬 **Follow-up friendly** — smart query cleaner handles chained queries like "and hostel?"
* 📝 **Easy FAQ editing** — update `faq_data.json`, restart server — done!
* 📟 **Chat logging** — MongoDB stores all user-bot chats for analytics
* 🌐 **Deployable anywhere** — designed for platforms like Render, Railway, etc.

---

## 🧠 How It Works

1. **Embeds FAQs:** At startup, it encodes all FAQ questions into vector embeddings.
2. **Semantic Match:** User query is cleaned and compared via cosine similarity (> 0.75).
3. **Follow-up Support:** Recognizes and handles chained or partial follow-up queries.
4. **Logs Chats:** Every interaction is stored in MongoDB (`database.py`) for future insights.

---

## 🛠️ Setup & Local Testing

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

| File / Folder      | Purpose                            |
| ------------------ | ---------------------------------- |
| `api.py`           | Flask backend with REST API        |
| `chatbot_rag.py`   | Core AI chatbot logic using MiniLM |
| `database.py`      | MongoDB connection and logging     |
| `faq_data.json`    | Editable FAQ question-answer pairs |
| `index.html`       | Responsive NIE-themed frontend     |
| `requirements.txt` | Python dependencies                |

---

## 🧹 FAQ Data Format

All questions and answers are stored in:

```json
[
  {
    "question": "What are the hostel facilities at NIE?",
    "answer": "NIE provides separate hostel facilities for boys and girls..."
  }
]
```

Want to update the bot’s answers? Just edit this file and restart the server.

---

## 📱 Mobile Friendly

Built with responsive HTML/CSS. Works great on desktops, tablets, and phones.

---

## ✅ Requirements

* Python 3.7+
* MongoDB (local or cloud via Atlas)
* See `requirements.txt` for dependencies

---

## 🙌 Credits

* Developed by **Abhishek C Patil** (NIE Mysuru, 2027)
* Created using publicly available FAQ data and the official NIE website
* UI theme inspired by [nie.ac.in](https://nie.ac.in)
* Built as part of the IIT Chatbot Challenge

---

Have fun chatting, and feel free to build on top of this! 🚠
