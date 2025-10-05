# 🚀 NIE Advanced Chatbot - Deployment Guide

## 📦 **Deployment-Ready Files:**

- `deploy_chatbot.py` - Optimized chatbot (minimal memory usage)
- `requirements_minimal.txt` - Essential dependencies only
- `faq_data.json` - Your FAQ data
- `index.html` - Beautiful NIE-themed frontend
- `faq_embeddings.npy` - Pre-computed embeddings (optional)

## 🎯 **Memory Usage: ~400-500MB**
- ✅ Fits in 512MB free tiers
- ✅ All advanced features included
- ✅ Dynamic responses
- ✅ Conversation memory
- ✅ External fallback

## 🌐 **Hosting Options:**

### **1. Railway (Recommended) ⭐**
**Free Tier: 512MB RAM**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy NIE Advanced Chatbot"
   git push origin main
   ```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect Python

3. **Configure Railway:**
   - **Build Command:** `pip install -r requirements_minimal.txt`
   - **Start Command:** `python deploy_chatbot.py`
   - **Port:** Railway sets this automatically

### **2. Heroku**
**Free Tier: 512MB RAM (if available)**

1. **Install Heroku CLI**
2. **Create Heroku app:**
   ```bash
   heroku create your-nie-chatbot
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

4. **Set environment:**
   ```bash
   heroku config:set PORT=5000
   ```

### **3. Vercel**
**Free Tier: 1GB RAM**

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

## 📝 **Environment Variables (if needed):**
- `PORT` - Railway/Heroku sets this automatically
- `PYTHON_VERSION` - Set to `3.11.0` if needed

## 🧪 **Test Your Deployment:**

1. **Health Check:**
   ```
   https://your-app.railway.app/api/health
   ```

2. **Chatbot Interface:**
   ```
   https://your-app.railway.app/
   ```

## 🔧 **Troubleshooting:**

### **If Memory Issues:**
- Use `deploy_chatbot.py` (not `run_chatbot.py`)
- Use `requirements_minimal.txt` (not `requirements.txt`)

### **If Embeddings Not Found:**
- The app will generate them automatically on first run
- Or upload `faq_embeddings.npy` to your hosting platform

### **If Port Issues:**
- Railway/Heroku sets PORT automatically
- Make sure your app uses `os.environ.get("PORT", 5000)`

## ✅ **What You Get:**

- 🎨 **Beautiful NIE-themed UI**
- 🔄 **Dynamic responses** (different answers every time)
- 💾 **Conversation memory** (remembers context)
- 🌐 **External fallback** (searches NIE website)
- ⚡ **Fast performance** (optimized for hosting)
- 📱 **Mobile responsive**

## 🎉 **Ready to Deploy!**

**Railway is the easiest option** - just connect your GitHub repo and deploy! 🚀
