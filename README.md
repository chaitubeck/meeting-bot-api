# 🧠 Meeting Insight – Semantic Search App

A full-stack internal tool built with **FastAPI**, **OpenAI**, **FAISS**, **MongoDB**, and a **React frontend** to semantically search across Webex meeting summaries.

---

## ⚙️ Backend Setup (Python + FastAPI)

### ✅ Prerequisites
- Python 3.10+ installed
- MongoDB running locally (`brew services start mongodb-community`)
- OpenAI API key saved in a file called `gpt-key.txt`

### 📦 Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate

pip install fastapi uvicorn pymongo openai faiss-cpu numpy python-dotenv
