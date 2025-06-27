

```md
# ⚽ Football AI Service

An end-to-end AI-driven FastAPI backend for football performance benchmarking using pose estimation, CLIP-based similarity, and ball tracking. It compares player practice videos against coach benchmarks and logs results in a PostgreSQL database.

---

## 🚀 Features

- ✅ MoveNet-based pose tracking
- ✅ CLIP-based drill understanding
- ✅ Ball detection with OpenCV
- ✅ Accuracy comparison between coach & player
- ✅ Real-time scoring
- ✅ Video overlay outputs
- ✅ PostgreSQL integration

---

## 📦 Project Structure

```

.
├── models/                  # TFLite + CLIP models
├── services/               # Pose, ball, and CLIP service classes
├── utils/                  # Helper utilities
├── videos/                 # Uploaded videos
├── outputs/                # Generated overlays
├── templates/              # HTML for drill report
├── results/                # Exported CSV logs
├── app.py                  # FastAPI app with all endpoints
├── drill\_evaluator.py      # Core comparison pipeline
├── requirements.txt
├── .env.example

````

---

## 🛠️ Setup Instructions

1. **Clone the repo:**

```bash
git clone https://github.com/Manojnarayan0424/football-ai-service.git
cd football-ai-service
````

2. **Create virtual env and install:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Setup environment:**

```bash
cp .env.example .env
# Fill in your credentials
```

4. **Run server:**

```bash
uvicorn app:app --reload
```

5. **Access Swagger UI:**

```
http://127.0.0.1:8000/docs
```

---

## 📤 API Endpoints

| Method | Endpoint                             | Description                    |
| ------ | ------------------------------------ | ------------------------------ |
| POST   | `/v1/upload-drill/`                  | Upload coach and student video |
| POST   | `/v1/player/performance`             | Log performance to DB          |
| GET    | `/v1/player/{player_id}/performance` | Fetch summary from DB          |

---

## 📸 Screenshots

You can upload Swagger screenshots here or add these lines:

```
![Swagger UI](screenshots/swagger-ui.png)
```

---

## 🧠 Tech Stack

* FastAPI
* OpenCV
* TensorFlow Lite (MoveNet)
* CLIP (ViT-B/32)
* PostgreSQL
* SendGrid (for OTP)

---

## 👨‍💻 Author

**Manoj Narayan**
[GitHub Profile](https://github.com/Manojnarayan0424)

---

## 📄 License

This project is licensed under the MIT License.

````

---

