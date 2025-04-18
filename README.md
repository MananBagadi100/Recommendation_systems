# 📦 Product Recommendation System

A collaborative filtering-based product recommendation system built with Python, Flask, and Docker.  
It uses user ratings and cosine similarity to suggest similar products based on a selected product ID.

---

## 🚀 Features

- Item-based collaborative filtering using cosine similarity
- Flask web app for clean user input/output
- Dockerized setup for easy deployment
- Input: Product ID  
- Output: Top 5 similar products with similarity scores

---

## 🛠️ Tech Stack

- Python (core logic)
- Flask (web framework)
- Jinja2 (HTML templating)
- Pandas, NumPy, Scikit-learn (matrix operations)
- Docker (containerization)

---

## ⚙️ How to Run

### 🧱 1. Build the Docker Image
```bash
docker build -t recommender-app .
```

### ▶ 2. Run the Container
```bash
docker run -p 5050:5000 recommender-app
```

### 🌐 3. Access the App
Visit [http://localhost:5050](http://localhost:5050)

---

## 📄 Example

**Input**: `5`  
**Output**:
```
26 – Similarity: 0.9996  
35 – Similarity: 0.9984  
24 – Similarity: 0.9977  
...
```

---

## 📁 Folder Structure

```
.
├── app.py                     # Flask application
├── builder_app.py             # Builds similarity matrix & saves .pkl files
├── recommendation.py          # Core recommendation logic
├── templates/
│   └── index.html             # Frontend form + output
├── cosine_item_similarity.pkl
├── user_item_matrix.pkl
├── Dockerfile
└── requirements.txt
```

---

## 🙋‍♂️ Author

**Manan Bagadi**  
Feel free to fork, use, and improve it.
