
from flask import Flask, render_template, request
import pickle
from recommendation import get_collab_recommendation

app = Flask(__name__)

# Load .pkl files
with open('cosine_item_similarity.pkl', 'rb') as f:
    cosine_item_similarity = pickle.load(f)
    print('Loaded the cosine similarity : ',cosine_item_similarity.shape)

with open('user_item_matrix.pkl', 'rb') as f:
    user_item_matrix = pickle.load(f)
    print('Loaded user item matrix ',user_item_matrix.shape)

@app.route('/', methods=['GET', 'POST'])
def index():
    print('index route triggered')
    recommendations = {}

    if request.method == 'POST':
        product_id = int(request.form['product_id'])
        #debuging statements
        print("🛠 Raw product_id input:", request.form['product_id'])
        print("✅ After strip:", product_id)
        print("🧪 Type of product_id:", type(product_id))
        product_id=int(product_id)
        recommendations = get_collab_recommendation(
            product_id,
            cosine_item_similarity,
            user_item_matrix
        )

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',use_reloader=True,port=5000)
'''[ 1,  2,  3,  4,  5,  6,  7,  9, 15, 16, 17, 18, 24, 26, 27, 28, 29, 30,
32, 33, 34, 35, 38, 39, 40, 41, 43, 45, 47, 48, 49, 51, 52, 53, 54],are 
valid product_id's  '''