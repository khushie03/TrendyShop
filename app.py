from flask import Flask, render_template, request, redirect
import os
from PIL import Image
from io import BytesIO
import requests
from colormap import rgb2hex
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\PROJECTS\TrendyShop\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_hex_url(num_clusters, img_path):
    if img_path.startswith(('http://', 'https://')):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path)
    return dominant_colors(img, num_clusters)

def dominant_colors(img, num_clusters=10):
    img = img.resize((150, 150))
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.prod(shape[:2]), shape[2]).astype(float)
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        init="k-means++",
        max_iter=20,
        random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_
    vecs = kmeans.predict(ar)
    counts = np.bincount(vecs)
    
    sorted_indices = np.argsort(counts)[::-1]
    peak = codes[sorted_indices[0]]
    
    return rgb2hex(int(peak[0]), int(peak[1]), int(peak[2]))

def extract_product_info(data):
    extracted_info = []
    
    for product in data.get("products", []):
        product_info = {
            "displayName": product.get("displayName"),
            "heroImage": product.get("heroImage"),
            "price": product.get("price"),
            "productId": product.get("productId"),
            "reviews": product.get("reviews"),
            "rating": product.get("rating"),
            "onSaleData": product.get("onSaleData"),
            "isLimitedTimeOffer": product.get("isLimitedTimeOffer")
        }
        extracted_info.append(product_info)
    
    return extracted_info

def match_foundation_list(hex_code, sort_by, min_price, max_price):
    url = "https://sephora.p.rapidapi.com/us/products/v2/search"
    query = hex_code + " foundation"
    querystring = {"pageSize": "60", "currentPage": "1", "q": query, "sortBy": sort_by, "pl": min_price, "ph": max_price}
    headers = {
        "x-rapidapi-key": "2feb5257eemsh93c5c510406ef6dp102d41jsncb609158bbf1",
        "x-rapidapi-host": "sephora.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    results = extract_product_info(response.json())
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main_page')
def main_page():
    return render_template('main_page.html')

@app.route('/foundation')
def foundation():
    return render_template('foundation.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        sort_by = request.form.get('sort_by')
        min_price = request.form.get('min_price')
        max_price = request.form.get('max_price')
        
        face_hex = get_hex_url(10, file_path)
        foundations = match_foundation_list(face_hex, sort_by, min_price, max_price)
        return render_template('results.html', foundations=foundations, face_hex=face_hex)

    return redirect(request.url)

prompt = """
You are a color palette analysis expert with respect to the eye color and face color in hex format.
Suggest the colors that will suit people and the type of clothes they should wear. 
Write the text in the paragraph form no bullet points or bold text . Just simple text. And Just Provide the text nothing else . """

@app.route('/color_analysis', methods=['POST'])
def color_analysis():
    face_hex = request.form.get('face_hex')
    eye_hex = request.form.get('eye_hex')
    transcript_text = f"eye color: {eye_hex} and face color: {face_hex}"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return render_template('color_analysis.html', analysis=response.text)

if __name__ == '__main__':
    app.run(debug=True)
