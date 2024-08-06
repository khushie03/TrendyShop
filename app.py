from flask import Flask, render_template, request, redirect
import os
from PIL import Image
from io import BytesIO
import requests
from colormap import rgb2hex
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import google.generativeai as genai
from serpapi import GoogleSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\PROJECTS\TrendyShop\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Configure API keys
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
        "x-rapidapi-key": "YOUR_RAPIDAPI_KEY",
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
Write the text in the paragraph form no bullet points or bold text. Just simple text. And Just Provide the text nothing else.
"""

@app.route('/color_analysis', methods=['POST'])
def color_analysis():
    face_hex = request.form.get('face_hex')
    eye_hex = request.form.get('eye_hex')
    transcript_text = f"eye color: {eye_hex} and face color: {face_hex}"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return render_template('color_analysis.html', analysis=response.text)

def trend_search(product_name):
    def link_extraction(product_name):
        params = {
            "engine": "youtube",
            "search_query": product_name,
            "api_key": "c8b912a9727723424bffac813a03eb897d43cee8cfac0741c3b266a6cb8bef71"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        video_results = results.get("video_results", [])
        return video_results

    def get_video_id_from_url(youtube_video_url):
        if "watch?v=" in youtube_video_url:
            return youtube_video_url.split("watch?v=")[1]
        elif "shorts/" in youtube_video_url:
            return youtube_video_url.split("shorts/")[1]
        else:
            return None

    def fetch_transcript(video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return transcript
        except (CouldNotRetrieveTranscript, NoTranscriptFound):
            print(f"Error: Could not retrieve a transcript for the video {video_id}!")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    
    def generate_gemini_content(transcript_text):
        prompt = """
        Here I am trying to analyze the trends through the trending videos and accordingly generating a list of the products.
        Here you have given the transcripts of the trending YouTube videos and you have to generate a list of the
        products that are mentioned in that transcripts. Here automated trend extractor is there which will automatically extract the products 
        related to the videos.
        """
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt + transcript_text)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    video_results = link_extraction(product_name)
    links, thumbnails = [], []

    for video in video_results:
        links.append(video['link'])
        thumbnails.append(video['thumbnail'])
    
    products = []
    for link in links:
        video_id = get_video_id_from_url(link)
        if video_id:
            script = fetch_transcript(video_id)
            print(script)
            if script is None:
                continue
            if script is not None:
                transcript_text = " ".join([item['text'] for item in script])
                product_list = generate_gemini_content(transcript_text)
                products.append(product_list)
            else:
                continue
        else:
            products.append("Invalid video link")
    
    def link_create(product):
        params = {
            "engine": "google_shopping",
            "q": product,
            "api_key": "c8b912a9727723424bffac813a03eb897d43cee8cfac0741c3b266a6cb8bef71"
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results.get("shopping_results", [])
        return shopping_results

    shopping_links = []
    for product in products[:3]:
        if product.startswith("Error:"):
            continue
        shopping_results = link_create(product)
        if shopping_results:
            for result in shopping_results:
                shopping_links.append(result['link'])

    return shopping_links, thumbnails, products

@app.route('/trend_search', methods=['GET', 'POST'])
def trend_search_route():
    if request.method == 'POST':
        product_name = request.form.get('product_name')
        print(f"Searching for product: {product_name}")
        results, thumbnails, products = trend_search(product_name)
        if results is None:
            results = []
        print(f"Search results: {results}")
        return render_template('trend_results.html', results=results, thumbnails=thumbnails, products=products, product_name=product_name)
    return render_template('trend_search.html')

if __name__ == '__main__':
    app.run(debug=True)
