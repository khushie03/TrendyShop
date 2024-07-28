import requests
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from colormap import rgb2hex , hex2rgb 
from skimage.color import rgb2lab, deltaE_cie76
import requests
from io import BytesIO
import os
from face import detect_and_crop_faces
from eye import detect_and_crop_eyes
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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


def return_top_6(face_hex , foundation_hexlist):
    face_hex = hex2rgb(face_hex)
    face_array = np.unit8(np.asarray([[face_hex]]))
    face_ = rgb2lab(face_array)
    foundat = []
    for hex_code in foundation_hexlist:
        foundation_rgb = hex2rgb(hex_code)
        foundat.append(foundation_rgb)
    facefound_array = np.unit8(np.asarray([foundat]))
    face_found = rgb2lab(facefound_array)
    distance_color = deltaE_cie76(face_found , face_)
    sort_distance = np.argsort(distance_color)
    sort_distance = sort_distance.squeeze()
    top_6 = []
    for i in sort_distance[:6]:
        top_6.append(i)
    return top_6

"Returning the list of the top foundation with respect to the matching hex code"

import json
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



def match_foundation_list(hex_code , sort_by , min_price , max_price ):
    
    url = "https://sephora.p.rapidapi.com/us/products/v2/search"
    query = hex_code + "foundation"
    sort_by = sort_by
    querystring = {"pageSize":"60","currentPage":"1","q":query,"sortBy":sort_by , "pl" : min_price ,
                  "ph" : max_price }
    headers = {
        "x-rapidapi-key": "2feb5257eemsh93c5c510406ef6dp102d41jsncb609158bbf1",
        "x-rapidapi-host": "sephora.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    #print(response.json())
    results = extract_product_info(response.json())
    return results


def product_Recommend(image , sort_by ,min_price , max_price ):
    img = Image.open(img_path)
    img_eye = detect_and_crop_eyes(img)
    img_face= detect_and_crop_faces(img)
    eye = get_hex_url(num_clusters= 10 , img_path= img_eye)
    face = get_hex_url(num_clusters= 10 , image_path = eye)
    match_foundation = match_foundation_list(face , sort_by ,min_price , max_price)
    extract_product_info(match_foundation)
    return extract_product_info
prompt="""You are the Color Analyzer that will recommend color to the people that
the color that will suit them according to their eye color and face color . 
So according to it recommend the best palatte colors and fashion analysis to wear  """

def color_analyis(eye_hex_code , face_hex_code):
    transcript_text = f"eye color : {eye_hex_code} and face color : {face_hex_code}"
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

img_path = r"C:\PROJECTS\TrendyShop\cropped_faces\face_1.jpg"  
dominant_color_hex = get_hex_url(num_clusters=10, img_path=img_path)
print(dominant_color_hex)
eye_img_path = r"C:\PROJECTS\TrendyShop\cropped_eyes\eye_1.jpg"  
dominant_color_eye = get_hex_url(num_clusters=10, img_path=eye_img_path)
print(dominant_color_eye)