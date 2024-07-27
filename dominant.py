import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from colormap import rgb2hex , hex2rgb 
from skimage.color import rgb2lab, deltaE_cie76
import requests
from io import BytesIO
import os

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

def match_foundation_list(hex_code):
    pass

img_path = r"C:\PROJECTS\TrendyShop\cropped_faces\face_1.jpg"  
dominant_color_hex = get_hex_url(num_clusters=10, img_path=img_path)
print(dominant_color_hex)
