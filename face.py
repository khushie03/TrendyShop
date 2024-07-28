import cv2
import os

def detect_and_crop_faces(image_path):
    face_cascade = cv2.CascadeClassifier('C:/PROJECTS/TrendyShop/haarcascade_frontalface_alt.xml')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return face
    
    return None

image_path = r'C:\PROJECTS\TrendyShop\WhatsApp Image 2024-07-24 at 20.53.24.jpeg'  
output_dir = 'cropped_faces'          
num_faces = detect_and_crop_faces(image_path, output_dir)
print(f"Total faces detected and saved: {num_faces}")
