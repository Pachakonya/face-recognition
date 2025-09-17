import cv2

alg = "/Users/sugirbay/face_recognition/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
file_name = "/Users/sugirbay/face_recognition/my_boys.png"
img = cv2.imread(file_name, 0)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)

i = 0
for x, y, w, h in faces:
    cropped_image = img[y : y + h, x : x + w]
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1

import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

conn = psycopg2.connect("postgres://avnadmin:AVNS_GBFtYZZ0J-QLFN5dLVV@pg-3b146124-faces02123.c.aivencloud.com:17875/defaultdb?sslmode=require")

for filename in os.listdir("stored-faces"):
    img = Image.open("stored-faces/" + filename)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pictures (picture, embedding)
        VALUES (%s, %s)
        ON CONFLICT (picture) DO NOTHING
    """, (filename, embedding[0].tolist()))
    print(f"Inserted: {filename}")
conn.commit()

file_name = "/Users/sugirbay/face_recognition/IMG_0913.png"  
img = Image.open(file_name)
ibed = imgbeddings()
embedding = ibed.to_embeddings(img)

from IPython.display import Image, display

cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
cur.close()

for row in rows:
    img_path = f"stored-faces/{row[0]}"
    img = cv2.imread(img_path)
    if img is not None:
        cv2.imshow("Matched Image", img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
