import cv2 
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

from scipy.spatial.distance import cosine
from IPython.display import display, Image as IPImage
import numpy as np
from PIL import Image

#firstly the facial regonition algorithm itself import it and store into var"
face_id_alg = "haarcascade_frontalface_default.xml"

#then pass the algorithm into openCV library 
haar_cascade = cv2.CascadeClassifier(face_id_alg)

#store avengers case into var 
avengers_src = "avengers.jpg"

# load the image to scan into var, and use cv2 to read the image
# 0 implies reading type, 0 means read the img in grayscale mode 
avengers_cv2 = cv2.imread(avengers_src, 0)

#create and black and white version of img 
bw_avengers = cv2.cvtColor(avengers_cv2, cv2.COLOR_RGB2BGR)


'''
a little bit about faces and whats happening here 
algorithm runs thru image to detect faces, and then stores them in an array of 4 values representing a square 
1st value - x coordinate of square 
2nd value - y coordinate of square 
3rd value - width of sqaure 
4th value - height of square
'''
#detect the faces via the algorithm 
#image - bw_avengers
#scaleFactor - compression amount
#min neighbors - how many neighboring faces to find when it does find one
#minSize - minimum size of a face in terms of pixels 
faces = haar_cascade.detectMultiScale(
    bw_avengers, 
    scaleFactor=1.05, 
    minNeighbors=5, 
    minSize=(100,100)
)

#now that we have detected the face, we modify our selections 

i = 0
for x,y,w,h in faces:
    #firstly crop the image so only the face is selected 
    cropped = avengers_cv2[y : y + h, x : x + w]

    #now we create a file for that specific face 
    target_file_name = 'stored_faces/' + str(i) + '.jpg'

    #cv2 method to create and save an image 
    cv2.imwrite(
        target_file_name,
        cropped,
    )
    i = i+1




'''
step 2 - convert the images into vector representations
'''
embed_arr = []
for filename in os.listdir("stored_faces"):
    img = Image.open("stored_faces/" + filename)
    img_bedding = imgbeddings()
    embedding = img_bedding.to_embeddings(img)
    embed_arr.append(embedding)


'''
step 3 - load the image to compare and make it into an embedding  
'''
hemsworth = "chris.jpg"
hemsworth = Image.open(hemsworth)
hemsworth_ibed = imgbeddings()
embedding_hemsworth = hemsworth_ibed.to_embeddings(hemsworth)


'''
step 4, make the comparision
'''

# Function to calculate similarity (e.g., cosine similarity)
def calculate_similarity(embedding1, embedding2):
    # Cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Step 4: Compare the embeddings
highest_similarity = 0
most_similar_face = None
most_similar_file = None

for i, embedding in enumerate(embed_arr):
    # Calculate similarity between the standalone face and each stored face
    similarity = 1 - cosine(embedding_hemsworth[0], embedding[0])
    
    # Update the most similar face if current similarity is higher
    if similarity > highest_similarity:
        highest_similarity = similarity
        most_similar_face = embedding
        most_similar_file = f"stored_faces/{i}.jpg"

# Display the most similar face, if found
if most_similar_file:
    print(f"Most similar face found with similarity score: {highest_similarity}")
    Image.open(most_similar_file).show()
else:
    print("No similar face found.")



