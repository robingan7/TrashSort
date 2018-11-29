import cv2
import tensorflow as tf

CATEGORIES = ["Recy", "Unrecy"]  


def prepare(filepath):
    IMG_SIZE = 50  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("GSF.model")

prediction = model.predict([prepare('Desktop/box.jpg')])
print(prediction)  
print(CATEGORIES[int(prediction[0][0])])
