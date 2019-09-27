import cv2
import tensorflow as tf

CATEGORIES = ["Recycleable", "UnRecycleable"]  
#CATEGORIES = ["可回收", "不可回收"]  

def prepare(filepath):
    IMG_SIZE = 50  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("GSF2.model")


prediction = model.predict([prepare('Desktop/tem/pum.jpg')])
print(prediction)  

if prediction[0][0] !=1 and prediction[0][0] !=0 and prediction[0][0]<0: 
    if(abs(1-prediction[0][0])>abs(0-prediction[0][0])):
        prediction[0][0]=1
    else:
        prediction[0][0]=0
else: 
    if(abs(1-prediction[0][0])>abs(0-prediction[0][0])):
        prediction[0][0]=0
    else:
        prediction[0][0]=1
print(CATEGORIES[int(prediction[0][0])])
