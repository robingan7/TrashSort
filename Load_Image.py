import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

data = "Desktop/Programs/GSF/images"
Cate = ["recy","unrecy"]
'''
for category in Cate:
    path = os.path.join(data,category)
    class_num = Cate.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        break
    break  '''
    
size=50

training_data=[]
def create_training_data():
    for category in Cate:
        path = os.path.join(data,category)
        class_num = Cate.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(size,size))
                training_data.append([new_array,class_num])
            except Exception as e:
                print("error")
            
create_training_data()  
print(len(training_data))       
