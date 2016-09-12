from PIL import Image
import numpy as np

# load the image, convert to graylevel and numpy array
def load_image(filename) :
    img = Image.open(filename)
    img = img.convert('L')
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

# save a numpy array as a graylevel image
def save_image(arr, filename) :
    img = Image.fromarray( np.asarray( np.clip(arr,0,255), dtype="uint8"), "L" )
    img.save(filename)
    
# adjust luminosity of numpy array, rectify if saturation occurs
def adjust_luminosity(arr, mean):
    arr = arr*(mean/arr.mean())
    arr[arr > 255] = 255
    return arr

#load all the files in the folder
path = "C:\\faces\\"
images = list()
first = True

import os
for file in os.listdir(path):
    if file.endswith(".jpg"):
        images.append(load_image(path + file))
        if(first):
            mean = images[-1].mean()
            first = False
        else:
            images[-1] = adjust_luminosity(images[-1], mean)
            print(images[-1].mean())
        
M = images[0].shape[0]

# flatten each images and remove the mean face
mean_face = np.empty((M*M,)) 

for i in range(len(images)):
    images[i] = images[i].flatten()

mean_face = np.mean(images, axis=0)
mean_face = mean_face.astype(int)
images = images - mean_face

# get the principal components and sort them by eigenvalues
C= np.dot(images, np.transpose(images))
eigVal, eigVec = np.linalg.eig(C)

idx = eigVal.argsort()[::-1]   
eigVal = eigVal[idx]
eigVec = eigVec[:,idx]

# create the ghost images and save them
ghost = list()

for i in range(len(images)):
    ghost.append(np.dot(np.transpose(images), eigVec[i]))
    ghost[i] = 255 * (ghost[i] + (-1)*np.min(ghost[i])) / (np.max(ghost[i]) + (-1)*np.min(ghost[i]))
    ghost[i] = np.reshape(ghost[i], (64,64))
    save_image(ghost[i], "ghost" + str(i) + ".jpg")


# face detection: calculate a distance
path = "C:\\detection\\"
d = []
for file in os.listdir(path):
    if file.endswith(".jpg"):
        new_face = load_image(path + file)
        new_face = adjust_luminosity(new_face, mean)
        new_face = new_face.flatten()
        
        u = np.dot(np.transpose(images), eigVec)
        u = u[:,1:] #remove the first eigenface since it is mainly affected by luminosity
        a = np.identity(np.shape(new_face)[0]) - np.dot(u, np.transpose(u))
        b = new_face - mean_face

        distance = np.dot(a, b)
        d.append(np.linalg.norm(distance, ord=1))
        
print(d)