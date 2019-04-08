import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


img = cv2.imread("roi_extract.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.reshape((img.shape[0] * img.shape[1],3))
clt = KMeans(n_clusters=1)
clt.fit(img)
hist = find_histogram(clt)
color_array = [[clt.cluster_centers_][0][0][0],[clt.cluster_centers_][0][0][1],[clt.cluster_centers_][0][0][2]]


print(color_array)
delta_array = 255-max(color_array)
max_index = color_array.index(max(color_array))
min_index = color_array.index(min(color_array))
color_array[max_index] = 255
color_array[min_index] = int(color_array[min_index])

for id_color in range(0,len(color_array)):
    if id_color != min_index and id_color != max_index:
        color_array[id_color] = int(color_array[id_color])


img = cv2.imread("roi_extract.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color_array1 = color_array.copy()
i = 0
j = 0
while i < len(color_array1):
    if color_array1[i] != 255 and j != 1:
        color_array1[i] = 255
        j += 1
    i += 1

color_array2 = color_array.copy()
i = 0
j = 0
while i < len(color_array2):
    if color_array2[-i] != 255 and j != 1:
        color_array2[-i] = 255
        j += 1
    i += 1



lower = np.array([0, 0, 0])
upper = np.array(color_array1)
mask = cv2.inRange(hsv, lower, upper)
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 0:
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx+1, topy:bottomy+1]
        cv2.imwrite('X.png',out)
image1 = cv2.imread("roi_extract.png")
image1 = cv2.resize(image1,(int(255),int(255)))
image2 = cv2.imread('X.png')
image2 = cv2.resize(image2,(int(255),int(255)))
img = image1 - image2
cv2.imwrite('crop_out.png',img)


img = cv2.imread("crop_out.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array(color_array2)
mask = cv2.inRange(hsv, lower, upper)
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 0:
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx+1, topy:bottomy+1]
        cv2.imwrite('X.png',out)
image1 = cv2.imread("crop_out.png")
image1 = cv2.resize(image1,(int(255),int(255)))
image2 = cv2.imread('X.png')
image2 = cv2.resize(image2,(int(255),int(255)))
img = image1 - image2
cv2.imwrite('crop_out1.png',img)



img = cv2.imread("roi_extract.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array([255,255,255])
mask = cv2.inRange(hsv, lower, upper)
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 0:
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx+1, topy:bottomy+1]
        cv2.imwrite('X.png',out)

image1 = cv2.imread("roi_extract.png")
image1 = cv2.resize(image1,(int(255),int(255)))
image2 = cv2.imread('X.png')
image2 = cv2.resize(image2,(int(255),int(255)))
img = image1 - image2
cv2.imwrite('crop_out3.png',img)

image1 = cv2.imread("crop_out1.png")
image1 = cv2.resize(image1,(int(255),int(255)))
image2 = cv2.imread('crop_out3.png')
image2 = cv2.resize(image2,(int(255),int(255)))
img = image1 - image2
cv2.imwrite('final.png',img)


