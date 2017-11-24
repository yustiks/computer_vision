import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

#mass_n = int(input("Input size of matrix convolution n: "))
#mass_m = int(input("Input size of matrix convolution m: "))
#grey_ = int(input("If grey picture, input 1, else input 0"))
#standart_dev = float(input("Input standart deviation: "))


dog = cv2.imread('dog1.bmp')
cat = cv2.imread('cat1.bmp')

# convolution!!!!

def con(im, mass, mass_n, mass_m):
    kh = mass_n
    kw = mass_m
    hh = kh // 2
    hw = kw // 2
    im_h, im_w, t = im.shape
    im_result = np.zeros_like(im)
    for y in range(hh, im_h-(kh-hh)):
        for x in range(hw, im_w-(kw-hw)):
            sum = [0,0,0]
            for j in range(kh):
                jj = kh - j -1
                for i in range(kw):
                    ii = kw - i -1
                    rx = x+i-hw
                    ry = y+j-hh
                    b = im[ry,rx,0]
                    g = im[ry,rx,1]
                    r = im[ry,rx,2]
                    sum[0] += b*mass[jj][ii]
                    sum[1] += g*mass[jj][ii]
                    sum[2] += r*mass[jj][ii]
            im_result[y, x] = sum
    return im_result

# gaus massiv
grey_ = 0
sigma = 0.9

size = int(8*sigma + 1)
if size%2 == 0: size+=1

gaus_m = np.zeros((size,size))
centre = size//2+1
sum = 0
for i in range(size):
    for j in range(size):
        gaus_m[j,i] = 1/(2*math.pi*sigma**2)*math.exp(-(((j-centre)*(j-centre))+((i-centre)*(i-centre)))/(2*sigma*sigma))
        sum = sum + gaus_m[j,i]
gaus_m = gaus_m/sum

#high pass matrix
mtr = [[0.002, 0.013, 0.220, 0.013, 0.002],
       [0.013, 0.060, 0.098, 0.060, 0.013],
       [0.220, 0.098, 0.162, 0.098, 0.220],
       [0.013, 0.060, 0.098, 0.060, 0.013],
       [0.002, 0.013, 0.220, 0.013, 0.002]]
s1 = np.sum(mtr)
mtr /= s1
mtr_size = 5


dog_clon = con(dog,gaus_m,size,size)
cat_clon = con(cat,gaus_m,size,size)

plt.figure(1)
plt.subplot(121)
plt.imshow(cv2.cvtColor(dog_clon, cv2.COLOR_BGR2RGB))
plt.axis('off')


ht, wt, t = cat.shape
cat_clon1 = np.zeros_like(cat)
for i in range(ht):
    for j in range(wt):
        cat_clon1[i,j] = cat[i,j] - cat_clon[i,j]+dog_clon[i,j]


#dog_clon = con(dog, gaus_m, size, size)
#cat_clon = con(cat, high_pass, size, size)


plt.subplot(122)
#plt.imshow(cv2.cvtColor(dog, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(cat_clon1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


