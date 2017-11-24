
import numpy as np
import cv2
import math


def conv(im, size_y, size_x, t, t_size_h, t_size_w, grey_level=0):
    if t_size_h%2==0:
        t_size_h+=1
    if t_size_w%2==0:
        t_size_w+=1
    clone = np.zeros_like(im)
    kh = t_size_h
    kw = t_size_w
    hh = kh//2
    hw = kw//2
    if grey_level==0:
        for y in range(hh, size_y-(kh-hh)):
            for x in range(hw, size_x-(kw-hw)):
                sum = [0,0,0]
                for j in range(kh):
                    jj = kh - 1 - j
                    for i in range(kw):
                        ii = kw - 1 - i
                        rx = x + i - hw
                        ry = y + j - hh
                        sum[0] += im[ry,rx,0]*t[jj][ii]
                        sum[1] += im[ry,rx,1]*t[jj][ii]
                        sum[2] += im[ry,rx,2]*t[jj][ii]
                clone[y,x,0]=sum[0]
                clone[y,x,1]=sum[1]
                clone[y,x,2]=sum[2]
    else:
        for y in range(hh, size_y-(kh-hh)):
            for x in range(hw, size_x-(kw-hw)):
                sum = 0
                for j in range(kh):
                    jj = kh - 1 - j
                    for i in range(kw):
                        ii = kw - 1 - i
                        rx = x + i - hw
                        ry = y + j - hh
                        sum += im[ry,rx]*t[jj][ii]
                clone[y,x]=sum
    return clone

def gaus_m(sigma, winsize):
    c1 = 1/(2*math.pi*sigma*sigma)
    c2 = 1/(2*sigma*sigma)
    centre = winsize//2+1
    sum = 0
    template = np.zeros((winsize,winsize))
    for i in range(winsize):
        for j in range(winsize):
            template[j,i]= c1*math.exp(-c2*((j-centre)**2+(i-centre)**2))
            sum += template[j,i]
    template/=sum
    return template

sigma_1 = float(input('Input sigma for a low pass filter: '))
#sigma_1 = 2
size_1 = int(8*sigma_1+1)
size_1 = int(input('Input matrix size (preferably ) '+str(size_1)+': '))
if size_1%2==0: size_1+=1

gaus_mtr_1 = np.zeros((size_1,size_1))
gaus_mtr_1 = gaus_m(sigma_1, size_1)

sigma_2 = float(input('Input sigma for a high pass filter: '))
size_2 = int(8*sigma_2+1)
size_2 = int(input('Input matrix size (preferably ) '+str(size_2)+': '))
if size_2%2==0: size_2+=1

gaus_mtr_2 = np.zeros((size_2,size_2))
gaus_mtr_2 = gaus_m(sigma_2, size_2)

print ('which images pair do you choose to hybrid?')
print ('1.dog-cat')
print ('2.Marilyn-Einstein')
print ('3.motorcycle-bicycle')
print ('4.fish-submarine')
print ('5.airplane-bird')
choise = int(input())
if choise==1:
    name_smooth = 'dog.bmp'
    name_thresh = 'cat.bmp'
elif choise==2:
    name_smooth = 'marilyn.bmp'
    name_thresh = 'einstein.bmp'
elif choise==3:
    name_smooth = 'motorcycle.bmp'
    name_thresh = 'bicycle.bmp'
elif choise==4:
    name_smooth = 'fish.bmp'
    name_thresh = 'submarine.bmp'
else:
    name_smooth = 'plane.bmp'
    name_thresh = 'bird.bmp'
grey_ = int(input('input 1, if gray image, 0 if RGB: '))

if grey_==0:
    cat = cv2.imread(name_thresh).astype(float)
    dog = cv2.imread(name_smooth).astype(float)
    x_dog, y_dog, t_dog = dog.shape
else:
    cat = cv2.imread(name_thresh,0).astype(float)
    dog = cv2.imread(name_smooth,0).astype(float)
    x_dog, y_dog = dog.shape

dog = conv(dog, x_dog, y_dog, gaus_mtr_1, size_1, size_1,grey_)
cat =  dog+ cat - conv(cat, x_dog, y_dog, gaus_mtr_2, size_2, size_2,grey_)
if (grey_==0):
    for i in range(x_dog):
        for j in range(size_2//2+1):
            cat[i,j]=[0,0,0]
        for j in range(y_dog-size_2//2-1,y_dog):
            cat[i,j]=[0,0,0]
    for i in range(y_dog):
        for j in range(size_2//2+1):
            cat[j,i]=[0,0,0]
        for j in range(x_dog-size_2//2-1,x_dog):
            cat[j,i]=[0,0,0]
else:
    for i in range(x_dog):
        for j in range(size_2//2+1):
            cat[i,j]=0
        for j in range(y_dog-size_2//2-1,y_dog):
            cat[i,j]=0
    for i in range(y_dog):
        for j in range(size_2//2+1):
            cat[j,i]=0
        for j in range(x_dog-size_2//2-1,x_dog):
            cat[j,i]=0

cv2.imwrite('dog_1.png',dog)
cv2.imwrite('cat_1.png',cat)
cv2.imshow("outImg", cat/255)
cv2.waitKey(0)
