# from paddleocr import PaddleOCR, draw_ocr
# from paddleocr import *
from PIL import Image

import cv2
import numpy as np
# from paddleocr import PaddleOCR, draw_ocr
# import easyocr

# 读取图片
src_img = cv2.imread("C:/Users/wenting/PaddleOCR/image/321350.jpg")
# 灰度化
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
# 保存图片
cv2.imwrite("./image/5.jpg", img)
# 这是为了截图图片的部分，因为图片曝光不同，所以多个区域使用了不同的二值化阈值
img1 = img[:img.shape[0]//5, :img.shape[1]//5]
# 自动化求取二值化阈值
graythread = np.min(img1) + (np.max(img1)-np.min(img1))//2
# 进行二值化
img1 = np.array(cv2.threshold(img1, np.min(img1)+np.std(img1[img1<graythread])*1.5, 255, cv2.THRESH_BINARY)[1])
# 显示图片
cv2.imshow("img1", img1)
cv2.waitKey(0)

img2 = img[img.shape[0]//5*4: img.shape[0]-1, img.shape[1]//5*4: img.shape[1]-1]
graythread = np.min(img2) + (np.max(img2)-np.min(img2))//2
img2 = np.array(cv2.threshold(img2, np.min(img2)+np.std(img2[img2<graythread])*1.5, 255, cv2.THRESH_BINARY)[1])
cv2.imshow("img2", img2)
cv2.waitKey(0)

img3 = img[:img.shape[0]//5, img.shape[1]//5*4: img.shape[1]-1]
graythread = np.min(img3) + (np.max(img3)-np.min(img3))//2
img3 = np.array(cv2.threshold(img3, np.min(img3)+np.std(img3[img3<graythread])*1.5, 255, cv2.THRESH_BINARY)[1])
cv2.imshow("img3", img3)
cv2.waitKey(0)

img4 = img[img.shape[0]//5*4: img.shape[0]-1, :img.shape[1]//5]
graythread = np.min(img4) + (np.max(img4)-np.min(img4))//2
img4 = np.array(cv2.threshold(img4, np.min(img4)+np.std(img4[img4<graythread])*1.5, 255, cv2.THRESH_BINARY)[1])
cv2.imshow("img4", img4)
cv2.waitKey(0)


    