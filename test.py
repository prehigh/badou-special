# from paddleocr import PaddleOCR, draw_ocr
# from paddleocr import *
from PIL import Image

import cv2
import numpy as np
# from paddleocr import PaddleOCR, draw_ocr
# import easyocr
import imutils
from imutils import contours
import torch



# model = torch.hub.load("ultralytics/yolov5", "custom", path = "./yolov5s.pt", force_reload=True)
model = torch.hub.load("ultralytics/yolov5", "custom", path = "C:/Users/wenting/PaddleOCR/best.pt", force_reload=True)

# Set Model Settings
# model.eval()
# model.conf = 0.6  # confidence threshold (0-1)
# model.iou = 0.45  # NMS IoU threshold (0-1) 


print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

src_img = cv2.imread("C:/Users/wenting/PaddleOCR/image/321350.jpg")
img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./image/5.jpg", img)
img1 = img[:img.shape[0]//5, :img.shape[1]//5]
graythread = np.min(img1) + (np.max(img1)-np.min(img1))//2
img1 = np.array(cv2.threshold(img1, np.min(img1)+np.std(img1[img1<graythread])*1.5, 255, cv2.THRESH_BINARY)[1])
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

# left top point
left1 = 0 
top1 = 0
for i in range(img1.shape[0]):
    if left1 != 0 and top1 != 0:
        break
    for j in range(img1.shape[1]):
        if img1[i, j] == 0 and np.mean(img1[i:img1.shape[0], j:j+20])<100 and np.mean(img1[i:i+20, j:img1.shape[1]])<100:
            left1 = j
            top1 = i
            break
print(left1, top1)

# right buttom  point
right2 = 0 
buttom2 = 0
for i in range(img2.shape[0]-1, 1, -1):
    if right2 != 0 and buttom2 != 0:
        break
    for j in range(img2.shape[1]-1, 0, -1):
        if img2[i, j] == 0 and np.mean(img2[0:i, j-20:j])<100 and np.mean(img2[i-20:i, 0:j])<100:
            right2 = j+img.shape[1]//5*4
            buttom2 = i+img.shape[0]//5*4
            break
print(right2, buttom2)

# right top  point
right3 = 0 
top3 = 0
for i in range(img3.shape[0]):
    if right3 != 0 and top3 != 0:
        break
    for j in range(img3.shape[1]-1, 0, -1):
        if img3[i, j] == 0 and np.mean(img3[i:img3.shape[0]-1, j-20:j])<100 and np.mean(img3[i:i+20, 0:j])<100:
            right3 = j+img.shape[1]//5*4
            top3 = i
            break
print(right3, top3)

# left buttom  point
left4 = 0 
buttom4 = 0
for i in range(img4.shape[0]-1, 0, -1):
    if left4 != 0 and buttom4 != 0:
        break
    for j in range(img4.shape[1]):
        if img4[i, j] == 0 and np.mean(img4[i-20: i, j: img4.shape[1]-1])<100 and np.mean(img4[:i, j:j+20])<100:
            left4 = j
            buttom4 = i+img.shape[0]//5*4
            break
print(left4, buttom4)

# warpPerspective
dstTri = np.array( [ [0, 0],[0, 4000],[3000, 0] ,[3000,4000]] ).astype(np.float32)
srcTri = np.array( [[left1, top1],[right3, top3] ,[left4, buttom4],[right2,buttom2]] ).astype(np.float32)
warp_mat = cv2.getPerspectiveTransform(srcTri, dstTri)
warp_dst = cv2.warpPerspective(src_img, warp_mat, (3000, 4000))
terminal_img = cv2.rotate(cv2.flip(warp_dst, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
#cv2.imwrite("./image/9.jpg", terminal_img)



cut_list = [[160, 272, 1668, 1847],
            [273, 386, 1756, 2586],
            [46, 163, 2823, 3040],
            [48, 165, 3034, 3267],
            [54, 171, 3262, 3495],
            [156, 275, 2823, 3041],
            [158, 274, 3035, 3268],
            [162, 280, 3264, 3502],
            [923, 1037, 493, 753],
            [820, 938, 496, 756],
            [1496, 1614, 497, 759],
            [1396, 1512, 498, 757],
            [1786, 1897, 501, 758],
            [1684, 1801, 502, 758],
            [2074, 2185, 505, 762],
            [1973, 2091, 502, 763],
            [2353, 2466, 505, 764],
            [2251, 2368, 503, 767],
            [780, 890, 1361, 1629],
            [868, 989, 1360, 1630],
            [1877, 1993, 1230, 1428],
            [1975, 2087, 1227, 1552],
            [2356, 2464, 1240, 1406]
            ]

# ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
# reader = easyocr.Reader(['en'], gpu=True)
for i in range(23):
    cut_image = terminal_img[cut_list[i][0]:cut_list[i][1], cut_list[i][2]:cut_list[i][3]]
    
    # cut_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2GRAY)
    # cut_image = np.array(cv2.threshold(cut_image, np.max(cut_image)-np.std(cut_image), 255, cv2.THRESH_BINARY)[1])
    
    # cut_image = np.array(cv2.threshold(cut_image, 200, 255, cv2.THRESH_BINARY)[1])
    # cut_image = cv2.copyMakeBorder(cut_image,20,20,20,20, cv2.BORDER_CONSTANT,value=[0,0,0])

    cv2.imwrite("./image/cut1_image/cut_img"+str(i)+".jpg", cut_image)
    cut_image = cv2.resize(cut_image, (512, 512))
    results = model(cut_image)
    # print(results)
    print(results.pandas().xyxy[0])
    
    '''
    # my test
    edge, hierarchy = cv2.findContours(cut_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(len(edge))
    print(edge)
    #refCnts = imutils.grab_contours(edge)
    #print(refCnts)
    # cv2.imshow("cut_image", cut_image[contours[0][0, 0, 0]:contours[0][-1, 0, 0], contours[0][0, 0, 1]:contours[0][-1, 0, 1]])
    # cv2.waitKey(0)


    result = ocr.ocr(cut_image, cls=True)
    for line in result:
        print(line[1][0])
        
    result = reader.readtext(cut_image, detail = 0)
    print(result)

    # 显示结果
    
    image = Image.open("./image/111.jpg").convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    '''
    # cv2.imshow("cut_image", cut_image)
    # cv2.waitKey(0)

    