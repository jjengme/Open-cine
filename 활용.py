import numpy as np
from basicfunctions import cine2nparrall, cine2nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv
import random

# Phantom 인스턴스 생성
ph = Phantom() # cine 파일을 불러올 때 항상 해줘야됨

# cine 파일 경로와 프레임 설정
fname = 'Z:/03 exp/220126 ilasskorea/2bar 2x/2bar 112.cine'
frame = 0

# cine 파일을 numpy array로 변환
src = cine2nparr(fname, frame)




#####################히스토그램 평탄화 후 이진화 후 외곽선 검출#######################
# # 히스토그램 평활화 적용
# src_eq = cv.equalizeHist(src)

# # Otsu의 방법을 사용하여 이진화
# _, src_bin = cv.threshold(src_eq, 0, 255, cv.THRESH_OTSU)

# # 외곽선 검출
# contours, hier = cv.findContours(src_bin, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# # 원본 이미지와 동일한 크기의 빈 이미지 생성
# h, w = src.shape[:2]
# dst = np.zeros((h, w, 3), np.uint8)

# # 외곽선 그리기
# for i in range(len(contours)):
#     c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#     cv.drawContours(dst, contours, i, c, 1, cv.LINE_AA)

# # 결과 출력
# cv.imshow('Original Image', src)
# cv.imshow('Equalized Image', src_eq)
# cv.imshow('Binary Image', src_bin)
# cv.imshow('Contours', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()



#############################코너 검출###########

# corners = cv.goodFeaturesToTrack(src, 400,0.01, 10)
# #400개 픽셀 검출, quality level : 0.01, minDistance : 너무 근접하면 하나를 버리겠다.
# dst1 = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
# if corners is not None:
#     for i in range(corners.shape[0]):
#         pt = (int(corners[i,0,0]), int(corners[i,0,1]))
#         cv.circle(dst1, pt, 5, (0,0,255),2)


# fast = cv.FastFeatureDetector_create(25)
# keypoints = fast.detect(src)

# dst2 = cv.cvtColor(src,cv.COLOR_GRAY2BGR)
# for kp in keypoints :
#     pt = (int(kp.pt[0]), int(kp.pt[1]))
#     cv.circle(dst2, pt, 5, (0,0,255), 2)

# cv.putText(dst1,'good Features To Track', (50,50),cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
# cv.putText(dst2,'Fast Feature Detector', (50,50),cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

# cv.imshow('src',src)
# cv.imshow('dst1',dst1)
# cv.imshow('dst2',dst2)
# cv.waitKey()
# cv.destroyAllWindows()


########################원검출#################

# # 이미지 전처리 (가우시안 블러 적용)
# src_blur = cv.GaussianBlur(src, (9, 9), 2)

# # 원 검출 (HoughCircles 함수 사용)
# circles = cv.HoughCircles(src_blur, 
#                           cv.HOUGH_GRADIENT, 
#                           dp=1, 
#                           minDist=20, 
#                           param1=50, 
#                           param2=30, 
#                           minRadius=0, 
#                           maxRadius=0)

# # 원이 검출된 경우, 원을 이미지에 그리기
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # 원의 외곽선 그리기
#         cv.circle(src, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # 원의 중심점 그리기
#         cv.circle(src, (i[0], i[1]), 2, (0, 0, 255), 3)

# # 결과 출력
# cv.imshow('Detected Circles', src)
# cv.waitKey(0)
# cv.destroyAllWindows()



####################### 원검출....##########################

# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# blr = cv.GaussianBlur(gray, (0, 0), 1.0)

# def on_trackbar(pos):
#     rmin = cv.getTrackbarPos('minRadius', 'img')
#     rmax = cv.getTrackbarPos('maxRadius', 'img')
#     th = cv.getTrackbarPos('threshold', 'img')
    
#     circles = cv.HoughCircles(blr, cv.HOUGH_GRADIENT, 1, 50, param1=120, param2=th, minRadius=rmin, maxRadius=rmax)
#     dst = src.copy()
    
#     if circles is not None:
#         for i in  range(circles.shape[1]):
#             cx, cy, radius = circles[0][i]
#             cv.circle(dst, (int(cx), int(cy)), int(radius), (0, 0, 255), 2, cv.LINE_AA)



# cv.imshow('circles', circles)
# cv.waitKey()
# cv.destroyAllWindows() 










# 이미지 전처리 (가우시안 블러 적용)
src_blur = cv.GaussianBlur(src, (9, 9), 2)

# Canny 엣지 검출
edges = cv.Canny(src_blur, 50, 150)

# 외곽선 검출
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 원만 검출하여 그리기
dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)  # 원본 이미지를 컬러 이미지로 변환
for cnt in contours:
    # 근사 다각형으로 외곽선 압축
    approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
    
    # 외곽선이 충분히 큰지 확인 (작은 노이즈 무시)
    area = cv.contourArea(cnt)
    if area > 100:
        # 외곽선의 원형률 (perimeter^2 / (4π * area)) 계산하여 원에 가까운지 확인
        circularity = (4 * np.pi * area) / (cv.arcLength(cnt, True) ** 2)
        if 0.7 < circularity <= 1.2:  # 원형률이 1에 가까울수록 원에 가까움
            # 원 모양으로 판단된 외곽선 그리기
            cv.drawContours(dst, [cnt], -1, (0, 255, 0), 2)

# 결과 출력
cv.imshow('Original Image', src)
cv.imshow('Edges', edges)
cv.imshow('Circles Contours', dst)
cv.waitKey(0)
cv.destroyAllWindows()
