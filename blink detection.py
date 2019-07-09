import dlib
import cv2
import os
from imutils import face_utils


# 确定眼睛纵横比
def eye_aspect_ratio(eye):
    a = eye[5][1] - eye[1][1]
    b = eye[4][1] - eye[2][1]
    c = eye[3][0] - eye[0][0]
    return (a + b) / (2 * c)


# 获取当前路径
pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
# 加载模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)
# 开启摄像头
cap = cv2.VideoCapture(0)
frame_counter = 0
blink_count = 0
# 读取每一帧进行处理
while 1:
    ret, img = cap.read()
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    rects = detector(gray, 0)
    # 对检测到的每个人脸处理
    for rect in rects:
        # 在图像中框出人脸区域
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 0))
        # 检测人脸特征点
        shape = predictor(gray, rect)
        # 将检测到的特征点转换成numpy数组
        points = face_utils.shape_to_np(shape)
        # 提取左右眼特征点
        lefteye = points[42: 48]
        righteye = points[36: 42]
        # 计算左右眼纵横比
        left_ear = eye_aspect_ratio(lefteye)
        right_ear = eye_aspect_ratio(righteye)
        ear = (left_ear+right_ear)/2
        # 判断是否眨眼
        if ear > 0.2:
            frame_counter += 1
        else:
            if ear < 0.15:
                if frame_counter >= 3:
                    blink_count += 1
                frame_counter = 0
        # 在图像中绘出特征点
        for i in range(0, 68):
            cv2.circle(img, (points[i][0], points[i][1]), 3, (255, 0, 255))
    cv2.putText(img, 'blinks:{0}'.format(blink_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Frame', img)
    # 按esc退出程序
    a = cv2.waitKey(1)
    if a == 27:
        break
cap.release()
cv2.destroyAllWindows()
