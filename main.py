import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def cv_show(img, name='img'):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def grayContrast(img, fa, fb):
    img = img.copy()
    k = 255 / (fb - fa)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] < fa:
                img[i, j] = 0
            elif img[i, j] > fb:
                img[i, j] = 255
            else:
                img[i, j] = k * (img[i, j] - fa)
    return img


def transform(image):
    orig = image.copy()
    image = orig
    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    kernal = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernal, iterations=3)

    blur = cv2.GaussianBlur(erosion, (7, 7), 0)

    edged = cv2.Canny(blur, 75, 200)

    # 轮廓检测
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # C表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if peri < 500:
            return []
        # 4个点的时候就拿出来
        if len(approx) == 4:
            return approx
    return []


def process(warped):
    # 转灰度图像
    img2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 提高对比度
    img2 = grayContrast(img2, 50, 100)
    # cv_show(img2)

    # 取中间1行像素点进行处理
    imgShape = img2.shape
    img = img2[(imgShape[0] // 2 - 1):(imgShape[0] // 2), :]
    imgShape = img.shape

    # 非白即黑，这是因为提高对比度之后的图像，有些线是灰色的
    ret, thresh = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    img = thresh
    # for i in range(0, imgShape[1]):
    #     if img[0, i] != 255:
    #         img[0, i] = 0

    # 有些条形码四周有黑框要把他们和条形码区别，算相邻距离即可
    blackStartLeft, blackStartRight = [], []
    for k in range(1, imgShape[1] // 2):  # 从左向右扫描
        if img[0, k] == 0 and img[0, k - 1] == 255:
            blackStartLeft.append(k)
    for k in range(imgShape[1] - 2, imgShape[1] // 2, -1):  # 从右向左扫描
        if img[0, k] == 0 and img[0, k + 1] == 255:
            blackStartRight.append(k)
    disBetweenBlackLeft = np.diff(blackStartLeft)
    disBetweenBlackRight = np.diff(blackStartRight)

    # 结果列表
    result = []
    if disBetweenBlackLeft != [] and disBetweenBlackRight != []:
        if disBetweenBlackLeft[0] - disBetweenBlackLeft[1] > 25:
            left = blackStartLeft[1]
        else:
            left = blackStartLeft[0]
        if disBetweenBlackRight[1] - disBetweenBlackRight[0] > 25:
            right = blackStartRight[1]
        else:
            right = blackStartRight[0]
        img = img[:, left:right]  # 去除黑框，得到条形码结果

        # 下面的工作就是识别条形码，为了便于后面分割，先把y长度resize为95
        imgShape = img.shape
        img = cv2.resize(img, dsize=(95, imgShape[0]))
        imgShape = img.shape

        # 等分95份
        offset = int(imgShape[1] / 95)  # 计算偏移量，其实也就是1
        for j in range(offset, imgShape[1] + 1):
            if j % offset == 0:
                ratio = np.sum(img[:, j - offset:j]) / (255 * offset)
                if ratio >= 0.3:
                    result.append(0)
                else:
                    result.append(1)
        # print(result)
        if result[:3] == [1, 0, 1] or result[-3:] == [1, 0, 1]:  # 条形码的起始符和终止符必须正确
            return True, result[3:-3]
        else:
            return False, result
    return False, result


def decode(result):
    # 编码表
    singularCharDict = {'0001101': 0, '0011001': 1, '0010011': 2,
                        '0111101': 3, '0100011': 4, '0110001': 5,
                        '0101111': 6, '0111011': 7, '0110111': 8,
                        '0001011': 9
                        }
    evenCharDict = {'0100111': 0, '0110011': 1, '0011011': 2,
                    '0100001': 3, '0011101': 4, '0111001': 5,
                    '0000101': 6, '0010001': 7, '0001001': 8,
                    '0010111': 9
                    }
    evenCharDict2 = {'1110010': 0, '1100110': 1, '1101100': 2,
                     '1000010': 3, '1011100': 4, '1001110': 5,
                     '1010000': 6, '1000100': 7, '1001000': 8,
                     '1110100': 9}
    firstNumberDict = {'000000': 0, '001011': 1, '001101': 2,
                       '001110': 3, '010011': 4, '011001': 5,
                       '011100': 6, '010101': 7, '010110': 8,
                       '011010': 9}
    # 最终的数组与记录奇偶性的列表
    number, oddAndEven = [], []
    for i in range(7, 42 + 1):  # 左侧数据区
        if i % 7 == 0:
            tempList = list(map(lambda x: str(x), result[i - 7:i]))  # 设置为str类型元素的列表
            tempStr = ''.join(tempList)
            if not singularCharDict.get(tempStr):  # 非None值
                oddAndEven.append('1')
            if not evenCharDict.get(tempStr):
                oddAndEven.append('0')
            number.append(singularCharDict.get(tempStr) or evenCharDict.get(tempStr))

    for i in range(7, 42 + 1):  # 右侧数据区
        if i % 7 == 0:
            tempList = list(map(lambda x: str(x), result[47 + i - 7:47 + i]))
            tempStr = ''.join(tempList)
            number.append(evenCharDict2.get(tempStr))

    number.insert(0, firstNumberDict.get(''.join(oddAndEven)))  # 插入奇偶位，也就是第0位
    number = ['X' if item == 'None' else item for item in list(map(lambda x: str(x), number))]  # 将所有的None值替换为X
    return number


def main():
    capture = cv2.VideoCapture(0)
    number = ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    if capture.isOpened():  # 检查是否打开正确
        openSucess, frame = capture.read()
    else:
        openSucess = False

    while openSucess:
        ret, frame = capture.read()
        if frame is None:
            break
        if ret:
            # 启动计时器
            timeStart = time.time()
            # 先进行轮廓检测和投射变换
            screenCnt = transform(frame)
            ContourDetect = 'False'
            SignDetect = 'False'
            if screenCnt != []:
                ContourDetect = 'True'
                cv2.drawContours(frame, [screenCnt], -1, (255, 0, 0), 2)  # 绘制轮廓
                warped = four_point_transform(frame.copy(), screenCnt.reshape(4, 2))  # 投射变换
                state, result = process(warped)  # 得到投射变换图之后进行图像处理，如果最后条形码起始符和终止符不是101，图片里没有条形码返回false
                if state:
                    numberGenList = decode(result)  # decode函数对result里编码进行解码
                    SignDetect = 'True'
                    # 替换掉X
                    number = [numberGenList[i] if item == 'X' and numberGenList[i] != 'X' else item for i, item in
                              enumerate(number)]
                    # 替换上一次结果
                    number = [numberGenList[i] if item != 'X' and numberGenList[i] != 'X' else item for i, item in
                              enumerate(number)]
            # 翻转镜头
            # frame = cv2.flip(frame, 180)
            timeEND = time.time()
            FRAME = 1 / (timeEND - timeStart)
            cv2.putText(frame, 'FPS:{:.2f}'.format(FRAME), (0, 22), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.putText(frame, 'EAN13 Code:' + ''.join(number), (0, 44), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

            cv2.putText(frame, 'Contour:' + ContourDetect, (0, 445), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.putText(frame, 'Sign:' + SignDetect, (0, 475), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            # frameShape = frame.shape
            # frame = cv2.resize(frame, (int(frameShape[1] * 1.4), int(frameShape[0] * 1.2)))
            cv2.rectangle(frame, (140, 420), (510, 50), (211, 50, 148), 2)
            cv2.imshow('Book Barcode Recognition by Wang Yi', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

    capture.release()
    cv2.destroyAllWindows()


main()
