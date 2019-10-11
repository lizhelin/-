import numpy as np
import cv2 as cv

"""
左上横坐标x0，左上纵坐标y0，裁剪宽度w、裁剪高度h
"""
crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]
"""
随机裁剪
area_ratio裁剪画面占原画面比例
hw_vari是扰动占高宽比的比例范围
"""


def random_crop(img, area_ratio, hw_vari):
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
    # 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w * np.sqrt(area_ratio * hw_mult)))
    # 裁剪宽度不能超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
    h_crop = int(round(h * np.sqrt(area_ratio * hw_mult)))
    if h_crop > h:
        h_crop = h
    # 随机生成左上角的位置
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)
    return crop_image(img, x0, y0, w_crop, h_crop)


"""
定义旋转函数：
angle是逆时针旋转的角度
crop是布尔值，表明是否要裁剪去除黑边
"""


def rotate_image(img, angle, crop):
    h, w = img.shape[:2]
    # 旋转角度的周期是360度
    angle %= 360
    # 用opencv内置函数计算仿射矩阵
    M_rotate = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv.warpAffine(img, M_rotate, (w, h))
    # 如果需要裁剪去除黑边
    if crop:
        # 对于裁剪角度的等效周期是180度
        angle_crop = angle % 180
        # 并且关于90度对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        # 转换角度为弧度
        theta = angle_crop * np.pi / 180.0
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        # 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 计算最终的边长系数
        crop_mult = numerator / denominator
        # 得到裁剪区域
        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)
        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)
        return img_rotated


"""
随机旋转
angle_vari是旋转角度的范围[-angle_vari,angle_vari)
crop是否进行去黑边操作
"""


def random_rotate(img, angle_vari, crop):
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = crop
    return rotate_image(img, angle, crop)


"""
hsv变换函数
hue_delta是色调变化比例
sat_mult是饱和度变化比例
val_mult是明度变化比例
"""


def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)


"""
随机hsv变换
hue_vari是色调变化范围
sat_vari是饱和度变化范围
val_vari是明度变化范围
"""


def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)


"""
gamma变换函数
"""


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)


"""
随机gamma变换
"""


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
