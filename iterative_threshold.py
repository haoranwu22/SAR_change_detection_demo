# 迭代阈值法分割
import numpy as np


def best_thresh(img):
    # step 1: 设置初始阈值
    img_array = np.array(img)  # 转化成数组
    zmax = np.max(img_array)
    zmin = np.min(img_array)
    tk = (zmax + zmin)/2.0
    # step 2: 根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度zo和zb
    b = 1.0
    m, n = img_array.shape
    while b == 0:
        ifg = 0
        ibg = 0
        fnum = 0.0
        bnum = 0.0
        for i in range(1, m):
            for j in range(1, n):
                tmp = img_array(i, j)
                if tmp >= tk:
                    ifg = ifg + 1
                    fnum = fnum + (tmp)  # 前景像素的个数以及像素值的总和
                else:
                    ibg = ibg + 1
                    bnum = bnum + (tmp)  # 背景像素的个数以及像素值的总和
        # step 3: 计算前景和背景的新平均值
        zo = (fnum / ifg)
        zb = (bnum / ibg)
        # step 4: 比较tk与tk+1的差值
        if abs(tk-((zo+zb) / 2.0)) < 1e-3:
            b = 1
        else:
            tk = ((zo+zb) / 2.0)
    # step 5: 返回的就是迭代计算后的阈值
    return tk
