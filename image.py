import pandas as pd
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import image
import matplotlib.cm as cm
import scipy
import cv2
import scipy.io as sio
from iterative_threshold import best_thresh
import seaborn as sns


# 原始数据四周补0

def pad_data(data, nei_size):
    m, n = data.shape
    t1 = np.zeros([nei_size//2, n])
    data = np.concatenate((t1, data, t1), axis=0)  # 列方向补零
    m, n = data.shape
    t2 = np.zeros([m, nei_size//2])
    data = np.concatenate((t2, data, t2), axis=1)  # 行方向补零
    return data


# 逐像素取大小为nei_size*nei_size的邻域数据
def gen_dataX(data, nei_size):
    x, y = data.shape
    m = x-nei_size//2*2  # //表示除法向下取整
    n = y-nei_size//2*2  # 得到填充前的图片数组大小m,n
    res = np.zeros([m*n, nei_size**2])
    k = 0
    for i in range(nei_size//2, m+nei_size//2):  # 按行取像素
        for j in range(nei_size//2, n+nei_size//2):  # i,j遍历原始图像的位置
            res[k, :] = np.reshape(data[i-nei_size//2:i+nei_size//2+1, j-nei_size//2:j+nei_size//2+1].T, (1, -1))
            k += 1
    return res


# 读取图像
# 三通道的图像转为单通道的图像
img_bef = cv2.imread("dataset/ArcadiaLake1986.jpg", cv2.IMREAD_GRAYSCALE)  # img1
img_rec = cv2.imread("dataset/ArcadiaLake2011.jpg", cv2.IMREAD_GRAYSCALE)  # img2
img_truth = cv2.imread("dataset/cleanchangemap.jpg", cv2.IMREAD_GRAYSCALE)  # groud truth

# 统一尺寸
img_bef = cv2.resize(img_bef, img_truth.shape, interpolation=cv2.INTER_CUBIC)
img_rec = cv2.resize(img_rec, img_truth.shape, interpolation=cv2.INTER_CUBIC)
img_truth = cv2.resize(img_truth, img_truth.shape, interpolation=cv2.INTER_CUBIC)

# opencv二值图反色处理；类标签0表示改变的像素，类标签1表示不变的像素
mask1 = np.where(img_truth == 1)
mask0 = np.where(img_truth == 0)
img_truth[mask1] = 0
img_truth[mask0] = 1
# print(img_truth)

# print(img_truth.shape)
# print(img_bef)
# print(img_bef.shape)
# print(img_rec)
# print(img_rec.shape)
# gray_bef_image = cv2.cvtColor(img_bef, cv2.COLOR_RGB2GRAY)  # 灰度化
# gray_rec_image = cv2.cvtColor(img_rec, cv2.COLOR_RGB2GRAY)  # 灰度化

# # 邻域取训练数据
# nei_size = 9
# img_full_withzero = pad_data(img_bef, nei_size)  # 四周补零
# data = gen_dataX(img_full_withzero, nei_size)
# print(data.shape)

# 计算灰度相似性矩阵
arr_add = img_bef.astype(np.float16) + img_rec.astype(np.float16)
arr_sub = img_bef.astype(np.float16) - img_rec.astype(np.float16)
zero_mask = np.where(arr_add == 0)  # 得到零元素位置
arr_add[zero_mask] = 1  # 分母为0的地方置1，防止出现计算错误0+0
S = abs(arr_sub)/(arr_add)  # 灰度相似度矩阵
# print(S)
# # 绘制原图直方图并显示最佳阈值
# plt.figure()
# plt.hist(S.ravel(), 256)
# plt.title('hist')
# plt.show()

# img1_delta = img_rec*(img_bef*img_rec)/(img_bef + img_rec)*S**2  # 图像1的方差
# img2_delta = img_bef*(img_bef*img_rec)/(img_bef + img_rec)*S**2  # 图像2的方差

pre_label = np.zeros(S.shape)

T = best_thresh(S)  # 迭代阈值法求出最佳阈值
unchange_mask = np.where(S < T)
pre_label[unchange_mask] = 1
print(pre_label)

acc = np.sum(pre_label == img_truth)/(pre_label.size)
print(acc)
# plt.imshow(pre_label, cmap=cm.gray)
# plt.show()

nei_size = 9
# 邻域取判断数据
label_full_withzero = pad_data(pre_label, nei_size)  # 四周补零
data = gen_dataX(label_full_withzero, nei_size)  # 邻域数据

# 邻域取训练数据
bef_img_full_withzero = pad_data(img_bef, nei_size)  # 四周补零
bef_data = gen_dataX(bef_img_full_withzero, nei_size)  # 邻域数据

rec_img_full_withzero = pad_data(img_rec, nei_size)  # 四周补零
rec_data = gen_dataX(rec_img_full_withzero, nei_size)  # 邻域数据

# 选取标准
equal_num = np.zeros(data.shape[0])
pre_label = pre_label.flatten()  # 将数据拉平
a = 0.6
print(data.shape)
print(pre_label.shape)
for i in range(0, data.shape[0]):
    equal_num[i] = np.sum(data[i, :] == pre_label[i])

arr_bool = equal_num/(nei_size**2) > a  # 得到符合条件的位置的布尔值

bef_data_true = bef_data[arr_bool, :]  # 取数据
rec_data_true = rec_data[arr_bool, :]
true_label = pre_label[arr_bool]

# bef_data_true = np.zeros(data.shape[1])
# rec_data_true = np.zeros(data.shape[1])
# true_label = []

# for i in range(0, pre_label.shape[1]):
#     equal_num = np.sum(data[i, :] == pre_label[0, i])
#     if((equal_num/(nei_size**2)) > a):
#         bef_data_true = np.concatenate((bef_data_true, bef_data[i, :]), axis=0)  # 列方向拼接
#         rec_data_true = np.concatenate((rec_data_true, rec_data[i, :]), axis=0)
#         true_label.append(pre_label[0, i])

# bef_data_true = np.delete(bef_data_true, 0, 0)
# rec_data_true = np.delete(rec_data_true, 0, 0)
# true_label = np.numpy(true_label)

total_data = np.concatenate((bef_data_true, rec_data_true), axis=1)  # 行方向拼接

print(total_data.shape)

train_size = 2000001

train_feature = total_data[0:train_size, :]
test_feature = total_data[train_size:, :]

train_label = true_label[0:train_size]
test_label = true_label[train_size:]

# 训练数据保存
np.save('params/train_feature', train_feature)
np.save('params/train_label', train_label)

# 测试数据保存
np.save('params/test_feature', test_feature)
np.save('params/test_label', test_label)
