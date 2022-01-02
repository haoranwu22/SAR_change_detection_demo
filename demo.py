from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3],
                  [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
train_ids = TensorDataset(a, b)  # 封装数据a与标签b

# 切片输出
print(train_ids[0:2])
print('=' * 80)
#  循环取数据
for x_train, y_label in train_ids:
     print(x_train, y_label)
     # DataLoader进行数据封装
print('=' * 80)

train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
for i, data in enumerate(train_loader):
    # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0}\n x_data:{1}\nlabel: {2}'.format(i, x_data, label))
