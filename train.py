import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import Batch_Net
import numpy as np
import matplotlib.pyplot as plt


# # 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
# class GetLoader(torch.utils.data.Dataset):
#     # 初始化函数，得到数据
#     def __init__(self, data_root, data_label):
#         self.data = data_root
#         self.label = data_label

#     # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
#     def __getitem__(self, index):
#         data = self.data[index]
#         labels = self.label[index]
#         return data, labels

#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
#     def __len__(self):
#         return len(self.data)


# 读取保存的训练集的数据
print("读取数据......")
train_feature = np.load('params/train_feature.npy')
train_label = np.load('params/train_label.npy')

# 转化为张量，并转换为float32
train_feature = torch.from_numpy(train_feature)
train_label = torch.from_numpy(train_label)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
deal_dataset = TensorDataset(train_feature, train_label)
# # 切片输出
# print(deal_dataset[0:2])
print("数据读取完毕")
print('=' * 80)

# '''加上transform'''
# print("数据预处理")
# train_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307, ), (0.3081, ))
#             ])
# # 数据预处理
# deal_dataset = train_transform(deal_dataset)
# print("数据预处理完毕")

# 数据装载
print("开始装载数据......")
train_loader = DataLoader(dataset=deal_dataset,
                          batch_size=48,
                          shuffle=True,
                          # num_workers加载数据的时候使用几个子进程
                          # num_workers=2
                          )
print("装载完毕")
print('=' * 80)
classes = ('0', '1')
# 因为最后是变与不变两类，所以out_dim = 2
net = Batch_Net(train_feature.shape[1], 50, 250, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)  # 网络保存到GPU
train_loss_list = []  # 画图用的误差


def train(epoch):
    running_loss = 0.0
    # 每次遍历输入的是一个batch
    # for循环迭代完毕是一个1个epoch
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # fc层接收数据类型为float32
        inputs = inputs.to(torch.float32)
        target = target.long()

        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()  # 清除上次的积累
        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        # update weights
        optimizer.step()
        # 在训练时统计loss变化时，会用到loss.item()，能够防止tensor无线叠加导致的显存爆炸
        # loss.item()应该是一个batch的平均损失
        running_loss += loss.item()
        # 每300个batch打印一次loss
        if batch_idx % 1000 == 999:
            # running_loss除以300是为了算300个batch的平均损失
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 1000))
            train_loss_list.append(running_loss / 1000)
            running_loss = 0.0


if __name__ == '__main__':
    print("开始训练模型......")
    for epoch in range(5):
        train(epoch)
    # 保存训练的模型：
    print("模型训练完毕!")
    print('=' * 80)
    print("绘制loss曲线")
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list, 'b', label="loss")
    plt.xlabel("iter_num")
    plt.ylabel("loss")
    plt.title("train_loss")
    plt.legend()
    plt.show()
    print("开始保存模型")
    torch.save(net.state_dict(), "trained_net/net.pth")
    print("模型保存完毕!")
