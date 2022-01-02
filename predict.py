import torch
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn
# import torch.optim as optim
from model import Batch_Net
import numpy as np

# 读取保存的测试集的数据
print("读取数据......")
test_feature = np.load('params/test_feature.npy')
test_label = np.load('params/test_label.npy')

# 转化为张量，并转换为float32
test_feature = torch.from_numpy(test_feature)
test_label = torch.from_numpy(test_label)

# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
deal_dataset = TensorDataset(test_feature, test_label)

print("数据读取完毕")
print('=' * 80)

# 数据装载
print("开始装载数据......")
test_loader = DataLoader(dataset=deal_dataset,
                         batch_size=48,
                         shuffle=True,
                         # num_workers加载数据的时候使用几个子进程
                         # num_workers=2
                         )
print("装载完毕")
print('=' * 80)
classes = ('0', '1')

model = Batch_Net(test_feature.shape[1], 50, 250, 2)
# 加载预训练模型的参数
model.load_state_dict(torch.load("trained_net/net.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 网络保存到GPU


def test():
    correct = 0
    total = 0
    # 因为是预测只有前向传播，不需要保存梯度
    with torch.no_grad():
        # 迭代完毕后就遍历完测试集
        for data in test_loader:
            inputs, target = data
            # fc层接收数据类型为float32
            inputs = inputs.to(torch.float32)
            target = target.long()

            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)  # 计算总样本数
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %.5f %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    test()
