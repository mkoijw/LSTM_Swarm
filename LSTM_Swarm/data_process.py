import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time

def load_data(file_name):
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    df.fillna(df.mean(), inplace=True)#将NAN替换成每列均值
    return df

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def nn_seq_us(args,b,file_name):
    start_time = time.time()
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    m = train.iloc[:, 0:12].max(axis=0).values  # 每列的最大值
    n = train.iloc[:, 0:12].min(axis=0).values  # 每列的最小值

    def process(data, batch_size):
        # 提取 0 到 6 列数据
        load = data.iloc[:, 0:12].values  # 使用 .values 直接获取 NumPy 数组
        wmm_load = data.iloc[:, 6:9].values
        load = (load - n) / (m - n)  # 正规化

        # 初始化序列列表
        seq = []
        wmm_label = []

        # 计算数据集大小和批次数
        seq_len = 3600*1
        step = 30

        # 通过步长创建训练序列和标签
        for i in range(0, len(data) - seq_len, step):
            # 构造 3600 步长的输入序列和对应的标签
            train_seq = load[i:i + seq_len, 0:3]
            train_label = load[i + seq_len, 9:12]
            train_wmm2020_label = wmm_load[i + seq_len, 0:3]

            # 将 NumPy 数组转换为 PyTorch 张量
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)

            seq.append((train_seq, train_label))
            wmm_label.append(train_wmm2020_label)


        # 构建 DataLoader 对象
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        # wmm_label = MyDataset(wmm_label)
        # wmm_label = DataLoader(dataset=wmm_label, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        return seq, wmm_label

    dtr, dtr_wl = process(train, b)
    val, val_wl = process(val, b)
    dte, dte_wl = process(test,b)
    print("代码块 A 耗时: {:.2f}秒".format(time.time() - start_time))
    return dtr, val, dte, m, n, dte_wl