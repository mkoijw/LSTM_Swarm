import torch
import torch.nn as nn
import os
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from args import args_parser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def args_parser():
#     Args = namedtuple('Args', ['input_size', 'hidden_size', 'num_layers', 'output_size', 'batch_size', 'optimizer','lr','weight_decay','step_size','gamma','epochs','inout_mode'])
#     args = Args(
#         input_size = 9,
#         hidden_size = 20,
#         num_layers = 1,
#         output_size = 3,
#         batch_size =100,
#         optimizer = 'RMSprop',#RMSprop, adam, adamW
#         lr = 0.005,#学习率，控制每次参数更新的步长大小
#         weight_decay = 1e-5,#L2正则化的系数（也叫权重衰减）。它用于在优化过程中对权重施加惩罚，防止过拟合
#         step_size = 10,#每隔多少个训练周期（epochs）就更新一次学习率
#         gamma = 0.95,#缩放因子，如果 gamma 设置为 0.1，那么每隔 step_size 个周期，学习率将变成原来的 0.1 倍。
#         epochs = 500,
#         inout_mode = 'MIMO'#MISO
#     )
#     return args
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred_ttt = self.relu(output)
        pred = pred[:, -1, :]
        return pred
def predPosB2error(data, model_path, m, n):
    args = args_parser()

    pred = []
    y = []
    input_size, output_size, hidden_size, num_layers, inout_mode= (args.input_size, args.output_size,
                                                        args.hidden_size, args.num_layers, args.inout_mode)

    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=1).to(device)

    model.load_state_dict(torch.load(model_path)['models'])
    model.eval()

    seq = data.to(device)
    with torch.no_grad():
        y_pred = model(seq)
        # pred.extend(y_pred)

    def convert_tensor_list_to_numpy(tensor_list):
        # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
        numpy_list = [tensor.cpu().numpy() for tensor in tensor_list]
        return np.array(numpy_list)

    y_pred = convert_tensor_list_to_numpy(y_pred)
    y_pred = np.array(y_pred)

    for i in range(3):
        tempm = m[i + 12]
        tempn = n[i + 12]
        y_pred[:, i] = y_pred[:, i] * (m[i + 12] - n[i + 12]) + n[i + 12]

    return y_pred

