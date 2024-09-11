import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import numpy as np
from itertools import chain
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        batch_size, seq_len = input_seq[0], input_seq[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred_ttt = self.relu(output)
        pred = pred[:, -1, :]
        return pred

def get_val_loss(model, val_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    loss_function = nn.MSELoss().to(device)
    with torch.no_grad():  # 在评估模式下不计算梯度
        for inputs, targets in val_loader:  # 遍历 DataLoader 中的每个批次
            val_loss = []
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)  # 模型预测
            # 计算损失
            loss = loss_function(out, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)  # 平均损失
    model.train()  # 恢复训练模式

    return avg_loss


def train(args, dtr, val, path):
    (input_size, output_size, hidden_size,
     num_layers,batch_size, lr, step_size, gamma) = (args.input_size, args.output_size,args.hidden_size,
                                                     args.num_layers,args.batch_size,args.lr,
                                                     args.step_size, args.gamma)

    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr,0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size, gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in dtr:
            total = len(dtr)
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()#清零优化器中所有参数的梯度。PyTorch 会在每次反向传播时累积梯度，因此需要在每个批次训练之前清零。
            loss.backward()#计算损失函数对模型参数的梯度
            optimizer.step()#根据计算出的梯度更新模型参数

        scheduler.step()#更新学习率调度器。根据设定的步长和缩放因子调整学习率。
        # validation
        val_loss = get_val_loss(model, val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
    state = {'models': best_model.state_dict()}
    torch.save(state, path)

def get_mape(x, y):
    return np.mean(np.abs((x - y) / x))

def test(args, dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, output_size, hidden_size, num_layers, inout_mode= (args.input_size, args.output_size,
                                                        args.hidden_size, args.num_layers, args.inout_mode)

    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(dte):
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            pred.extend(y_pred)

    def convert_tensor_list_to_numpy(tensor_list):
        # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
        numpy_list = [tensor.cpu().numpy() for tensor in tensor_list]
        return np.array(numpy_list)

    pred = convert_tensor_list_to_numpy(pred)
    y = np.array([tensor.numpy() for tensor in y])
    for i in range(3):
        y[:, i] = y[:, i] * (m[i + 3] - n[i + 3]) + n[i + 3]
        pred[:, i] = pred[:, i] * (m[i+3] - n[i+3]) + n[i+3]
    std_error = np.std(y-pred, axis=0)
    print("标准差为:", std_error)
    # y, pred = np.array(y), np.array(pred)
    # 创建子图：每列数据一个图
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # 每个子图的标题
    titles = ['B_N', 'B_E', 'B_C']

    for i in range(3):
        axes[i].plot(y[:, i], label='label', color='blue')
        axes[i].plot(pred[:, i], label='predict', color='orange')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('nT')
        axes[i].legend()

    # 调整布局以使子图之间的间距更合适
    plt.tight_layout()

    # 显示图形
    plt.show()