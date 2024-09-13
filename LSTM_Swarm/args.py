from collections import namedtuple

def args_parser():
    Args = namedtuple('Args', ['input_size', 'hidden_size', 'num_layers', 'output_size', 'batch_size', 'optimizer','lr','weight_decay','step_size','gamma','epochs','inout_mode'])
    args = Args(
        input_size = 3,
        hidden_size = 20,
        num_layers = 1,
        output_size = 3,
        batch_size =100,
        optimizer = 'RMSprop',#RMSprop, adam, adamW
        lr = 0.005,#学习率，控制每次参数更新的步长大小
        weight_decay = 1e-5,#L2正则化的系数（也叫权重衰减）。它用于在优化过程中对权重施加惩罚，防止过拟合
        step_size = 10,#每隔多少个训练周期（epochs）就更新一次学习率
        gamma = 0.95,#缩放因子，如果 gamma 设置为 0.1，那么每隔 step_size 个周期，学习率将变成原来的 0.1 倍。
        epochs = 500,
        inout_mode = 'MIMO'#MISO
    )
    return args
#预测误差标准差为: [37.96527  27.661133 52.924072]
#模型误差标准差为: [52.213047 50.74241  54.248318]