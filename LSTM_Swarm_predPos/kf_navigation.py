import numpy as np
import time
import torch
from scipy.integrate import ode
from navigation_plugin import orbit_dynamics_j2, JD2000, ecf2sphere_mag, H1Mag2Sph, H2SpheTver, JD2000_
from predPosB2error import predPosB2error
# import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd

# 开始计时
start_time = time.time()

# 计算地球磁场模型WMM2020的高斯球谐系数g,h
year = 2024
month = 9
day = 5
DAY_t0 = JD2000(2020, 1, 1, 0, 0, 0)
DAY_t = JD2000(2024, 9, 5, 0, 0, 0)
DAY_decimal = (DAY_t - DAY_t0) / 365

#读训练文件
def load_data(file_name):
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    df.fillna(df.mean(), inplace=True)#将NAN替换成每列均值
    return df
test_dataset = load_data('predPosTrain20240905.csv')
test_load = test_dataset.iloc[:, :].values
label = test_load[:,13:16]

# 读取WMM2020文件中的高斯球谐系数
data = np.loadtxt('WMM2020.txt')
wmm_n, wmm_m, gvali, hvali, gsvi, hsvi = data.T
g = np.zeros((12, 13))
h = np.zeros((12, 13))
for i in range(len(wmm_n)):
    g[int(wmm_n[i])-1, int(wmm_m[i])] = gvali[i] + gsvi[i] * DAY_decimal
    h[int(wmm_n[i])-1, int(wmm_m[i])] = hvali[i] + hsvi[i] * DAY_decimal

# 读取swarm卫星轨道数据
swarm_data = np.loadtxt('data/' + 'swarm 20240905.txt')
n_s, Bx_s, By_s, Bz_s, x_f, y_f, z_f, vx_f, vy_f, vz_f = swarm_data.T
Br = np.column_stack((Bx_s, By_s, Bz_s))
# detaB_model = np.loadtxt('data/' + 'detaB_model.txt', delimiter=',', dtype=np.float64).T

model_path = 'E:/ai_software/LSTM_Swarm/LSTM_Swarm_predPos/model/predPosTrain20240825_20240831_7.pkl'

# 仿真的初始值和条件
len_sim = 57999
X = np.zeros((6, len_sim+1))  # 系统状态量
P = np.zeros((6, 6, len_sim+1))  # 估计均方误差阵
Pt = np.zeros((6, 6))  # 一步预测均方误差阵
K = np.zeros((6, 3))  # 滤波增益矩阵
J2 = 0.00108263
R_p = 6378.137  # 地球赤道平均半径，单位km
miu = 3.986004418e5  # 地球引力常数
Wez = 2 * np.pi / 86164  # 地球自转角速度
T = 1  # 采样周期，单位s
P[:, :, 0] = np.diag([20 ** 2, 20 ** 2, 20 ** 2, 0.2 ** 2, 0.2 ** 2, 0.2 ** 2])
X[:, 0] = [x_f[0] + 20, y_f[0] + 20, z_f[0] - 20, vx_f[0] - 0.2, vy_f[0] + 0.2, vz_f[0] - 0.2]
Q = np.diag([1e-6 ** 2] * 6)
R = np.diag([20 ** 2] * 3)
detaB = np.zeros((3, len_sim))
Rd = np.zeros(len_sim)
detaB_2 = np.zeros(len_sim)
detaR = np.zeros((3, len_sim))
Vd = np.zeros(len_sim)
detaV = np.zeros((3, len_sim))
detaB_model = np.array([0,0,0])
detaB_model_seq = np.zeros((len_sim+1, 3))
detaB_model_seq[0,:] = detaB_model
B_wmm = np.zeros((len_sim+1, 3))
Bx, By, Bz, H_m, D_m, I_m, F_m = ecf2sphere_mag(X[0, 0], X[1, 0], X[2, 0], 12, g, h)
B_wmm[0,:] = [Bx, By, Bz]
JDtime = np.zeros(len_sim)
# m = np.array([6848.64575, 6848.278586, 6826.249344, 32780.2716, 12313.3102, 48739.0162, 32854.00103, 12222.92653, 48726.94414, 6845.208253, 6845.220864, 6825.495695, 572.2563781, 3192.003036, 226.2260618])
# n = np.array([-6848.846862, -6846.401115, -6846.957215, -11573.6838, -12775.5389, -52494.6381, -11503.10242, -12775.24134, -52479.77697, -6843.566714, -6844.326181, -6844.151817, -22364.90198, -1075.734999, -25131.76197])

m = np.array([6835.427252,   6781.312654 ,  6829.351228 , 32699.3376, 12328.7433  ,  48617.3694  ,  32773.88052   ,12213.36506  , 48563.0042,  6829.928918 ,  6774.021365  , 6827.990067  ,  698.5235691 ,  965.0445525,   204.3129807])
n = np.array([-6820.66282   , -6849.905062 ,  -6846.727315, -11661.8053  ,  -12786.2691  ,  -52088.668   ,  -11499.04219, -12724.30192 ,  -52114.23552 ,   -6819.190517  , -6846.877078,  -6845.542857  , -1151.306063  ,  -558.1916223   ,-229.62014  ])
# 卡尔曼滤波算法
for i in range(len_sim):
    JDtime[i] = JD2000_(year,month,day,i)
    DAY_decimal=(JDtime[i]-DAY_t0)/365
    for ii in range(len(n)):
        g[int(wmm_n[ii]) - 1, int(wmm_m[ii])] = gvali[ii] + gsvi[ii] * DAY_decimal
        h[int(wmm_n[ii]) - 1, int(wmm_m[ii])] = hvali[ii] + hsvi[ii] * DAY_decimal
    # 状态一步预测
    # solver = ode(orbit_dynamics_j2).set_integrator('dopri5', atol=1e-8, rtol=1e-8)
    solver = ode(orbit_dynamics_j2).set_integrator('dop853', atol=1e-6, rtol=1e-6)
    solver.set_initial_value(X[:, i], i).integrate(i + 1)
    xe = solver.y
    Re = np.linalg.norm(xe[:3])

    # 计算当地地磁场矢量
    lamda_center = np.arctan2(xe[1], xe[0])  # 地心经度
    sita_center = np.arcsin(xe[2] / Re)  # 地心纬度
    Bx, By, Bz, H_m, D_m, I_m, F_m = ecf2sphere_mag(xe[0], xe[1], xe[2], 12, g, h)  # WMM模型阶数12
    B_wmm[i+1,:] = [Bx, By, Bz]
    # if i >= 60:
    #     traindata1 = X[:3, i - 60:i].T
    #     traindata2 = Br[i - 60:i, :]
    #     traindata3 = B_wmm[i - 60:i, :]
    #     # traindata = np.hstack((traindata1, traindata2, traindata3))
    #     traindata = np.hstack((traindata1, traindata2))
    #     # traindata = traindata.reshape(1, 60, 9)
    #     traindata = traindata.reshape(1, 60, 6)
    #     traindata = (traindata - n[0:6]) / (m[0:6] - n[0:6])
    #     traindata = torch.FloatTensor(traindata)
    #     detaB_model = predPosB2error(traindata, model_path, m, n)

    # tempBr = Br[i + 1, :]
    # tempBs = np.array([Bx, By, Bz])
    detaB_model_seq[i+1,:] = detaB_model
    detaB_model = label[i+1,:]
    tempdetaB = Br[i + 1, :] - np.array([Bx, By, Bz]) - detaB_model
    detaB[:, i] = Br[i + 1, :] - np.array([Bx, By, Bz]) - detaB_model
    # detaB[:, i] = Br[i + 1, :] - np.array([Bx, By, Bz]) - detaB_model[i + 1,:]
    detaB_2[i] = np.sqrt(detaB[0,i] ** 2 + detaB[1,i] ** 2 + detaB[2,i] ** 2)

    # 计算雅克比矩阵F
    F41 = -miu / Re ** 3 + Wez ** 2 + 3 * miu * xe[0] ** 2 / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                (Re ** 2 - 5 * xe[0] ** 2) / Re ** 7 - 5 * xe[2] ** 2 * (Re ** 2 - 7 * xe[0] ** 2) / Re ** 9)
    F42 = 3 * miu * xe[0] * xe[1] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -5 * xe[0] * xe[1] / Re ** 7 + 35 * xe[0] * xe[1] * xe[2] ** 2 / Re ** 9)
    F43 = 3 * miu * xe[0] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -5 * xe[0] * xe[2] / Re ** 7 - 5 * xe[0] * xe[2] * (2 * Re ** 2 - 7 * xe[2] ** 2) / Re ** 9)
    F44 = 0
    F45 = 2 * Wez
    F46 = 0

    # F51, F52, F53, F54, F55, F56
    F51 = 3 * miu * xe[0] * xe[1] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -5 * xe[0] * xe[1] / Re ** 7 + 35 * xe[0] * xe[1] * xe[2] ** 2 / Re ** 9)
    F52 = -miu / Re ** 3 + Wez ** 2 + 3 * miu * xe[1] ** 2 / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                (Re ** 2 - 5 * xe[1] ** 2) / Re ** 7 - 5 * xe[2] ** 2 * (Re ** 2 - 7 * xe[1] ** 2) / Re ** 9)
    F53 = 3 * miu * xe[1] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -5 * xe[1] * xe[2] / Re ** 7 - 5 * xe[1] * xe[2] * (2 * Re ** 2 - 7 * xe[2] ** 2) / Re ** 9)
    F54 = -2 * Wez
    F55 = 0
    F56 = 0

    # F61, F62, F63, F64, F65, F66
    F61 = 3 * miu * xe[0] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -15 * xe[0] * xe[2] / Re ** 7 + 35 * xe[0] * xe[2] ** 3 / Re ** 9)
    F62 = 3 * miu * xe[1] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                -15 * xe[1] * xe[2] / Re ** 7 + 35 * xe[1] * xe[2] ** 3 / Re ** 9)
    F63 = -miu / Re ** 3 + 3 * miu * xe[2] ** 2 / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (
                3 * (Re ** 2 - 5 * xe[2] ** 2) / Re ** 7 - 5 * xe[2] ** 2 * (
                    3 * Re ** 2 - 7 * xe[2] ** 2) / Re ** 9)
    F64 = 0
    F65 = 0
    F66 = 0

    F = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [F41, F42, F43, F44, F45, F46],
                  [F51, F52, F53, F54, F55, F56],
                  [F61, F62, F63, F64, F65, F66]])

    # 计算系统状态转移矩阵Fai
    Fai = np.eye(6) + F * T

    # 构造量测矩阵Ht
    y1 = H1Mag2Sph(Re, np.pi / 2 - sita_center, lamda_center, 12, g, h)
    y2 = H2SpheTver(np.pi / 2 - sita_center, lamda_center, Re)
    Ht = np.hstack([y1 @ y2, np.zeros((3, 3))])

    # kalman 滤波算法递推估计
    Pt = Fai @ P[:, :, i] @ Fai.T + Q
    K = Pt @ Ht.T @ np.linalg.inv(Ht @ Pt @ Ht.T + R)
    X[:, i + 1] = xe + K @ detaB[:, i]
    P[:, :, i + 1] = (np.eye(6) - K @ Ht) @ Pt @ (np.eye(6) - K @ Ht).T + K @ R @ K.T

    # 统计位置误差detaR
    xd = X[0, i + 1] - x_f[i + 1]
    yd = X[1, i + 1] - y_f[i + 1]
    zd = X[2, i + 1] - z_f[i + 1]
    tempRd = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)
    Rd[i] = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)
    detaR[:, i] = np.array([xd, yd, zd])

    # 统计速度误差detaV
    vxd = X[3, i + 1] - vx_f[i + 1]
    vyd = X[4, i + 1] - vy_f[i + 1]
    vzd = X[5, i + 1] - vz_f[i + 1]
    Vd[i] = np.sqrt(vxd ** 2 + vyd ** 2 + vzd ** 2)
    detaV[:, i] = np.array([vxd, vyd, vzd])
    print('epoch {:03d} '.format(i))

# 结束计时
print(f"运行时间: {time.time() - start_time} 秒")

# 统计误差均方根
# LEN = range(5000, len_sim)
# a1, a2, a3 = np.std(detaR[0, LEN]), np.std(detaR[1, LEN]), np.std(detaR[2, LEN])
# b1 = np.array([a1, a2, a3])
# a4 = np.linalg.norm(b1)
#
# a5, a6, a7 = np.std(detaV[0, LEN]), np.std(detaV[1, LEN]), np.std(detaV[2, LEN])
# b2 = np.array([a5, a6, a7])
# a8 = np.linalg.norm(b2)

# print(f"总位置误差均值: {np.mean(Rd[LEN])} km")
# print(f"总速度误差均值: {np.mean(Vd[LEN])} m/s")

# 位置误差均方根
# LEN = range(20000, len_sim)
# print(f"位置误差均方根: {np.std(Rd[LEN])} km")
# print(f"速度误差均方根: {np.std(Vd[LEN])} m/s")


# 绘制位置误差曲线
plt.figure()  # 创建图形窗口
plt.plot(range(0, len_sim), Rd, 'k')  # 绘制 Rd[0, :] 的数据
plt.xlabel('time/s')  # 设置x轴标签
plt.ylabel('Pos Error/km')  # 设置y轴标签

plt.figure()  # 创建图形窗口
plt.plot(range(0, len_sim), detaB_2, 'b')  # 绘制 Rd[0, :] 的数据
plt.xlabel('time/s')  # 设置x轴标签
plt.ylabel('B Error/km')  # 设置y轴标签

# 创建子图：每列数据一个图
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# 每个子图的标题
titles = ['B_N', 'B_E', 'B_C']
for i in range(3):
    axes[i].plot(label[:len_sim, i], label='label', color='blue')
    axes[i].plot(detaB_model_seq[:, i], label='predict', color='orange')
    # axes[i].plot(dte_wl[:, i], label='wmm2020', color='yellow')
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('time')
    axes[i].set_ylabel('nT')
    axes[i].legend()

plt.grid(True)  # 打开网格
plt.show()  # 显示图形