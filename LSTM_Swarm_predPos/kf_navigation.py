import numpy as np
from scipy.integrate import ode
import time
from navigation_plugin import orbit_dynamics_j2, JD2000, ecf2sphere_mag, H1Mag2Sph, H2SpheTver
# import onnxruntime as ort
import matplotlib.pyplot as plt
# 开始计时
start_time = time.time()

# 计算地球磁场模型WMM2020的高斯球谐系数g,h
DAY_t0 = JD2000(2020, 1, 1, 0, 0, 0)
DAY_t = JD2000(2024, 8, 25, 0, 0, 0)
DAY_decimal = (DAY_t - DAY_t0) / 365

# 读取WMM2020文件中的高斯球谐系数
data = np.loadtxt('WMM2020.txt')
n, m, gvali, hvali, gsvi, hsvi = data.T
g = np.zeros((12, 13))
h = np.zeros((12, 13))
for i in range(len(n)):
    g[int(n[i])-1, int(m[i])] = gvali[i] + gsvi[i] * DAY_decimal
    h[int(n[i])-1, int(m[i])] = hvali[i] + hsvi[i] * DAY_decimal

# 读取swarm卫星轨道数据
swarm_data = np.loadtxt('swarm_20240825.txt')
n_s, Bx_s, By_s, Bz_s, x_f, y_f, z_f, vx_f, vy_f, vz_f = swarm_data.T
Br = np.column_stack((Bx_s, By_s, Bz_s))
# detaB_model = np.load('detaB_model_0825.npy', allow_pickle=True)

# 仿真的初始值和条件
len_sim = 19999
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
detaR = np.zeros((3, len_sim))
Vd = np.zeros(len_sim)
detaV = np.zeros((3, len_sim))

# 导入ONNX模型
# model = ort.InferenceSession('model.onnx')

# 卡尔曼滤波算法
for i in range(len_sim):
    # 状态一步预测
    solver = ode(orbit_dynamics_j2).set_integrator('dopri5', atol=1e-8, rtol=1e-8)
    solver.set_initial_value(X[:, i], i).integrate(i + 1)
    xe = solver.y
    Re = np.linalg.norm(xe[:3])

    # 计算当地地磁场矢量
    lamda_center = np.arctan2(xe[1], xe[0])  # 地心经度
    sita_center = np.arcsin(xe[2] / Re)  # 地心纬度
    Bx, By, Bz, H_m, D_m, I_m, F_m = ecf2sphere_mag(xe[0], xe[1], xe[2], 12, g, h)  # WMM模型阶数12

    # if i >= 60:
    #     model_path = 'E:/ai_software/LSTM_Swarm/LSTM_Swarm_predPos/model/predPostrain20240825.pkl'
    #     traindata1 = X[:3, i - 60 + 1:i].T
    #     traindata2 = Br[i - 60 + 1:i, :]
    #     traindata = np.hstack((traindata1, traindata2))
    tempdetaB = Br[i + 1, :] - np.array([Bx, By, Bz])
    detaB[:, i] = Br[i + 1, :] - np.array([Bx, By, Bz])

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
plt.grid(True)  # 打开网格
plt.show()  # 显示图形
