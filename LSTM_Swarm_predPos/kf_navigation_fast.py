import numpy as np
from scipy.integrate import ode
import time
from navigation_plugin import orbit_dynamics_j2, JD2000, ecf2sphere_mag, H1Mag2Sph, H2SpheTver
import concurrent.futures  # 并行计算
import matplotlib.pyplot as plt
# import onnxruntime as ort  # ONNX推理

# 开始计时
start_time = time.time()

# 计算地球磁场模型WMM2020的高斯球谐系数g,h
DAY_t0 = JD2000(2020, 1, 1, 0, 0, 0)
DAY_t = JD2000(2024, 8, 25, 0, 0, 0)
DAY_decimal = (DAY_t - DAY_t0) / 365

# 读取WMM2020文件中的高斯球谐系数，并转换为CuPy数组
data = np.loadtxt('WMM2020.txt')  # 使用 NumPy 加载数据
n, m, gvali, hvali, gsvi, hsvi = data.T
g = np.zeros((12, 13))
h = np.zeros((12, 13))
for i in range(len(n)):
    g[int(n[i])-1, int(m[i])] = gvali[i] + gsvi[i] * DAY_decimal
    h[int(n[i])-1, int(m[i])] = hvali[i] + hsvi[i] * DAY_decimal

# 读取swarm卫星轨道数据并转换为CuPy数组
swarm_data = np.loadtxt('swarm_20240825.txt')  # 使用 NumPy 加载数据
n_s, Bx_s, By_s, Bz_s, x_f, y_f, z_f, vx_f, vy_f, vz_f = swarm_data.T
Br = np.column_stack((Bx_s, By_s, Bz_s))

# 仿真的初始值和条件
len_sim = 86399
X = np.zeros((6, len_sim))  # 系统状态量
P = np.zeros((6, 6, len_sim))  # 估计均方误差阵
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

# 使用ONNX GPU推理
# ort_sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])

def kalman_filter_step(i, xe, Re, detaB, Br):
    # 计算当地地磁场矢量
    lamda_center = np.arctan2(xe[1], xe[0])  # 地心经度
    sita_center = np.arcsin(xe[2] / Re)  # 地心纬度

    # 计算雅克比矩阵F
    F41 = -miu / Re ** 3 + Wez ** 2 + 3 * miu * xe[0] ** 2 / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * ((Re ** 2 - 5 * xe[0] ** 2) / Re ** 7 - 5 * xe[2] ** 2 * (Re ** 2 - 7 * xe[0] ** 2) / Re ** 9)
    F42 = 3 * miu * xe[0] * xe[1] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (-5 * xe[0] * xe[1] / Re ** 7 + 35 * xe[0] * xe[1] * xe[2] ** 2 / Re ** 9)
    F43 = 3 * miu * xe[0] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (-5 * xe[0] * xe[2] / Re ** 7 - 5 * xe[0] * xe[2] * (2 * Re ** 2 - 7 * xe[2] ** 2) / Re ** 9)
    F44 = 0
    F45 = 2 * Wez
    F46 = 0

    # F51, F52, F53, F54, F55, F56
    F51 = 3 * miu * xe[0] * xe[1] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (-5 * xe[0] * xe[1] / Re ** 7 + 35 * xe[0] * xe[1] * xe[2] ** 2 / Re ** 9)
    F52 = -miu / Re ** 3 + Wez ** 2 + 3 * miu * xe[1] ** 2 / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * ((Re ** 2 - 5 * xe[1] ** 2) / Re ** 7 - 5 * xe[2] ** 2 * (Re ** 2 - 7 * xe[1] ** 2) / Re ** 9)
    F53 = 3 * miu * xe[1] * xe[2] / Re ** 5 - 0.5 * (3 * miu * J2 * R_p ** 2) * (-5 * xe[1] * xe[2] / Re ** 7 - 5 * xe[1] * xe[2] * (2 * Re ** 2 - 7 * xe[2] ** 2) / Re ** 9)
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

    # 计算量测矩阵Ht
    y1 = H1Mag2Sph(Re, np.pi / 2 - sita_center, lamda_center, 12, g, h)
    y2 = H2SpheTver(np.pi / 2 - sita_center, lamda_center, Re)
    Ht = np.hstack([y1 @ y2, np.zeros((3, 3))])

    # kalman 滤波算法递推估计
    Pt = Fai @ P[:, :, i] @ Fai.T + Q
    K = Pt @ Ht.T @ np.linalg.inv(Ht @ Pt @ Ht.T + R)
    X[:, i + 1] = xe + K @ detaB[:, i]
    P[:, :, i + 1] = (np.eye(6) - K @ Ht) @ Pt @ (np.eye(6) - K @ Ht).T + K @ R @ K.T

    return X[:, i + 1], P[:, :, i + 1]

# 并行执行卡尔曼滤波步骤
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i in range(len_sim):
        solver = ode(orbit_dynamics_j2).set_integrator('dop853', atol=1e-6, rtol=1e-6)  # 更快的积分器
        solver.set_initial_value(X[:, i], i).integrate(i + 1)
        xe = solver.y
        Re = np.linalg.norm(xe[:3])

        Bx, By, Bz, H_m, D_m, I_m, F_m = ecf2sphere_mag(xe[0], xe[1], xe[2], 12, g, h)  # WMM模型阶数12
        detaB[:, i] = Br[i + 1, :] - np.array([Bx, By, Bz])
        # 并行执行Kalman滤波步骤
        futures.append(executor.submit(kalman_filter_step, i, xe, Re, detaB, Br))

    # 获取并行结果
    for i, future in enumerate(futures):
        X[:, i + 1], P[:, :, i + 1] = future.result()

# 结束计时
print(f"运行时间: {time.time() - start_time} 秒")

# 绘制位置误差曲线
plt.figure()
plt.plot(range(0, len_sim), Rd, 'k')  # Rd 已经是CuPy数组，不需要索引0
plt.xlabel('仿真时间/s')
plt.ylabel('位置误差/km')
plt.grid(True)
plt.show()
