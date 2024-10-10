import numpy as np
import math

# 定义常数
miu = 3.986004418 * 10 ** 5  # 地球引力常数单位km^3/s^2
# We = 2 * np.pi / 86164  # 地球自转角速度
We = 7.2921151496 * 10 ** (-5)  # rad/s
J2 = 0.00108263  # J2项常数
Re = 6378.137  # 地球赤道的平均半径，单位km
a = 6371.2  # 地磁参考半径，单位为km

# 假设 x 是一个输入的 6 元素数组
def orbit_dynamics_j2(t,x):
    # 计算地心距
    R = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    # J2项表达式
    Fg = np.zeros(3)
    Fg[0] = J2 * (miu * x[0] / R ** 3) * (Re / R) ** 2 * (7.5 * (x[2]) ** 2 / R ** 2 - 1.5)
    Fg[1] = J2 * (miu * x[1] / R ** 3) * (Re / R) ** 2 * (7.5 * (x[2]) ** 2 / R ** 2 - 1.5)
    Fg[2] = J2 * (miu * x[2] / R ** 3) * (Re / R) ** 2 * (7.5 * (x[2]) ** 2 / R ** 2 - 4.5)

    # 微分方程组
    dx = np.zeros(6)
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = -miu * x[0] / R ** 3 + We ** 2 * x[0] + 2 * We * x[4] + Fg[0]
    dx[4] = -miu * x[1] / R ** 3 + We ** 2 * x[1] - 2 * We * x[3] + Fg[1]
    dx[5] = -miu * x[2] / R ** 3 + Fg[2]

    return dx

def JD2000(JEAR, MONTH, KDAY, JHR, MI, SEC):
    # 计算 JJ
    JJ = math.floor((14 - MONTH) / 12)

    # 计算 L
    L = JEAR - JJ - 1900 * math.floor(JEAR / 1900) + 100 * math.floor(2000 / (JEAR + 1951))

    # 计算 DAY
    DAY = KDAY - 36496 + math.floor((1461 * L) / 4) + math.floor((367 * (MONTH - 2 + JJ * 12)) / 12)

    # 加上时间部分
    DAY += ((JHR * 60 + MI) * 60 + SEC) / 86400 - 0.5

    return DAY

def JD2000_(JEAR, MONTH, KDAY, SEC):
    # 计算 JJ
    JJ = math.floor((14 - MONTH) / 12)

    # 计算 L
    L = JEAR - JJ - 1900 * math.floor(JEAR / 1900) + 100 * math.floor(2000 / (JEAR + 1951))

    # 计算 DAY
    DAY = KDAY - 36496 + math.floor((1461 * L) / 4) + math.floor((367 * (MONTH - 2 + JJ * 12)) / 12)

    # 加上时间部分
    DAY += SEC / 86400 - 0.5

    return DAY


def simit_l(sita, Nmax):
    """
    计算勒让德多项式及其导数
    参数:
    sita: 余纬角
    Nmax: 阶数截断

    返回:
    p: 勒让德多项式
    dp: 勒让德多项式的导数
    """
    N = Nmax

    # ---------------------- 带谐项 (m=0) 用 p1 表示 ------------------------------- #
    p1 = np.zeros(N + 1)  # 初始化 p1
    p1[0] = 1
    p1[1] = np.cos(sita)

    for i in range(3, N + 2):
        p1[i-1] = ((2 * i - 3) / (i - 1)) * np.cos(sita) * p1[i - 2] - ((i - 2) / (i - 1)) * p1[i - 3]

    dp1 = np.zeros(N + 1)  # 初始化 dp1
    dp1[0] = 0
    dp1[1] = -np.sin(sita)

    for i in range(3, N + 2):
        dp1[i-1] = ((2 * i - 3) / (i - 1)) * (-np.sin(sita) * p1[i - 2] + np.cos(sita) * dp1[i - 2]) - (
                    (i - 2) / (i - 1)) * dp1[i - 3]

    # ----------------------------------------------------------------------------- #

    # ---------------------- 扇谐项 (m=n) 用 p2 表示 ------------------------------- #
    p2 = np.zeros(N)  # 初始化 p2
    p2[0] = np.sin(sita)

    for i in range(2, N + 1):
        p2[i-1] = np.sqrt((2 * i - 1) / (2 * i)) * np.sin(sita) * p2[i - 2]

    dp2 = np.zeros(N)  # 初始化 dp2
    dp2[0] = np.cos(sita)

    for i in range(2, N + 1):
        dp2[i-1] = np.sqrt((2 * i - 1) / (2 * i)) * (np.cos(sita) * p2[i - 2] + np.sin(sita) * dp2[i - 2])

    # ----------------------------------------------------------------------------- #

    # ------------------------ 田谐项 (m<n) 用 p3 表示 ---------------------------- #
    p3 = np.zeros((N, N))  # 初始化 p3

    for i in range(1, N):  # n=m+1
        p3[i, i - 1] = np.sqrt(2 * i + 1) * np.cos(sita) * p2[i-1]

    for i in range(1, N + 1):
        p3[i - 1, i - 1] = p2[i - 1]  # 将 p2 项归入到 p3 项中

    dp3 = np.zeros((N, N))  # 初始化 dp3

    for i in range(1,N):  # n=m+1
        dp3[i, i-1] = np.sqrt(2 * i + 1) * (-np.sin(sita) * p2[i-1] + np.cos(sita) * dp2[i-1])

    for i in range(1,N+1):
        dp3[i-1, i-1] = dp2[i-1]  # 将 dp2 项归入到 dp3 项中

    # n > m + 1
    for i in range(1,N - 1):  # i 代表 m
        for j in range(i + 2, N+1):  # j 代表 n
            p3[j-1, i-1] = ((2 * j - 1) / np.sqrt(j ** 2 - i ** 2)) * np.cos(sita) * p3[j - 2, i-1] - np.sqrt(((j - 1) ** 2 - i ** 2) / (j ** 2 - i ** 2)) * p3[j - 3, i-1]

    # n > m + 1
    for i in range(1,N - 1):
        for j in range(i + 2, N+1):
            dp3[j-1, i-1] = ((2 * j - 1) / np.sqrt(j ** 2 - i ** 2)) * (-np.sin(sita) * p3[j - 2, i-1] + np.cos(sita) * dp3[j - 2, i-1]) - np.sqrt(((j - 1) ** 2 - i ** 2) / (j ** 2 - i ** 2)) * dp3[j - 3, i-1]

    # ------------------------- 完成勒让德多项式的计算 --------------------------- #

    p = np.column_stack((p1[1:N + 1], p3))  # 组合 p1 数据和 p3
    dp = np.column_stack((dp1[1:N + 1], dp3))  # 组合 dp1 数据和 dp3

    return p, dp

def ecf2sphere_mag(X, Y, Z, Nmax, g, h):
    """
    计算地磁场的矢量分量 Bx, By, Bz 以及 H, D, I, F
    """
    N = Nmax  # 阶数

    # 由地固直角坐标求地固球坐标
    L = np.arctan2(Y, X)  # 经度，单位为rad
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)  # 地心距，单位为km
    lat_prime = np.arcsin(Z / r)  # 地心纬度，单位为rad
    lat_co = np.pi / 2 - lat_prime  # 地心余纬，单位为rad

    # 计算勒让德多项式 p 和多项式对余纬的导数
    p, dp = simit_l(lat_co, Nmax)  # 调用辅助函数

    # 初始化磁场分量
    dV_lat_co = 0
    dV_lon = 0
    dV_r = 0  # 地固球坐标系下的三轴分量，单位为nT

    # 按级数公式计算磁场强度分量
    for i in range(1,N+1):
        for j in range(1,i + 2):
            m = j - 1  # 计算次数 m
            factor = (a / r) ** (i + 2)
            cos_mL = np.cos(m * L)
            sin_mL = np.sin(m * L)

            # 计算 dV_lat_co, dV_lon, dV_r
            dV_lat_co += factor * (g[i-1][j-1] * cos_mL + h[i-1][j-1] * sin_mL) * dp[i-1][j-1]
            dV_lon += factor * (g[i-1][j-1] * sin_mL - h[i-1][j-1] * cos_mL) * m * p[i-1][j-1] / np.sin(lat_co)
            dV_r -= (i + 1) * factor * (g[i-1][j-1] * cos_mL + h[i-1][j-1] * sin_mL) * p[i-1][j-1]

    # 地固球坐标系下的三轴分量（北向、东向、垂直向下）
    X_prime = dV_lat_co
    Y_prime = dV_lon
    Z_prime = dV_r

    # 计算地固球坐标系下的三个磁场分量
    Bx = X_prime
    By = Y_prime
    Bz = Z_prime  # 单位为 nT

    # 计算磁场的七个矢量分量
    H = np.sqrt(Bx ** 2 + By ** 2)  # 水平分量 H，单位为 nT
    D = np.arctan2(By, Bx) * 180 / np.pi  # 磁偏角 D，单位为度
    I = np.arctan(Bz / H) * 180 / np.pi  # 磁倾角 I，单位为度
    F = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)  # 总磁场强度 F，单位为 nT

    return Bx, By, Bz, H, D, I, F

def H1Mag2Sph(r, sita, lamda, Nmax, g, h):
    """
    将磁场从直角坐标转换到球坐标系
    参数:
    r: 半径
    sita: 余纬角
    lamda: 经度
    Nmax: 阶数截断

    返回:
    H1: 转换矩阵 H1
    """

    N = Nmax  # 截断阶数
    p, dp = simit_l(sita, Nmax)  # 调用函数计算勒让德多项式及其导数
    H1 = np.zeros((3, 3))  # 初始化 H1 矩阵

    # 初始化导数
    dV2_lamda2 = 0
    dV2_r2 = 0
    dV2_sita_lamda = 0
    dV2_r_sita = 0
    dV2_r_lamda = 0
    dV_r = 0
    dV_lamda = 0
    dV_sita = 0

    # 计算标量磁位对 θ、λ、r 的导数
    for i in range(1, N+1):
        for j in range(1, i+2):
            dV2_sita_lamda += a * (a / r)**(i+1) * (j-1) * (-g[i-1, j-1] * np.sin((j-1) * lamda) + h[i-1, j-1] * np.cos((j-1) * lamda)) * dp[i-1, j-1]
            dV2_lamda2 += -a * (a / r)**(i+1) * (j-1)**2 * (g[i-1, j-1] * np.cos((j-1) * lamda) + h[i-1, j-1] * np.sin((j-1) * lamda)) * p[i-1, j-1]
            dV2_r_sita += -(a / r)**(i+2) * (i+1) * (g[i-1, j-1] * np.cos((j-1) * lamda) + h[i-1, j-1] * np.sin((j-1) * lamda)) * dp[i-1, j-1]
            dV2_r_lamda += -(i+1) * (a / r)**(i+2) * (j-1) * (-g[i-1, j-1] * np.sin((j-1) * lamda) + h[i-1, j-1] * np.cos((j-1) * lamda)) * p[i-1, j-1]
            dV2_r2 += (1 / r) * (a / r)**(i+2) * (i+1) * (i+2) * (g[i-1, j-1] * np.cos((j-1) * lamda) + h[i-1, j-1] * np.sin((j-1) * lamda)) * p[i-1, j-1]

            dV_sita += a * (a / r)**(i+1) * (g[i-1, j-1] * np.cos((j-1) * lamda) + h[i-1, j-1] * np.sin((j-1) * lamda)) * dp[i-1, j-1]
            dV_lamda += a * (a / r)**(i+1) * p[i-1, j-1] * (-g[i-1, j-1] * np.sin((j-1) * lamda) + h[i-1, j-1] * np.cos((j-1) * lamda)) * (j-1)
            dV_r += -(i+1) * (a / r)**(i+2) * p[i-1, j-1] * (g[i-1, j-1] * np.cos((j-1) * lamda) + h[i-1, j-1] * np.sin((j-1) * lamda))

    # 计算 dV2_sita2
    dV2_sita2 = -2 * r * dV_r - r**2 * dV2_r2 - (np.cos(sita) / np.sin(sita)) * dV_sita - (1 / np.sin(sita)**2) * dV2_lamda2

    # 填充 H1 矩阵
    H1[0, 0] = (1 / r) * dV2_sita2
    H1[0, 1] = (1 / r) * dV2_sita_lamda
    H1[0, 2] = (1 / r) * (dV2_r_sita - (1 / r) * dV_sita)

    H1[1, 0] = (1 / (r * np.sin(sita))) * ((np.cos(sita) / np.sin(sita)) * dV_lamda - dV2_sita_lamda)
    H1[1, 1] = -(1 / (r * np.sin(sita))) * dV2_lamda2
    H1[1, 2] = (1 / (r * np.sin(sita))) * ((1 / r) * dV_lamda - dV2_r_lamda)

    H1[2, 0] = dV2_r_sita
    H1[2, 1] = dV2_r_lamda
    H1[2, 2] = dV2_r2

    return H1

def H2SpheTver(sita, lamda, r):
    """
    计算球坐标转直角坐标的雅可比矩阵
    参数:
    sita: 余纬角
    lamda: 经度
    r: 半径

    返回:
    H2: 雅可比矩阵
    """
    # 计算直角坐标
    xe = r * np.sin(sita) * np.cos(lamda)
    ye = r * np.sin(sita) * np.sin(lamda)
    ze = r * np.cos(sita)

    # 初始化雅可比矩阵
    H2 = np.zeros((3, 3))

    H2[0, 0] = (np.cos(sita) * np.cos(lamda)) / r
    H2[0, 1] = (np.cos(sita) * np.sin(lamda)) / r
    H2[0, 2] = -np.sin(sita) / r

    # 防止除以零
    xy_sqrt = np.sqrt(xe**2 + ye**2)
    if xy_sqrt != 0:
        H2[1, 0] = -np.sin(lamda) / xy_sqrt
        H2[1, 1] = np.cos(lamda) / xy_sqrt
    else:
        H2[1, 0] = 0
        H2[1, 1] = 0

    H2[1, 2] = 0

    H2[2, 0] = np.sin(sita) * np.cos(lamda)
    H2[2, 1] = np.sin(sita) * np.sin(lamda)
    H2[2, 2] = np.cos(sita)

    return H2

