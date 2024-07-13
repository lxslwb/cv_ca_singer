"""
Main script for nonlinear filtering (EKF or SrCKF) of an object's state in 3D space.
Initializes system state, noise covariances, and filters.
Simulates dynamics, applies filters to update state estimates, and plots velocity
error to visualize performance.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_symbol.dynamical_model import f_ca_fun, \
    f_ca_jac_fun, f_cv_fun, f_cv_jac_fun, f_singer_fun, f_singer_jac_fun
from model_symbol.observational_model import h_fun, h_jac_fun
from nonlinear_filter.ekf import EKF
from nonlinear_filter.ckf import SrCKF

# 初始状态向量 [x, y, z, vx, vy, vz, ax, ay, az]
x_initial = np.array([[0], [0], [0], [1], [1], [1], [1], [1], [1]])
num_x = 9
num_y = 6
# 离散时间步长
dt = 1  # 1e-8
# 初始状态协方差矩阵
p_noise = 10
p_initial = np.eye(num_x) * p_noise ** 2  # 100
# 过程噪声协方差矩阵 Q
sigma_a = 1e-6  # 过程噪声标准差
q = np.eye(num_x) * sigma_a ** 2
# 测量噪声协方差矩阵R
r_noise = 1e-1
r = np.eye(num_y) * r_noise ** 2

# 初始化状态估计和协方差矩阵
rng = np.random.default_rng(42)
x_estimate = x_initial + rng.standard_normal(size=(num_x, 1)) * p_noise
p = p_initial

# 保存轨迹的列表
x_true = [x_initial]
x_estimate_list = [x_estimate]
ekf = EKF(num_x=num_x, num_y=num_y, delta_t=dt, q=q, r=r, p_pos=p, state_pos=x_estimate)
ckf = SrCKF(num_x=num_x, num_y=num_y, delta_t=dt, q=q, r=r, p_pos=p, state_pos=x_estimate)
# 模拟时间步
alpha = 1 / 10
filter_kalman = ckf
num_steps = 2 ** 12
f_fun = f_singer_fun
f_jac_fun = f_singer_jac_fun
data_i = 1
for _ in tqdm(range(num_steps), desc='仿真剩余时间', unit='data'):
    # 真实状态
    x_initial = f_fun(x=x_initial, t=dt, alpha=alpha)
    z = h_fun(x=x_initial) + rng.standard_normal(size=(num_y, 1)) * r_noise
    # 滤波
    filter_kalman.time_update(f_fun, f_jac_fun, t=dt, alpha=alpha)
    filter_kalman.measure_update(h_fun, h_jac_fun, measurement_value=z)
    # 保存轨迹
    data_i += 1
    x_true.append(x_initial)
    x_estimate_list.append(filter_kalman.state_posteriori)

# 转换为NumPy数组
x_true = np.array(x_true)
x_estimate_list = np.array(x_estimate_list)
error_y = x_true[:, 3] - x_estimate_list[:, 3]
num = np.arange(0, num_steps + 1)
plt.figure()
plt.plot(num[10:], error_y[10:], label='velocity error x')
# plt.plot(num, x_estimate_list[:, 4, 0], label='Estimated Trajectory')
plt.xlabel('series')
plt.ylabel('x_dot')
plt.legend()
plt.grid(True)
plt.show()
