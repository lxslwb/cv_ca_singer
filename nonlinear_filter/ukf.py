import numpy as np
from nonlinear_filter.nonlinear_filter import NonlinearFilter


class UKF(NonlinearFilter):
    def __init__(self, num_x, num_y, delta_t, q, r, p_pos, state_pos):
        self.num_x = num_x
        self.num_y = num_y
        self.q_matrix = q  # process noise matrix
        self.r_matrix = r  # measurement noise matrix
        self.k_gain = np.zeros([self.num_x, self.num_y])  # kalman gain
        self.p_priori = np.zeros([self.num_x, self.num_x])  # estimation covariance priori matrix
        self.p_posteriori = p_pos  # initial estimation uncertainty covariance matrix
        self.state_priori = np.zeros([self.num_x, 1])
        self.state_posteriori = state_pos  # initial priori state
        # other
        self.inter_step = delta_t / 3
        self.delta_t = delta_t
        self.w_m = np.ones((self.num_x * 2 + 1, 1))
        self.w_c = np.ones((self.num_x * 2 + 1, 1))
        self.x_breve = np.zeros([self.num_x * 2 + 1, self.num_x, 1])  # var_kappa
        self.ut_param = {'alpha': 1e-3, 'kappa': 0, 'beta': 2, 'lambda': 3 - self.num_x}
        # self.ut_param['lambda'] = self.ut_param['alpha'] ** 2 * (self.num_x + self.ut_param['kappa']) - self.num_x
        self.w_m[0] = self.ut_param['lambda'] / (self.num_x + self.ut_param['lambda'])
        self.w_c[0] = self.ut_param['lambda'] / (self.num_x + self.ut_param['lambda']) + (
                1 - self.ut_param['alpha'] ** 2 + self.ut_param['beta'])
        tmp = 1 / (2 * (self.num_x + self.ut_param['lambda']))
        for i in range(1, self.num_x * 2 + 1):
            self.w_m[i] = tmp
            self.w_c[i] = tmp

    def get_points(self, choice):
        if choice == 1:
            l_tmp = np.linalg.cholesky((self.num_x + self.ut_param['lambda']) * self.p_posteriori)
            self.x_breve[0] = self.state_posteriori
            for i in range(1, self.num_x + 1):
                self.x_breve[i] = self.state_posteriori + l_tmp[:, i - 1:i]
                self.x_breve[self.num_x + i] = self.state_posteriori - l_tmp[:, i - 1:i]
        else:
            l_tmp = np.linalg.cholesky((self.num_x + self.ut_param['lambda']) * self.p_priori)
            self.x_breve[0] = self.state_priori
            for i in range(1, self.num_x + 1):
                self.x_breve[i] = self.state_priori + l_tmp[:, i - 1:i]
                self.x_breve[self.num_x + i] = self.state_priori - l_tmp[:, i - 1:i]

    def time_update(self, f_fun, f_jac_fun=None, **kwargs):
        for i in range(self.num_x * 2 + 1):
            # Hybrid
            for dt in np.arange(0, self.delta_t, self.inter_step):
                x_breve_i_dot = f_fun(x=self.x_breve[i], **kwargs)
                self.x_breve[i] = x_breve_i_dot * self.inter_step + self.x_breve[i]
        self.state_priori = np.zeros([self.num_x, 1])
        for i in range(self.num_x * 2 + 1):
            self.state_priori += self.w_m[i] * self.x_breve[i]
        self.p_priori = np.zeros([self.num_x, self.num_x])
        for i in range(self.num_x * 2 + 1):
            self.p_priori += self.w_c[i] * (self.x_breve[i] - self.state_priori) @ (
                    self.x_breve[i] - self.state_priori).T
        self.p_priori += self.q_matrix

    def measure_update(self, h_fun, h_jac_fun, **kwargs):
        measure_ukf = np.zeros([self.num_x * 2 + 1, self.num_y, 1])
        for i in range(self.num_x * 2 + 1):
            measure_ukf[i] = h_fun(self.x_breve[i])
        measure_hat = 0
        for i in range(self.num_x * 2 + 1):
            measure_hat += self.w_m[i] * measure_ukf[i]
        p_y = np.zeros([self.num_y, self.num_y])
        p_xy = np.zeros([self.num_x, self.num_y])
        for i in range(self.num_x * 2 + 1):
            p_y += self.w_c[i] * (measure_ukf[i] - measure_hat) @ (measure_ukf[i] - measure_hat).T
            p_xy += self.w_c[i] * (self.x_breve[i] - self.state_priori) @ (measure_ukf[i] - measure_hat).T
        p_y += self.r_matrix
        self.k_gain = p_xy @ np.linalg.inv(p_y)
        self.state_posteriori = self.state_priori + self.k_gain @ (kwargs['measurement_value'] - measure_hat)
        self.p_posteriori = self.p_priori - self.k_gain @ p_y @ self.k_gain.T
