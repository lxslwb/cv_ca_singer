import numpy as np
from nonlinear_filter.nonlinear_filter import NonlinearFilter


class SrCKF(NonlinearFilter):
    def __init__(self, num_x, num_y, delta_t, q, r, p_pos, state_pos):
        self.num_x = num_x
        self.num_y = num_y
        self.m = 2 * num_x
        self.q_matrix = q  # process noise matrix
        self.r_matrix = r  # measurement noise matrix
        self.sq_q_matrix = np.linalg.cholesky(q)
        self.sq_r_matrix = np.linalg.cholesky(r)
        self.k_gain = np.zeros([self.num_x, self.num_y])  # kalman gain
        self.p_priori = np.zeros([self.num_x, self.num_x])  # estimation covariance priori matrix
        self.p_posteriori = p_pos  # initial estimation uncertainty covariance matrix
        self.s_priori = np.zeros([self.num_x, self.num_x])  # square of estimation covariance priori matrix
        # initial square of estimation uncertainty covariance matrix
        self.s_posteriori = np.linalg.cholesky(p_pos)
        self.state_priori = np.zeros([self.num_x, 1])
        self.state_posteriori = state_pos  # initial priori state
        # other
        self.inter_step = delta_t
        # self.inter_step = delta_t / 10
        self.delta_t = delta_t
        self.x_breve = np.sqrt(self.num_x) * np.concatenate([np.eye(self.num_x), -1 * np.eye(self.num_x)], axis=1)

    def get_points(self, choice):
        pass

    def time_update(self, f_fun, f_jac_fun, **kwargs):
        x_star = np.zeros([self.num_x, self.m])
        for i in range(self.m):
            x_big = self.s_posteriori @ self.x_breve[:, i:i + 1] + self.state_posteriori
            for dt in np.arange(0, self.delta_t, self.inter_step):
                # x_star_dot = f_fun(x=x_big, **kwargs)
                # x_big[:, 0:1] = x_star_dot * self.inter_step + x_big[:, 0:1]
                x_big[:, 0:1] = f_fun(x=x_big, **kwargs)
            x_star[:, i:i + 1] = x_big[:, 0:1]
        self.state_priori = 1 / self.m * np.sum(x_star, axis=1).reshape(-1, 1)
        x_star_matrix = 1 / np.sqrt(self.m) * (x_star - self.state_priori)
        tmp_matrix = np.concatenate([x_star_matrix, self.sq_q_matrix], axis=1)
        self.s_priori = np.linalg.qr(tmp_matrix.T, 'r').T

    def measure_update(self, h_fun, h_jac_fun, **kwargs):
        z_big = np.zeros([self.num_y, self.m])
        x_big = np.zeros([self.num_x, self.m])
        for i in range(self.m):
            x_big_tmp = self.s_priori @ self.x_breve[:, i:i + 1] + self.state_priori
            x_big[:, i:i + 1] = x_big_tmp
            z_big[:, i:i + 1] = h_fun(x=x_big_tmp)
        z_hat = 1 / self.m * np.sum(z_big, axis=1).reshape(-1, 1)
        tmp_matrix_z = 1 / np.sqrt(self.m) * (z_big - z_hat)
        tmp_matrix_1 = np.concatenate([tmp_matrix_z, self.sq_r_matrix], axis=1)
        s_zz = np.linalg.qr(tmp_matrix_1.T, 'r').T
        tmp_matrix_x = 1 / np.sqrt(self.m) * (x_big - self.state_priori)
        p_xz = tmp_matrix_x @ tmp_matrix_z.T
        divide1 = np.linalg.solve(s_zz, p_xz.T).T
        k = np.linalg.solve(s_zz.T, divide1.T).T
        self.state_posteriori = self.state_priori + k @ (kwargs['measurement_value'] - z_hat)
        tmp_matrix_2 = np.concatenate([tmp_matrix_x - k @ tmp_matrix_z, k @ self.sq_r_matrix], axis=1)
        self.s_posteriori = np.linalg.qr(tmp_matrix_2.T, 'r').T
