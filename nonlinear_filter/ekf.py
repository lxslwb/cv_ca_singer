import numpy as np
from nonlinear_filter.nonlinear_filter import NonlinearFilter


class EKF(NonlinearFilter):
    def __init__(self, num_x, num_y, delta_t, q, r, p_pos, state_pos):
        self.num_x = num_x
        self.num_y = num_y
        self.delta_t = delta_t
        self.q_matrix = q  # process noise matrix
        self.r_matrix = r  # measurement noise matrix
        self.k_gain = np.zeros([self.num_x, self.num_y])  # kalman gain
        self.p_priori = np.zeros([self.num_x, self.num_x])  # estimation covariance priori matrix
        self.p_posteriori = p_pos  # initial estimation uncertainty covariance matrix
        self.state_priori = np.zeros([self.num_x, 1])
        self.state_posteriori = state_pos  # initial priori state

    def get_points(self, choice):
        pass

    # def time_update(self, f_fun, f_jac_fun, **kwargs):
    #     self.state_priori = self.state_posteriori + self.delta_t * f_fun(x=self.state_posteriori, **kwargs)
    #     a = f_jac_fun(x=self.state_posteriori, **kwargs)
    #     phi = np.eye(self.num_x) + self.delta_t * a
    #     self.p_priori = phi @ self.p_posteriori @ phi.T + self.q_matrix

    def time_update(self, f_fun, f_jac_fun, **kwargs):
        self.state_priori = f_fun(x=self.state_posteriori, **kwargs)
        phi = f_jac_fun(x=self.state_posteriori, **kwargs)
        self.p_priori = phi @ self.p_posteriori @ phi.T + self.q_matrix

    def measure_update(self, h_fun, h_jac_fun, **kwargs):
        h = h_jac_fun(x=self.state_priori)
        self.k_gain = self.p_priori @ h.T @ np.linalg.inv(h @ self.p_priori @ h.T + self.r_matrix)
        self.state_posteriori = self.state_priori + self.k_gain @ (
                kwargs['measurement_value'] - h_fun(x=self.state_priori))
        # self.p_posteriori = (np.eye(self.num_x) - self.k_gain @ h) @ self.p_priori @ (
        #         np.eye(self.num_x) - self.k_gain @ h).T + self.k_gain @ self.r_matrix @ self.k_gain.T  # first format
        self.p_posteriori = (np.eye(self.num_x) - self.k_gain @ h) @ self.p_priori
        # self.p_posteriori = (self.p_posteriori + self.p_posteriori.T) / 2  # symmetric

