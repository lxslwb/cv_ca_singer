from abc import ABCMeta, abstractmethod


class NonlinearFilter(metaclass=ABCMeta):
    @abstractmethod
    def get_points(self, choice):
        pass

    @abstractmethod
    def time_update(self, f_fun, f_jac_fun, **kwargs):
        pass

    @abstractmethod
    def measure_update(self, h_fun, h_jac_fun, **kwargs):
        pass
