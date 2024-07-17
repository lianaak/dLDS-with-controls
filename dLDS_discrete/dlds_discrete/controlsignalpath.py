

from re import A
import numpy as np


class ControlSignalPath:

    def __init__(self, data, all_A, all_B, all_U, all_err, num_iter):
        self.data = data
        self.all_A = all_A
        self.all_B = all_B
        self.all_U = all_U
        self.all_err = all_err
        self.num_iter = num_iter
        self.A = None
        self.B = None
        self.U = None

    def calc_best_control_signal(self):

        # self.objective_function = objective_function
        self.best_control_signal_idx = None

        scores = np.zeros(self.all_U.shape[0])

        for i in range(self.all_U.shape[0]):
            X1 = self.data[:-1, :]
            X2 = self.data[1:, :]
            predicted = self.predict(
                self.data, self.all_A[i], self.all_B[i], self.all_U[i])
            scores[i] = self.aic(self.data, self.all_U[i],
                                 self.all_A[i], self.all_B[i], )

        self.best_control_signal_idx = np.argmin(scores)
        self.A = self.all_A[self.best_control_signal_idx]
        self.B = self.all_B[self.best_control_signal_idx]
        self.U = self.all_U[self.best_control_signal_idx]

    def predict(self, data, A, B, U):
        predicted = np.zeros((data.shape[0], data.shape[1], 1))
        initial_conditions = np.random.randn(A.shape[0])
        predicted[0] = np.array(initial_conditions).reshape(-1, 1)
        for i in range(1, data.shape[0]-1):
            predicted[i] = A @ predicted[i-1] + \
                B @ U[i-1, :].reshape(-1, 1)
        return predicted

    def aic_2step_dmdc(dat, U, A, B, num_steps):
        RSS = 0

        num_signals = U.shape[1]
        k = np.count_nonzero(U) / num_signals
        n = dat.shape[0]

        aic = 2*num_signals*(k+n) + n*np.log(RSS)

        return aic
