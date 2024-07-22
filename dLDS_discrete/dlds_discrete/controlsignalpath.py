

from re import A
import numpy as np
from scipy.linalg import svd
from scipy.integrate import quad


class ControlSignalPath:

    def __init__(self, data, all_A, all_B, all_U, num_iter):
        self.data = data
        self.all_A = all_A
        self.all_B = all_B
        self.all_U = all_U
        self.num_iter = num_iter
        self.all_err = None
        self.A = None
        self.B = None
        self.U = None
        self.aic_first_term = None
        self.window = None
        self.scores = None
        self.errors = None

    def calc_best_control_signal(self, window=10):
        """ calculate the best control signal using the AIC criterion

        Args:
            window (int, optional): Defaults to 10. The window size for the AIC criterion.
        """

        # self.objective_function = objective_function
        self.best_control_signal_idx = None

        scores = np.zeros(self.all_U.shape[0])

        self.window = window

        lambda_ = self.data.shape[1] / window

        errors = np.zeros(scores.shape)

        # first we have to z-score the data
        self.data = (self.data - np.mean(self.data, axis=1).reshape(-1, 1)
                     ) / np.std(self.data, axis=1).reshape(-1, 1)

        self.aic_first_term = []

        for i in range(self.all_U.shape[0]):
            scores[i] = - self.aic_multi_step_dmdc(self.data, self.all_U[i],
                                                   self.all_A[i], self.all_B[i], lambda_)[0]
            errors[i] = self.aic_multi_step_dmdc(
                self.data, self.all_U[i], self.all_A[i], self.all_B[i], lambda_)[1]

        self.scores = scores
        self.errors = errors
        self.best_control_signal_idx = np.argmax(scores)
        self.A = self.all_A[self.best_control_signal_idx]
        self.B = self.all_B[self.best_control_signal_idx]
        self.U = self.all_U[self.best_control_signal_idx]

    def predict(self, data, A, B, U):
        # not being used currently
        predicted = np.zeros((*data.shape, 1))
        initial_conditions = np.random.randn(A.shape[0]).reshape(-1, 1)
        predicted[:, 0] = np.array(initial_conditions)
        for i in range(1, data.shape[1]-1):
            predicted[:, i] = A @ predicted[:, i-1] + \
                B @ U[:, i-1].reshape(-1, 1)
        return predicted.reshape(data.shape)

    def aic_multi_step_dmdc(self, dat, U, A, B, lambda_, num_steps=2):
        """ calculate the AIC criterion for a given control signal

        Args:
            dat (np.ndarray): X
            U (np.ndarray): controls
            A (np.ndarray): A matrix
            B (np.ndarray): B matrix
            lambda_ (np.ndarray): lambda for the AIC criterion
            num_steps (int, optional): Defaults to 2

        Returns:
            np.ndarray: aic_vec
            np.ndarray: RSS 
        """
        X1 = dat[:, :-1]

        RSS = self.calc_nstep_error(dat, U, A, B, num_steps)[-1]

        num_signals = U.shape[0]
        k = np.count_nonzero(U) / num_signals
        n = X1.shape[1]

        self.aic_first_term.append(- 2*num_signals*k + 2*n)

        aic_vec = - 2*num_signals*k + 2*n + \
            lambda_ * n*np.log(RSS)  # aic_window

        return aic_vec, RSS

    def calc_nstep_error(self, X, U, A, B, num_steps=2):
        """ calculate the error for a given control signal

        Returns:
            num_steps: number of steps to predict. if num_steps is e.g. 2, the function will predict the next two steps.
        """

        if num_steps > 1:
            err_steps_to_save = num_steps
            all_err = np.zeros(err_steps_to_save)

        start_idx = np.arange(0, X.shape[1] - num_steps)
        end_idx = np.arange(1, X.shape[1] - num_steps + 1)

        X1 = X[:, start_idx]

        X_hat = X1

        for i in range(num_steps):
            X_hat = A @ X_hat + B @ U[:, i:(X.shape[1] -
                                            (num_steps - i))]
            # Frobenius norm of the difference between the predicted and the actual data
            all_err[i] = np.linalg.norm(X_hat - X[:, end_idx + i - 1], 'fro')

        return all_err
