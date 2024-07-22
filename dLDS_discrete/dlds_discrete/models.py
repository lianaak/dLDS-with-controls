import numpy as np
from pydmd import DMDc
import util_controls as uc


class SLDSwithControl:

    """Switching Linear Dynamical System with Control
    """

    def __init__(self, A, B, K, D_control):
        self.A = A
        self.B = B
        self.K = K
        self.D_control = D_control

    def fit(self, x, u):
        pass

    def transform(self, x, u):
        pass


class ControlModel:

    def __init__(self, control_density=0.02):
        self.all_U = None
        self.all_A = None
        self.all_B = None
        self.sparsity_pattern = None
        self.control_density = control_density

    def fit(self, X, Y, num_iter, U_0=None, D_control=1):
        """Fit the control model

        """

        if U_0 is None:
            U_0 = uc.init_U(X, Y, D_control)  # initial control input

        D_control, T = U_0.shape
        D_obs = X.shape[0]

        self.all_U = np.zeros((num_iter, D_control, T))  # all control inputs
        self.all_U[0] = U_0

        self.all_A = np.zeros((num_iter, D_obs, D_obs))  # all A matrices
        self.all_B = np.zeros((num_iter, D_obs, D_control))  # all B matrices
        self.sparsity_pattern = np.zeros(
            (D_control, T), dtype=bool)  # sparsity pattern of U

        all_err = np.zeros((num_iter, 2))  # error of the model

        for i in range(num_iter):
            print(f'Iteration {i + 1}/{num_iter}')
            # solve for A and B

            AB = Y @ np.linalg.pinv(np.vstack([X, self.all_U[i]]))

            A = AB[:, :D_obs]
            B = AB[:, D_obs:]

            # compute error
            all_err[i] = np.linalg.norm(A @ X + B @ self.all_U[i] - Y, 2)

            if i > 0:
                self.all_A[i+1] = A
                self.all_B[i+1] = B

            # solve for U using the current A and B
            U = np.linalg.lstsq(B, Y - A @ X, rcond=None)[0]

            # ensure that U is positive
            # U = np.abs(U)

            # update sparsity pattern
            U_nonsparse = U.copy()
            U_nonsparse[~self.sparsity_pattern] = 0

            U[self.sparsity_pattern] = 0
            # re-updated sparsity pattern
            re_up_pattern = np.zeros_like(self.sparsity_pattern)
            has_converged = False

            for j in range(D_control):
                tmp = U[j, :]
                try:
                    threshold = max(np.quantile(tmp[tmp > 0], self.control_density),
                                    np.min(tmp[tmp > 0]))  # threshold for sparsity pattern, if control_density is 0.1, then we take the 0.1 quantile of the positive values of the control input as the threshold unless it is smaller than the smallest positive value
                except IndexError:
                    pass
                above_threshold = tmp > threshold
                if np.sum(above_threshold) == 0:
                    # if all control inputs are equal to or below the threshold, then the algorithm has converged
                    has_converged = True
                    print(
                        f'No values above threshold {threshold} in control input {j}')
                else:
                    has_converged = False
                # update sparsity pattern, if control input is below threshold
                self.sparsity_pattern[j, :] = ~(above_threshold)

            # is this correct
            if has_converged:
                break
            # set control inputs below threshold to 0
            U[self.sparsity_pattern] = 0

            # U = np.abs(U)
            self.all_U[i+1] = U
            # compute error
            all_err[i+1, 1] = np.linalg.norm(A @ X + B @ U - Y, 'fro')

        if has_converged:
            for n in range(i, num_iter):
                self.all_U[n] = self.all_U[i]
                self.all_A[n] = self.all_A[i]
                self.all_B[n] = self.all_B[i]
            pass
        else:

            print('not converged - calculate exact dmdc')
            dmdc = DMDc(svd_rank=2)
            dmdc.fit(X=X.T, I=self.all_U[num_iter-1].T)
            modes = dmdc.modes
            eigs = np.power(
                dmdc.eigs, dmdc.dmd_time["dt"] // dmdc.original_time["dt"]
            )
            A = np.linalg.multi_dot(
                [modes, np.diag(eigs), np.linalg.pinv(modes)]
            )
            self.all_A[-1] = A
            self.all_B[-1] = dmdc.B

    def transform(self, x, u):
        pass
