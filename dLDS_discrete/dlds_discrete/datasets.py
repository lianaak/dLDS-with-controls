from models import SLDSwithControl
import numpy as np
from scipy import sparse


class SLDSwithControlDataset(SLDSwithControl):
    """generates data from a SLDS with controls system

    Parameters
    ----------

    A : list of numpy arrays
        List of A matrices for each discrete state
    B : numpy array
        B matrix
    K : int
        Number of discrete states
    D_control : int
        Number of control inputs
    z_ : numpy array
        Discrete states
    u_ : numpy array
        Control inputs

    """

    def __init__(self, A, B, K, D_control, z_=None, u_=None):
        super().__init__(A, B, K, D_control)

    def generate(self, T,  discrete_state_maxT=200, control_density=0.02, initial_conditions=None, fix_point_change=False, add_noise=False):

        # define time intervals for each discrete state z
        z_interval = np.random.randint(1, discrete_state_maxT, size=T//10)
        z = np.zeros(T, dtype=int)

        # create a random sequence of discrete states z
        current_state = np.random.randint(0, self.K)
        z[0:z_interval[0]] = current_state  # initial state
        for i in range(1, T//10):
            start = np.sum(z_interval[:i])
            end = start + z_interval[i]
            current_state = (current_state + 1) % self.K
            z[start:end] = current_state

        self.z_ = z
        # one hot encoding of the states, e.g. [1, 0, 0] for state 0 at time point t
        z_one_hot = np.zeros((self.K, T))
        z_one_hot[z, np.arange(T)] = 1

        # Define control input
        u_temp = sparse.rand(self.D_control, T, density=control_density,
                             format="csr")
        # u_temp.data[:] = 1  # such that we only have 0 and 1
        # add the z one hot encoding to the control input
        if fix_point_change:
            self.u_ = sparse.vstack((u_temp, z_one_hot)).A
        else:
            self.u_ = u_temp.A

        # define first state x[0] with initial conditions
        if initial_conditions is None:
            initial_conditions = np.random.randn(self.A[0].shape[0])
        x = np.zeros((self.A[0].shape[0], T, 1))
        x[:, 0] = np.array(initial_conditions).reshape(-1, 1)

        # generate the rest of the states
        for i in range(1, T-1):
            x[:, i] = self.A[self.z_[i]] @ x[:, i-1] + \
                self.B @ self.u_[:, i-1].reshape(-1, 1)

        if add_noise:
            x += 0.1*np.random.randn(*x.shape)

        return x
