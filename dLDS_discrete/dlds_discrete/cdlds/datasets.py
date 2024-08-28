import numpy as np
from scipy import sparse
from scipy import linalg
from dataclasses import dataclass, field


@dataclass
class CdLDSDataGenerator:
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
    U_ : numpy array
        Control inputs

    """

    D_obs: int = 4
    A: list = field(default_factory=list)
    B: np.ndarray = None
    K: int = 2
    D_control: int = 1
    fix_point_change: bool = True
    eigenvalue_radius: float = 0.99

    U_: np.ndarray = None
    z_: np.ndarray = None

    def __post_init__(self):
        if len(self.A) == 0 and self.B is None:
            self.create_dynamics()
        elif len(self.A) > 0 and self.B is None:
            raise ValueError("A is defined but B is not defined")
        elif len(self.A) == 0 and self.B is not None:
            raise ValueError("B is defined but A is not defined")

    def generate_data(self, n_time_points,  discrete_state_maxT=200, control_density=0.02, initial_conditions=None, sigma=0):
        """ Generate data from the system

        Args:
            n_time_points (int): Number of time points
            discrete_state_maxT (int, optional): Maximum time interval for each discrete state. Defaults to 200.
            control_density (float, optional): Density of the control input matrix. Defaults to 0.02.
            initial_conditions (np.ndarray, optional): Initial conditions for the system. Defaults to None.
            sigma (int, optional): Noise level. Defaults to 0.
            eigenvalue_radius (float, optional): Radius of the eigenvalues. Defaults to 0.99.

        Returns:
            np.ndarray: Generated data
        """

        self.create_controls(
            n_time_points, discrete_state_maxT=discrete_state_maxT, control_density=control_density)

        # define first state x[0] with initial conditions
        if initial_conditions is None:
            initial_conditions = np.random.randn(self.A[0].shape[0])

        # generate with multi step reconstruction
        x = self.multi_step_reconstruction(
            n_time_points=n_time_points, X0=initial_conditions, A=self.A, B=self.B,  U=self.U_, z=self.z_)

        # add noise
        if sigma > 0:
            x += sigma*np.random.randn(*x.shape)

        return x

    def create_dynamics(self):
        """
        create parameters for the dynamics of the system, i.e. A and B matrices 

        Args:
            eigenvalue_radius (float, optional): radius of the eigenvalues. Defaults to 0.99.
        """

        assert self.D_obs % 2 == 0, "Dimension should be even"

        assert len(self.A) == 0, "A is already defined"
        assert self.B is None, "B is already defined"

        for k in range(self.K):

            np.random.seed(k)
            # define eigenvalues to be on the unit circle
            # eigenvalues on the unit circle, 1j is the imaginary unit such that we can sample points on the entire circle
            # not all values on the unit circle are complex, but we sample only complex ones here for simplicity
            # the imaginary part should be relatively small such that we have fewer oscillations
            complex_eigenvals = np.exp(
                0.06j * np.random.rand(int(self.D_obs/2)) * 2 * np.pi)

            # to obtain a real matrix with complex eigenvalues, we need to use the Jordan form which requires blocks with conjugate pairs of eigenvalues (aka. a + bi and a - bi)
            jordan_blocks = []
            for eigval in complex_eigenvals:
                a = eigval.real
                b = eigval.imag
                jordan_blocks.append(np.array([[a, -b], [b, a]]))

            # create a block diagonal matrix from all the Jordan blocks
            jordan = linalg.block_diag(*jordan_blocks)

            A_rand = np.random.rand(self.D_obs, self.D_obs)
            Q, _ = np.linalg.qr(A_rand)

            A_k = (Q @ jordan @ np.linalg.inv(Q)).real*self.eigenvalue_radius

            self.A.append(A_k)

        # B matrix is random
        self.B = np.array(np.random.rand(self.D_obs, self.D_control))

        # add additional B matrix for each state
        if self.fix_point_change:
            for _ in range(self.K):
                self.B = np.hstack(
                    (self.B, np.array(np.random.rand(self.D_obs, 1))))

    def create_controls(self, T,  discrete_state_maxT=200, control_density=0.02):
        """Create control inputs for the system

        Args:
            n_time_points (int): Number of time points
            discrete_state_maxT (int, optional): Maximum time interval for each discrete state. Defaults to 200.
            control_density (float, optional): Density of the control input matrix. Defaults to 0.02.
        """

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

        # add the z one hot encoding to the control input
        if self.fix_point_change:
            self.U_ = sparse.vstack((u_temp, z_one_hot)).toarray()
        else:
            self.U_ = u_temp.toarray()

    def single_step_reconstruction(self, x, A, B, u):
        """Single step of the system

        Args:
            A (np.ndarray): A matrix
            B (np.ndarray): B matrix
            x (np.ndarray): State vector
            u (np.ndarray): Control input vector

        Returns:
            np.ndarray: Next state
        """
        return A @ x + B @ u

    def multi_step_reconstruction(self, n_time_points, X0, A, B, U, z):
        """Multi step of the system

        Args:
            n_time_points (int): Number of time points
            X0 (np.ndarray): Initial state
            A (list): List of A matrices
            B (np.ndarray): B matrix
            U (np.ndarray): Control input matrix
            z (int): Discrete state

        Returns:
            np.ndarray: Reconstructed states
        """

        D = A[0].shape[0]
        X_reconstructed = np.zeros((D, n_time_points))
        X_reconstructed[:, 0] = X0

        for t in range(1, n_time_points):
            X_reconstructed[:, t] = self.single_step_reconstruction(
                x=X_reconstructed[:, t-1], A=A[z[t]], B=B, u=U[:, t-1])

        return X_reconstructed
