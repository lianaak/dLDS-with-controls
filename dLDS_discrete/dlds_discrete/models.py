import numpy as np


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
