# Image Treatment
# Authors: Renan Cabrera, Roberto Cabrera.
# Copyright (c) 2018
# All rights reserved.

import numpy as np
import ctypes
from ctypes import *

# from scipy.ndimage.filters import gaussian_filter


class ImageFunc:

    def __init__(self, x_amplitude):
        self.x_amplitude = x_amplitude
        return

    def toHermitian(self, A):
        return (A + A.conj().T) / 2.

    def commutator(self, A, B):
        return A.dot(B) - B.dot(A)

    def zeroPadding(self, A, extraDIM):
        x_gridDIM = A.shape[0]
        x_gridDIM_padded = x_gridDIM + extraDIM

        B = np.zeros((x_gridDIM_padded, x_gridDIM_padded), dtype=np.complex128)

        B[extraDIM / 2:x_gridDIM + extraDIM / 2,
            extraDIM / 2:x_gridDIM + extraDIM / 2] = A
        return B

    ###############
    ###   FFT   ###
    ###############

    def Fourier_ax1(self, w):
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(w, axes=1), axis=1), axes=1)

    def iFourier_ax1(self, w):
        return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(w, axes=1), axis=1), axes=1)

    def Fourier_ax0(self, w):
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(w, axes=0), axis=0), axes=0)

    def iFourier_ax0(self, w):
        return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(w, axes=0), axis=0), axes=0)

    ##################
    ###   SHEARS   ###
    ##################

    def shear_ax1(self, w, theta):
        gridDIM = w.shape[0]

        # x_amplitude = 10
        # x_amplitude = 1024
        dx = 2. * self.x_amplitude / np.float(gridDIM)

        dp = 2. * np.pi / (2. * self.x_amplitude)
        p_amplitude = dp * gridDIM / 2.

        x_range = np.linspace(-self.x_amplitude,
                              self.x_amplitude - dx, gridDIM)
        p_range = np.linspace(-p_amplitude, p_amplitude - dp, gridDIM)

        x = x_range[np.newaxis, :]
        y = x_range[:, np.newaxis]

        p_x = p_range[np.newaxis, :]
        p_y = p_range[:, np.newaxis]

        a = np.tan(theta / 2.)

        return self.iFourier_ax1(np.exp(-1j * y * p_x * a) * self.Fourier_ax1(w))

    def shear_ax0(self, w, theta):
        gridDIM = w.shape[0]

        # x_amplitude = 10
        # x_amplitude = 1024
        dx = 2. * self.x_amplitude / np.float(gridDIM)

        dp = 2. * np.pi / (2. * self.x_amplitude)
        p_amplitude = dp * gridDIM / 2.

        x_range = np.linspace(-self.x_amplitude,
                              self.x_amplitude - dx, gridDIM)
        p_range = np.linspace(-p_amplitude, p_amplitude - dp, gridDIM)

        x = x_range[np.newaxis, :]
        y = x_range[:, np.newaxis]

        p_x = p_range[np.newaxis, :]
        p_y = p_range[:, np.newaxis]

        b = -np.sin(theta)

        return self.iFourier_ax0(np.exp(-1j * x * p_y * b) * self.Fourier_ax0(w))

    def rotate(self, w, theta):
        return self.shear_ax1(self.shear_ax0(self.shear_ax1(w, theta), theta), theta)

    def doubleBracketOp(self, P, H, ds, steps):
        p = P.copy()

        J = []
        normMatrix = []
        normCommutator = []

        for i in range(steps):
            p += ds * self.commutator(p, self.commutator(p, H))

            J.append(np.abs(np.trace(p.dot(H))))
            normMatrix.append(np.linalg.norm(p - H))
            normCommutator.append(np.linalg.norm(self.commutator(p, H)))

        J = np.array(J)
        normMatrix = np.array(normMatrix)
        normCommutator = np.array(normCommutator)

        return (p, J, normMatrix, normCommutator)

    def Wigner_To_Rho(self, W):
        Rho = self.Fourier_ax0(W)
        # return Rho
        return self.rotate(Rho, -np.pi / 4.)

    def Rho_To_Wigner(self, Rho):
        Rho = self.rotate(Rho, np.pi / 4.)
        # W = iFourier_ax0( Rho )
        # return Rho
        return self.iFourier_ax0(Rho)
