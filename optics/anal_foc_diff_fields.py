from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.special as spf



def phi(x, y):
    # return np.arctan(y, x)
    return np.arctan2(-y,-x) + np.pi

def rho(x, y):
    return ( x**2. + y**2. )**0.5

def E_field(dipole_orientation_angle, xi, y, k):
    """ Defines the analytics approximation to the focused and
        diffracted field for dipole oriented in the
        focal plane at an angle 'dipole_orientation_angle' from the
        x-axis
        """
    # print('inside anal_foc_diff_fields.E_field, \n',
    #     'xi.shape = ',xi.shape,'\n',
    #     'y.shape = ',y.shape,
    #     'k = ',k
    #     # 'k.shape = ',k.shape
    #     )
    psi = dipole_orientation_angle
    phi_P = phi(xi, y) - psi

    E_xP = (
            (
                np.cos( phi_P )**2.
                +
                np.cos( 2*(phi_P) )
                )
            *
            np.nan_to_num(spf.spherical_jn( 1, k*rho(xi, y) )/(k*rho(xi, y)))
        +
        (
            np.sin( phi_P )**2.
            *
            spf.spherical_jn( 0, k*rho(xi, y) )
            )
        )

    E_yP = (
        np.sin(phi_P)
        *
        np.cos(phi_P)
        *
        spf.spherical_jn( 2, k*rho(xi, y) )
        )

    # print("rho(xi, y) = ",rho(xi, y))
    # print("xi, y = ",xi, ' ',y)
    E_zP = -np.cos(phi_P) * np.nan_to_num(
        spf.jv(2, k*rho(xi, y) )/(k*rho(xi, y))
        )


    E_x = np.cos(psi)*E_xP - np.sin(psi)*E_yP
    E_y = np.sin(psi)*E_xP + np.cos(psi)*E_yP
    E_z = E_zP

    return np.array([E_x, E_y, E_z])*k**3.

# def anal_psf(ori, xi, y, k):
#     inte_psf = (
#         np.abs(E_x(ori, xi, y, k))**2.
#         +
#         np.abs(E_y(ori, xi, y, k))**2.
#         +
#         np.abs(E_z(ori, xi, y, k))**2.
#         )
#     return inte_psf

# def anal_interf(ori, x, y, k, d):
#     first_term = (
#         (E_x(ori, x, y, k))*np.conj(E_x(ori, x-d, y, k))
#         +
#         (E_y(ori, x, y, k))*np.conj(E_y(ori, x-d, y, k))
#         +
#         (E_z(ori, x, y, k))*np.conj(E_z(ori, x-d, y, k))
#         )
#     second = (
#         (E_x(ori, x-d, y, k))*np.conj(E_x(ori, x, y, k))
#         +
#         (E_y(ori, x-d, y, k))*np.conj(E_y(ori, x, y, k))
#         +
#         (E_z(ori, x-d, y, k))*np.conj(E_z(ori, x, y, k))
#         )
#     return first_term+second

