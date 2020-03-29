import numpy as np

from ..calc import BEM_simulation_wrapper as bem
from ..calc import fitting_misLocalization as fit
from ..calc import coupled_dipoles as cp

import scipy.integrate as inte
import scipy.io as sio
import scipy.optimize as opt


import scipy.constants as con
## Import physical constants
hbar = con.physical_constants['Planck constant over 2 pi in eV s'][0]
c = con.physical_constants['speed of light in vacuum'][0]*1e2 #cm/m
kb = con.physical_constants['Boltzmann constant in eV/K'][0]
cm_per_nm = 1e-7
eps_water = 1.778

def invcmtohz(inv_lambda):
    return inv_lambda * 2 * np.pi * c
# script_d = 1.5

def coth(x):
    return 1/np.tanh(x)

def cor_fun(
    t,
#     timestep=.01*10**(-13),
    script_d,
    omega_q,
    gamma,
    T=50,
    ns = np.linspace(1,10, 10)
    ):
    """ Defines Correlation function

            C(t) = (m w^2 d / hbar) < q(t) q(0) rho(-inf) >

        with real and imaginary components:

            C(t) = C` + i C``

        Args:
            t: assumed to be in femptoseconds for order unity integration

        """

    t = t*1e-15
#     time_array = np.arange(0, t, timestep)
    beta = 1/(kb*T)
    zeta = np.sqrt(omega_q**2. - gamma**2./2 + 0j)
    phi = gamma/2 + 1j*zeta
    phip = np.conjugate(phi)
    ## Define array of nu_n
    nu_n = (2*np.pi/(hbar*beta))*(ns)
    ## expand for time axis
#     nu_n = nu_n[None, :

    def coth_of_args_and(p):
        return coth(1j*p*hbar*beta/2)

    terms1and2 = (
        (hbar / (4*zeta)) * (
            (coth_of_args_and(phip) - 1)*(
                np.exp(-phip*t))
            -
            (coth_of_args_and(phi) - 1)*(
                np.exp(-phi)*t)
            )
        )

    ## Handle sum over n differently depending on dimension of t
    if type(t) == np.ndarray:
        nu_n = nu_n[None, :]
        t = t[:,None]

    sum_over_n = np.sum(
        (
            nu_n/(
                (omega_q**2. + nu_n**2.)**2.
                +
                gamma**2.*nu_n**2.
                )
            *
            np.exp(-nu_n*t)
            ),
            axis=-1
        )

    last_term = (
        -2*gamma/beta
        *
        sum_over_n
        )

    c_of_t = (script_d**2 * omega_q**3. / hbar)*(terms1and2 + last_term)

    return c_of_t


def for_cor_fun(
    omega,
#     timestep=.01*10**(-13),
    script_d,
    omega_q,
    gamma,
    take_conjugate=True,
    T=50,
    ns = np.linspace(1,10, 10)
    ):
    """ Defines the Fourier Transform of the Correlation function

            F[C(t)] = (m w^2 d / hbar) F[<q(t) q(0) rho(-inf)>]

        Because all time dependence in the correlation function is
        negative exponential we simply replace the exponentials with
        their single sided Fourier transforms

            Integral[exp(-a t) exp(-i w t), from 0 to inf]
                = 1 / (a + i w)

        Args:
            t: assumed to be in femptoseconds for order unity integration

        """

    # t = t*1e-15
#     time_array = np.arange(0, t, timestep)
    beta = 1/(kb*T)
    zeta = np.sqrt(omega_q**2. - gamma**2./2 + 0j)
    if not take_conjugate:
        phi = gamma/2 + 1j*zeta
        phip = np.conjugate(phi)
    if take_conjugate:
        phi = gamma/2 - 1j*zeta
        phip = np.conjugate(phi)
    ## Define array of nu_n
    nu_n = (2*np.pi/(hbar*beta))*(ns)

    def coth_of_args_and(p):
        if not take_conjugate:
            return coth(1j*p*hbar*beta/2)
        if take_conjugate:
            return coth(-1j*p*hbar*beta/2)


    terms1and2 = (
        (hbar / (4*zeta)) * (
            (coth_of_args_and(phip) - 1)*(
                (phip + 1j*omega)**(-1))
            -
            (coth_of_args_and(phi) - 1)*(
                (phi + 1j*omega)**(-1))
            )
        )

    ## Handle sum over n differently depending on dimension of omega
    if type(omega) == np.ndarray:
        nu_n = nu_n[None, :]
        omega = omega[:,None]

    sum_over_n = np.sum(
        (
            nu_n/(
                (omega_q**2. + nu_n**2.)**2.
                +
                gamma**2.*nu_n**2.
                )
            *
            (nu_n + 1j*omega)**(-1)
            ),
            axis=-1
        )

    last_term = (
        -2*gamma/beta
        *
        sum_over_n
        )

    c_of_w = (script_d**2 * omega_q**3. / hbar)*(terms1and2 + last_term)

    return c_of_w


def g(
    t,
#     timestep=.01*10**(-13),
    script_d,
    omega_q=invcmtohz(600),
    gamma=invcmtohz(400),
    T=50,
    ns = np.linspace(1,10, 10)
    ):
    """ Defines linebroadening function

        Args:
            t: assumed to be in femptoseconds for order unity integration

        """

    t = t*1e-15
#     time_array = np.arange(0, t, timestep)
    beta = 1/(kb*T)
    zeta = np.sqrt(omega_q**2. - gamma**2./2 + 0j)
    phi = gamma/2 + 1j*zeta
    phip = np.conjugate(phi)
    ## Define array of nu_n
    nu_n = (2*np.pi/(hbar*beta))*(ns)

    def coth_of_args_and(p):
        return coth(1j*p*hbar*beta/2)

    def dub_t_int_exp_iphi(p):
        return (np.exp(-p*t) + p*t - 1)/p**2.

    goft_terms1and2 = (
        (script_d**2 * omega_q**3. / hbar)
        *
        (
            (hbar / (4*zeta)) * (
                (coth_of_args_and(phip) - 1)*(
                    dub_t_int_exp_iphi(phip))
                -
                (coth_of_args_and(phi) - 1)*(
                    dub_t_int_exp_iphi(phi))
                )
#             -
#             (
#                 2*gamma/beta
#                 *
#                 sum_over_n
#                 )
            )
        )

    ## Handle sum over n differently depending on dimension of t
    if type(t) == np.ndarray:
        nu_n = nu_n[None, :]
        t = t[:,None]

    sum_over_n = np.sum(
        (
            nu_n/(
                (omega_q**2. + nu_n**2.)**2.
                +
                gamma**2.*nu_n**2.
                )
            *
            dub_t_int_exp_iphi(nu_n)
            ),
            axis=-1
        )

    last_term = (
        (script_d**2 * omega_q**3. / hbar)
        *
        -2*gamma/beta
        *
        sum_over_n
        )

    goft = goft_terms1and2 + last_term

    return goft


def sigma_a(
    omega,
    script_d,
    t_bound=5,
    t_points=100,
    return_integrand=False,
    **kwargs):
    """ Absorption spectrum computed by integral over

            e^(i w t - g(t))

        where g(t) is the linebroadening function defined as the double
        time integral over the correlation function of the dipole
        operator.


        Args:

            t is in femptoseconds for g(t)

        """

    def integrand(t, omega=omega):
#         print(omega)
        if type(omega) is np.ndarray:
            return np.real(
                np.exp(
                    (1j*(omega[:, None])*t[None, :]*1e-15)
                    -
                    g(t, script_d, **kwargs)
                    )
                )

        else:
            return np.real(
                np.exp(1j*(omega)*t*1e-15 - g(t, script_d, **kwargs)))

    ## generate time domain for integration
    t = np.linspace(0, t_bound, t_points)

    if return_integrand:
        result = [t, integrand(t)]

    else:
         ## integrate last dimension of array (time)
        integral = inte.trapz(integrand(t), t, axis=-1)
        result = integral

    return result


def sigma_e(
    omega,
    script_d,
    omega_q=invcmtohz(600),
    t_bound=5,
    t_points=100,
    return_integrand=False,
    **kwargs):
    """ Absorption spectrum computed by integral over

            e^(i w t - g(t))

        where g(t) is the linebroadening function defined as the double
        time integral over the correlation function of the dipole
        operator.


        Args:

            t is in femptoseconds for g(t)

        """

    def integrand(t, omega=omega):
#         print(omega)
        if type(omega) is np.ndarray:
            return np.real(
                np.exp(
                    (1j*(
                        omega[:, None]
                        +
                        (script_d**2.*omega_q) # 2 lambda
                        )*t[None, :]*1e-15)
                    -
                    np.conj(g(
                        t,
                        script_d=script_d,
                        omega_q=omega_q,
                        **kwargs))
                    )
                )

        else:
            return np.real(
                np.exp(1j*(
                    omega+(script_d**2.*omega_q)
                    )*t*1e-15 - np.conj(g(
                        t,
                        script_d=script_d,
                        omega_q=omega_q,
                        **kwargs))))

    ## generate time domain for integration
    t = np.linspace(0, t_bound, t_points)

    if return_integrand:
        result = [t, integrand(t)]

    else:
         ## integrate last dimension of array (time)
        integral = inte.trapz(integrand(t), t, axis=-1)
        result = integral

    return result



def muk_mol_model(hbar_omegas,
    hbar_omega_eg_0,
    script_d,
    hbar_omega_0,
    hbar_gamma,
    T,
    t_bound=100,
    t_points=1000
    ):
    """ Model of emission lineshape
        """

    model = sigma_e(
        (
            (hbar_omegas - hbar_omega_eg_0)/hbar
            -
            (1/2 * script_d**2. * hbar_omega_0/hbar)
            ),
        script_d=script_d,
        omega_q=hbar_omega_0/hbar,
        gamma=hbar_gamma/hbar,
        T=T,
        t_bound=t_bound,
        t_points=t_points
        )

    return model



def muk_mol_fit_fun(params, *args):
    """ Try naive fit function with fixed integration differential size
        and bound.

        Params: (list of fit parameters)
        ~~~~~~~~~~~~~~~~~~~~~~~~
            hbar_omega_eg_0: the difference in zero point energy of the
                vibrational oscillators between the two electronic
                states (eV)

            script_d: unitless displacement of the vibronic potential surface
                between electronic states.

            hbar_omega_0: vibrational ressonance energy in eV

            hbar_gamma: damping rate from solvent or etc.

        Args: (list of x axis and data)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            hbar_omega:

            data:
        """
#     print(f'params:{params}, args:{args}')
    ## Define params and args with meaningful names
    hbar_omega_eg_0, script_d, hbar_omega_0, hbar_gamma, T = params
    hbar_omegas, data = args

    model = muk_mol_model(
        hbar_omegas,
        hbar_omega_eg_0,
        script_d,
        hbar_omega_0,
        hbar_gamma,
        T)

    ## Normalize model and data
    model = model / np.max(model)
    data = data / np.max(model)

    return model - data




