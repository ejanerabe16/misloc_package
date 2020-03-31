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
    ns = np.linspace(1,10, 10),
    take_conjugate=False,
    ):
    """ Defines Correlation function

            C(t) = (m w^2 d / hbar) < q(t) q(0) rho(-inf) >

        with real and imaginary components:

            C(t) = C` + i C``

        Args:
            t: assumed to be in femptoseconds for order unity integration

        """

    t = t*1e-15

    def exp_decay(phi):
        return np.exp(-phi*t)

    c_of_t = correlation_fun_root(
        exp_decay,
        script_d=script_d,
        omega_q=omega_q,
        gamma=gamma,
        T=T,
        ns=ns,
        take_conjugate=take_conjugate,
        )

    return c_of_t


def for_cor_fun(
    omega,
#     timestep=.01*10**(-13),
    script_d,
    omega_q,
    gamma,
    take_conjugate=False,
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

    def f_t_of_exp(phi):
        return (phi + 1j*omega)**(-1)

    c_of_w = correlation_fun_root(
        f_t_of_exp,
        script_d=script_d,
        omega_q=omega_q,
        gamma=gamma,
        T=T,
        ns=ns,
        take_conjugate=take_conjugate,
        )

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

    def dub_t_int_exp_iphi(p):
        return (np.exp(-p*t) + p*t - 1)/p**2.

    goft = correlation_fun_root(
        dub_t_int_exp_iphi,
        script_d=script_d,
        omega_q=omega_q,
        gamma=gamma,
        T=T,
        ns=ns
        )

    return goft



def correlation_fun_root(
    func_of_freq,
    # t,
#     timestep=.01*10**(-13),
    script_d,
    omega_q=invcmtohz(600),
    gamma=invcmtohz(400),
    T=50,
    ns = np.linspace(1,10, 10),
    take_conjugate=False,
    ):
    """ Defines linebroadening function

        Args:
            func_of_freq:
                A function handle that takes 1 argument frequency. For
                example for the correlation funtion
                    func_of_freq(phi) : e^(-pht*t)

        """

    # t = t*1e-15
#     time_array = np.arange(0, t, timestep)
    beta = 1/(kb*T)
    zeta = np.sqrt(omega_q**2. - gamma**2./4 + 0j)
    phi = gamma/2 + 1j*zeta
    phip = gamma/2 - 1j*zeta
    if take_conjugate:
        phi = np.conjugate(phi)
        phip = np.conjugate(phip)
    ## Define array of nu_n
    nu_n = (2*np.pi/(hbar*beta))*(ns)

    def coth_of_args_and(p):
        # print(f'coth(1j*{p}*hbar*beta/2) = {coth(1j*p*hbar*beta/2)}')
        imaginary_unit = 1j
        if take_conjugate:
            imaginary_unit = -1j
        return coth(imaginary_unit*p*hbar*beta/2)

    # def dub_t_int_exp_iphi(p):
    #     return (np.exp(-p*t) + p*t - 1)/p**2.

    goft_terms1and2 = (
        (hbar / (4*zeta)) * (
            (-coth_of_args_and(phi) + 1)*(
                func_of_freq(phi))
            -
            (-coth_of_args_and(phip) + 1)*(
                func_of_freq(phip))
            )
        )

    ## Handle sum over n differently depending on dimension of t
    if type(nu_n) == np.ndarray:
        nu_n = nu_n[:, None]
        # freqs = func_of_freq(nu_n)[:,None]

    sum_over_n = np.sum(
        (
            nu_n/(
                (omega_q**2. + nu_n**2.)**2.
                +
                gamma**2.*nu_n**2.
                )
            *
            func_of_freq(nu_n)
            ),
            axis=0
        )

    last_term = (
        -2*gamma/beta
        *
        sum_over_n
        )

    goft = (script_d**2 * omega_q**3. / hbar)*( #xi^2/m
        goft_terms1and2
        +
        last_term
        )

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




