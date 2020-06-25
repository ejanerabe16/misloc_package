import numpy as np

from ..calc import BEM_simulation_wrapper as bem
from ..calc import fitting_misLocalization as fit
from ..calc import coupled_dipoles as cp

import scipy.integrate as inte
import scipy.io as sio
import scipy.optimize as opt

## Dor matrix theorm implementation
import scipy.linalg as lin
import scipy.special as spl

## Some physical constants
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

    ## Original terms,
    goft_terms1and2 = (
        (hbar / (4*zeta)) * (
            (-coth_of_args_and(phi) + 1)*(
                func_of_freq(phi))
            +
            (+coth_of_args_and(phip) - 1)*(
                func_of_freq(phip))
            )
        )
    ## Try flipping sign,
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Nope, inverted whole spectrum
    #
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (-coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## This gave nonsense, spectrum no longer converged
    #
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         -
    #         (+coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Only a small change.
    ## Try flipping sign of phi terms,
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         -(-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         -
    #         (+coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Nope. Cause divergence of the cumulant again.
    #
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         -(-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Diverged again
    #
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (-coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (-coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Small change again, try both signs on bottom +
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Hope again! Looks realistic but vary different, try something similar
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Backwards again
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (+coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    ## Ang backwards again
    #
    ## Try last configuration with original signs on coth terms
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phip))
    #         )
    # #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-1 - coth_of_args_and(phi))*(
    #             func_of_freq(phi))
    #         +
    #         (+1 + coth_of_args_and(phip))*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-1 + coth_of_args_and(phi))*(
    #             func_of_freq(phi))
    #         +
    #         (+1 - coth_of_args_and(phip))*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phip))
    #         +
    #         (+coth_of_args_and(phip) - 1)*(
    #             func_of_freq(phi))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phi))
    #         +
    #         (+coth_of_args_and(phi) + 1)*(
    #             func_of_freq(phip))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phip))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phi))
    #         )
    #     )
    # goft_terms1and2 = (
    #     (hbar / (4*zeta)) * (
    #         (-coth_of_args_and(phi) - 1)*(
    #             func_of_freq(phip))
    #         +
    #         (+coth_of_args_and(phip) + 1)*(
    #             func_of_freq(phi))
    #         )
    #     )


    ## Handle sum over n differently depending on dimension of t
    if type(nu_n) == np.ndarray:
        nu_n = nu_n[:, None]
        # freqs = func_of_freq(nu_n)[:,None]

    sum_over_n = np.sum(
        (
            nu_n/(
                (omega_q**2. + nu_n**2.)**2.
                -
                gamma**2.*nu_n**2.
                )
            *
            func_of_freq(nu_n)
            ),
            axis=0
        )

    # last_term = (
    #     -
    #     2*gamma/beta
    #     *
    #     sum_over_n
    #     )
    last_term = (
        -
        2*gamma/beta
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


def j_star_of_omega(
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
            omega = omega[:, None]

        return np.exp(1j*(
            omega+(script_d**2.*omega_q)
            )*t*1e-15 - np.conj(g(
                t,
                script_d=script_d,
                omega_q=omega_q,
                **kwargs)))

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

    result = np.real(
        np.asarray(
            j_star_of_omega(
                omega,
                script_d,
                omega_q=omega_q,
                t_bound=t_bound,
                t_points=t_points,
                return_integrand=return_integrand,
                **kwargs)))

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


class mol_fluo_model(object):

    def __init__(self,
        num_vib_modes,
        hbar_omega_eg_0,
        script_d,
        hbar_omega_0,
        hbar_gamma,
        T,):
        """ Multimode displaced oscillator model from Mukamel with
            coupling to bath.

            Args:
                num_vib_modes: type int. Number of vibrational modes.
                    Used to check other args for consistent numbers of
                    parameters.
                hbar_omega_eg_0: type(float). Zero point energy shared
                    by all vibrational modes.
                script_d: type(array of length 'num_vib_modes').
                    Displacement of equalibrium position of vibrational
                    mode in the electronic excited state.
                hbar_omega_0: type(array of length 'num_vib_modes').
                    Vibrational frequency of the uncoupled modes.
                hbar_gamma: type(array of length 'num_vib_modes').
                    Effective damping resulting from coupling to bath.
                T: type(float). Absolute temperature.
            """
        ## Check consistent number of parameters
        for param_list in [script_d, hbar_omega_0, hbar_gamma,]:
            if len(param_list) != num_vib_modes:
                raise ValueError("Input args are not consistent length with 'num_vib_modes'")

        ## Store args as instance attributes
        self.num_vib_modes = num_vib_modes
        if type(hbar_omega_eg_0) is not float:
            raise ValueError('hbar_omega_eg_0 must be float')
        self.hbar_omega_eg_0 = hbar_omega_eg_0
        self.script_d = script_d
        self.hbar_omega_0 = hbar_omega_0
        self.hbar_gamma = hbar_gamma
        self.T = T

    def omega_eg(self,
        mode_idx=None,
        ):
        """Average energy gap"""
        if mode_idx is not None:
            _script_d = self.script_d[mode_idx]
            _hbar_omega_0 = self.hbar_omega_0[mode_idx]
            _hbar_gamma = self.hbar_gamma[mode_idx]

            hbar_omega_eg = (
                self.hbar_omega_eg_0
                +
                1/2 * _script_d**2. * _hbar_omega_0
                )

        if mode_idx is None:
            ## Initialize the average frequency before summinbe over
            ## vibrational modes.
            hbar_omega_eg = self.hbar_omega_eg_0
            ## Iterate through vibrational modes and sum g_i
            for i in range(self.num_vib_modes):

                _script_d = self.script_d[i]
                _hbar_omega_0 = self.hbar_omega_0[i]
                _hbar_gamma = self.hbar_gamma[i]

                hbar_omega_eg += 1/2 * _script_d**2. * _hbar_omega_0

        return hbar_omega_eg/hbar


    def _correlation_fun_root(self,
        func_of_freq,
        script_d,
        omega_q,
        gamma,
        T,
        ns=np.linspace(1, 10, 10),
        take_conjugate=False):

        return correlation_fun_root(
            func_of_freq=func_of_freq,
            script_d=script_d,
            omega_q=omega_q,
            gamma=gamma,
            T=T,
            ns=ns,
            take_conjugate=take_conjugate,
            )


    def g(self,
        t,
        mode_idx=None,
        ns=np.linspace(1,10, 10),
        ):

        def single_mode_g(t, script_d, omega_q, gamma, T, ns):

            t = t*1e-15

            def dub_t_int_exp_iphi(p):
                return (np.exp(-p*t) + p*t - 1)/p**2.

            goft = self._correlation_fun_root(
                dub_t_int_exp_iphi,
                script_d=script_d,
                omega_q=omega_q,
                gamma=gamma,
                T=T,
                ns=ns
                )

            return goft

        if mode_idx is not None:
            _script_d = self.script_d[mode_idx]
            _hbar_omega_0 = self.hbar_omega_0[mode_idx]
            _hbar_gamma = self.hbar_gamma[mode_idx]

            _g = single_mode_g(
                t,
                script_d=_script_d,
                omega_q=_hbar_omega_0/hbar,
                gamma=_hbar_gamma/hbar,
                T=self.T,
                ns=ns,
                )

        if mode_idx is None:

            _g = np.zeros(t.shape, dtype='complex')
            ## Iterate through vibrational modes and sum g_i
            for i in range(self.num_vib_modes):

                _script_d = self.script_d[i]
                _hbar_omega_0 = self.hbar_omega_0[i]
                _hbar_gamma = self.hbar_gamma[i]

                _g += single_mode_g(
                    t,
                    script_d=_script_d,
                    omega_q=_hbar_omega_0/hbar,
                    gamma=_hbar_gamma/hbar,
                    T=self.T,
                    ns=ns,
                    )

        return _g


    def _lineshape(
        self,
        omega,
        which_lineshape=None,
        mode_idx=None,
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
        if (
            (which_lineshape is not "emission")
            and
            (which_lineshape is not "absorption")
            ):
            raise ValueError(
                "Must set arg 'which_lineshape' to 'emission' or 'absorption'."
                )


        ## Shift frequency by the average
        omega_eg = self.omega_eg(mode_idx)
        omega_m_omega_eq = omega - omega_eg

        ## generate time domain for integration
        t = np.linspace(0, t_bound, t_points)

        ## Define g depending on mode index
        _g = self.g(
            t, mode_idx=mode_idx, **kwargs)

        ## Set varibles specific to emission
        if which_lineshape is 'emission':

            ## Shift spectrum to other side of symmetry point (omega_eg^0)
            lamb = omega_eg - self.hbar_omega_eg_0/hbar
            omega_m_omega_eq += 2*lamb

            ## take complex conjugate of linebroading function
            _g = np.conj(_g)

        ## Get integrand from method
        def integrand(t):
            return self._integrand(t, omega_m_omega_eq, _g)

        if return_integrand:
            result = [t, integrand(t)]

        else:
             ## integrate last dimension of array (time)
            integral = inte.trapz(integrand(t), t, axis=-1)
            result = integral

        return (result)

    def _integrand(self,
        t,
        omega_m_omega_eq,
        g):

        if type(omega_m_omega_eq) is np.ndarray:
            ## Have to replace the (1/pi) prefactor with (1/2) in order to
            ## get an area normalized result. Don't know why.
            return (1/2) * np.real(
                np.exp(
                    (1j*(omega_m_omega_eq[:, None])*t[None, :]*1e-15)
                    -
                    g)
                )

        else:
            return (1/2) * np.real(
                np.exp(1j*(omega_m_omega_eq)*t*1e-15 - g))

    def emission_lineshape(
        self,
        omega,
        mode_idx=None,
        t_bound=5,
        t_points=100,
        return_integrand=False,
        **kwargs):
        """ Fourier transforms the dipole self correlation function in
            the Cumulant expansion. Time bounds for integrals are in
            units of 1e-15.
            """

        return self._lineshape(
            omega,
            which_lineshape='emission',
            mode_idx=mode_idx,
            t_bound=t_bound,
            t_points=t_points,
            return_integrand=return_integrand,
            **kwargs)

    def absorption_lineshape(
        self,
        omega,
        mode_idx=None,
        t_bound=5,
        t_points=100,
        return_integrand=False,
        **kwargs):

        return self._lineshape(
            omega,
            which_lineshape='absorption',
            mode_idx=mode_idx,
            t_bound=t_bound,
            t_points=t_points,
            return_integrand=return_integrand,
            **kwargs)


class anda_mol_fluo_model(mol_fluo_model):

    def __init__(self,
        num_vib_modes,
        hbar_omega_eg_0,
        script_d,
        hbar_omega_0,
        hbar_gamma,
        T,):
        """ This model is a simplified version of the version from
            Mukamel where the damping is inserted phenominalogically
            by multiplying the single frequecy linear response frunction
            by e^(gamma t) at the level of the Fourier Transform.

            Multimode displaced oscillator model from Mukamel with
            coupling to bath.

            Args:
                num_vib_modes: type int. Number of vibrational modes.
                    Used to check other args for consistent numbers of
                    parameters.
                hbar_omega_eg_0: type(float). Zero point energy shared
                    by all vibrational modes.
                script_d: type(array of length 'num_vib_modes').
                    Displacement of equalibrium position of vibrational
                    mode in the electronic excited state.
                hbar_omega_0: type(array of length 'num_vib_modes').
                    Vibrational frequency of the uncoupled modes.
                hbar_gamma: type(array of length 'num_vib_modes').
                    Effective damping resulting from coupling to bath.
                    BE WARMED THAT THIS MODEL ONLY INTERPRETS THE FIRST
                    ELEMENT OF THE GAMMA LIST SINCE THE MODEL REQUIRES
                    IDENTICAL LINEWIDTHS PER VIBRATIONAL MODE.
                T: type(float). Absolute temperature.
            """
        mol_fluo_model.__init__(self,
            num_vib_modes,
            hbar_omega_eg_0,
            script_d,
            hbar_omega_0,
            hbar_gamma,
            T)

        ## Make sure gamma is an number and that all gammas given are
        ## the same.
        if (type(hbar_gamma) is np.ndarray) or type(hbar_gamma) is list:
            if np.all(np.asarray(hbar_gamma) != hbar_gamma[0]):
                raise ValueError("All values of gamma must be the same")

    ## Redefine the correlation function to that with no bath coupling
    def _correlation_fun_root(self,
        func_of_freq,
        script_d,
        omega_q,
        gamma,
        T,
        ns=np.linspace(1, 10, 10),
        take_conjugate=False):
        """ The generalized 'func_of_freq' are accounts for integration
            of the complex harmonic time dependence, so terms in the
            correlation function like
                e^-iwt -> func_of_freq(1j*omega_q)
                e^+iwt -> func_of_freq(-1j*omega_q)
            """

        beta = 1/(kb*T)

        n_bar = (np.exp(beta*hbar*omega_q) - 1)**(-1)

        coft = (script_d**2 * omega_q**2. / 2)*(
            (n_bar+1)*func_of_freq(1j*omega_q)
            +
            n_bar*func_of_freq(-1j*omega_q)
            )

        return coft

    ## Multiply inegrand of fourier transform by the decaying exponential
    def _integrand(self,
        t,
        omega_m_omega_eq,
        g):

        gamma = np.asarray(self.hbar_gamma)/hbar
        if (type(self.hbar_gamma) is np.ndarray) or (
            type(self.hbar_gamma) is list):
            gamma = gamma[0]

        if type(omega_m_omega_eq) is np.ndarray:
            ## Have to replace the (1/pi) prefactor with (1/2) in order to
            ## get an area normalized result. Don't know why.
            return 2**-1 * np.real(
                np.exp(
                    (1j*(omega_m_omega_eq[:, None])*t[None, :]*1e-15)
                    -
                    g)
                *
                np.exp(
                    (-gamma*t[None, :]*1e-15))
                )

        else:
            return 2**-1 * np.real(
                np.exp(1j*(omega_m_omega_eq)*t*1e-15 - g)
                *
                np.exp(
                    (-gamma*t[None, :]*1e-15))
                )


















## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Anharmonic implementation from Anda's 2016 JCTC
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def taylor_expm(A, n):
    e_A = np.identity(A.shape[-1], dtype='complex')
    if A.ndim is 3:
        e_A = e_A[None, ...]
    for i in range(1, n):
        e_A = e_A + (1/spl.factorial(i))*np.linalg.matrix_power(A, i)
    return e_A


def eigy_expm(A):
    vals,vects = np.linalg.eig(A)
    return np.einsum('...ik, ...k, ...kj ->...ij',
                     vects,np.exp(vals),np.linalg.inv(vects))

def loopy_expm(A):
    expmA = np.zeros_like(A)
    for n in range(A.shape[0]):
        expmA[n,...] = lin.expm(A[n,...])
    return expmA

def displaced_lambdas(
    lambda_array,
    d,
    include_H0_contri=True):
    """ Given the polynomial or order n = len('lambda_array')-1
            lambda_array[0]
            +
            lambda_array[1] * x
            +
            lambda_array[2] * x**2
            ...
            +
            lambda_array[-1] * x**n
        this function returns the coefficients upon introducing the
        displacement
            x -> x-d
        """
    n = len(lambda_array) - 1

    lambda_matrix = np.tri(n+1)
    lambda_matrix *= np.asarray(lambda_array)[:, None]

    ## For each term in the polynomial, we need to expand the binolial
    ## and resort coefficients into lambda array.
    for m in range(0, n+1):
         for k in range(0, m):
            lambda_matrix[m, k] *= (
                spl.factorial(m)/(spl.factorial(k)*spl.factorial(m-k))
                )*(-d)**(m-k)
    new_lambdas = np.sum(lambda_matrix, axis=0)

    if include_H0_contri:
        ## Add contribution from displacing H_0
        new_lambdas[:2] += [d**2/2, -d]

    return new_lambdas

def displaced_ham_e(self, H_g, zp_energy, d):
        """ Returns matrix representation of (unitless) nuclear Hamiltonian
            in the electronic ground state for the general anharmonic
            oscillator truncated at dimension 'basis_size'.

            The arg 'lambda_array' is expected to be 1D and contains the
            scalar prefactors at each order of the polynomial potential
            energy surface. The potential is therefore a polynomial of order
            len(lambda_array) + 1.
            """
        basis_size = H_g.shape[0]
        a = np.zeros((basis_size, basis_size))
        for n in range(basis_size-1):
            a[n, n+1] = np.sqrt(n+1)
        displacement_op = lin.expm(d*a.T - d*a)

        H_e = (
            displacement_op @ H_g @ displacement_op.T
            +
            zp_energy * np.identity(basis_size)
            )
        return H_e

def poly_from_lambdas(x, lambdas):

    y = np.zeros_like(x)
    for i, lam in enumerate(lambdas):
        y += lam * x**i
    ## Add back in harmonic piece
    y += x**2/2

    return y


class anharmonic_mat_exp_implementation(object):
    """ Implementation of the generalized solution to the displaced
        oscillator problem for arbitrary polynomial vibrational energy
        surfaces in both the ground and excited electronic states.
        Following the 2016 JCTC by Anda et al. (With some corrections
        for their cavalier treatment of units).

        Args:
        ~~~~~
            poly_prefs_g: (Array of length equal to polynomial order - 1)
                The prefacters to the perterbing potential surface,
                notated as lambda in the JCTC. They describe the vibrational
                potential surface for the electronic ground state minus the
                harmonic term. They are organized such that the [i] element is
                the ith order term of the polynomial
            poly_prefs_e: (same as above)
                Similarly, these are the lambda prefactors for the vibrational
                potential in the electronic excited state.
            basis_size: (int)
                The number of harmonic oscillator states used to represent the
                anharmonic eigenstates
            hbarw0: (float)
                The oscillatorion energy of the unperterbed harmonic oscillator
                potential. Any nonzero 'poly_prefs_g[2]' or 'poly_prefs_e[2]'
                will effectivly shift this value. Because of that subdlty, it
                is primarily used to dimensionalize/nondimensionalize various
                quantities like that displacement below.
            unitless_d: (float)
                nondimensionalized displacement between the equilibrium
                positions (minimum) of the vibrational potential energy
                surfaces of the ground and excited electronic states. For a
                harmonic system (), It is related to the Huang-Rys factor by
                    S = unitless_d^2 / 2
            T: (float)
                Temperature in Kelvin

        """

    def __init__(self,
        poly_prefs_g,
        poly_prefs_e,
        basis_size,
        hbarw0,
        hbar_gamma,
        unitless_d,
        T,
        integration_t_max=20,
        integration_t_points=600,
        calc_matricies_on_init=True,
        A_mat_order=7,
        ):

        self.poly_prefs_g = poly_prefs_g
        self.poly_prefs_e = poly_prefs_e
        self.basis_size = basis_size
        self.hbarw0 = hbarw0
        self.hbar_gamma = hbar_gamma
        self.unitless_d = unitless_d
        self.T = T
        self.integration_t_max = integration_t_max
        self.integration_t_points = integration_t_points

        ## Calculate 0-0 transition energy for fixing mysterios spectral offset
        self.hbar_omega_eg = self.calc_hbar_omega_eg(
            basis_size=self.basis_size,
            lambda_g_array=self.poly_prefs_g,
            lambda_e_array=self.poly_prefs_e,
            T=self.T
            )

        if calc_matricies_on_init:
            self.calc_matricies(A_mat_order)

    def calc_matricies(self, A_mat_order):

        self.H_g = (
            self.vib_ham(self.poly_prefs_e, self.basis_size)
            *self.hbarw0
            )

        self.H_e = (
            self.vib_ham(self.poly_prefs_e, self.basis_size)
            *self.hbarw0
            )

        self.Delta = self.gap_fluc_op(
            basis_size=self.basis_size,
            lambda_g_array=self.poly_prefs_g,
            lambda_e_array=self.poly_prefs_e,
            T=self.T
            )*self.hbarw0

        self.A_mat_order = A_mat_order
        self.A = self.a_matrix(self.H_e, self.Delta, order=self.A_mat_order)

        self.rho_g = self.rho(self.H_g, self.T)
        self.rho_e = self.rho(self.H_e, self.T)


    def position_operator(self, basis_size=None):
        """ Returns matrix representation of the position operator in the
            basis of number states. Truncated at dimension 'basis size'.
            """
        if basis_size is None:
            basis_size = self.basis_size

        x = np.zeros((basis_size, basis_size))
        for n in range(basis_size-1):
            x[n, n+1] = 1/2**0.5 * np.sqrt(n+1)
            x[n+1, n] = 1/2**0.5 * np.sqrt(n+1)
        return x

    def x_tothe_k(self, k=None, basis_size=None):
        if k is None:
            k = self.k
            if basis_size is None:
                basis_size = self.basis_size
        x = self.position_operator(basis_size)
        return np.linalg.matrix_power(x, k)

    def vib_ham(self, lambda_array, basis_size=None):
        """ Returns matrix representation of (unitless) nuclear Hamiltonian
            in the electronic ground state for the general anharmonic
            oscillator truncated at dimension 'basis_size'.

            The arg 'lambda_array' is expected to be 1D and contains the
            scalar prefactors at each order of the polynomial potential
            energy surface. The potential is therefore a polynomial of order
            len(lambda_array) + 1.
            """
        if basis_size is None:
            basis_size = self.basis_size

        H_prime = np.zeros((basis_size, basis_size))
        ## Define unperterbed harmonic hamiltonian
        H_0 = np.diag(np.arange(0, basis_size))

        poly_order_plus_1 = len(lambda_array)
        for order in range(poly_order_plus_1):
            H_prime += lambda_array[order]*self.x_tothe_k(order, basis_size)

        H_g = H_0 + H_prime
    #     H_g -= np.identity(basis_size)*lambda_array[0]
        return H_g


    def rho(self, H_e, T=None):
        if T is None:
            T = self.T
        if T is 0:
            return np.identity(H_e.shape[0])
        density_matrix_e = lin.expm(-H_e/(kb*T))
        density_matrix_e /= np.trace(density_matrix_e)
        return density_matrix_e


    def gap_fluc_op(self, basis_size, lambda_g_array, lambda_e_array, T):
        """ Returns energy gap fluctuation operator """
        density_matrix_e = self.rho(self.vib_ham(lambda_e_array, basis_size), T=T)

        delta = np.zeros((basis_size, basis_size))
        for n in range(len(lambda_e_array)):
            xk = self.x_tothe_k(n, basis_size)
            delta += (
                (lambda_g_array[n] - lambda_e_array[n])
                *
                (
                    xk
                    -
                    np.identity(basis_size)
                    *np.trace(xk@density_matrix_e)
                    )
                )
        return delta


    def calc_hbar_omega_eg(self, basis_size, lambda_g_array, lambda_e_array, T):
        density_matrix_g = self.rho(self.vib_ham(lambda_g_array, basis_size), T=T)

        hbar_omega_eg = 0
        for n in range(len(lambda_e_array)):
            xk = self.x_tothe_k(n, basis_size)
            hbar_omega_eg += (
                (lambda_e_array[n] - lambda_g_array[n])
                *
                -np.trace(xk@density_matrix_g)
                )
        return hbar_omega_eg


    def a_matrix(self, H_e, Delta, order):
        miHe = -1j*H_e/hbar
        A = np.block([
            [miHe,             Delta,  ],
            [np.zeros_like(Delta), miHe,]
            ])
        for o in range(1, order-1):
            right_col = np.block([np.zeros_like(Delta)]*o + [Delta]).T
            bottom_row = np.block([np.zeros_like(Delta)]*(o+1) + [miHe])

            A = np.block([
                [A, right_col],
                [bottom_row]
                ])
        return A





    def big_b_tilde(self,
        t,
        order_n,
        H_e,
        rho_ex,
        A=None,
        e_At=None,
        return_e_At_in_dict=False):

        """ Return shorthand B_n coefficient from Anda's paper
                B_n = (i/hbar)^n * Tr{e^-iH_et [e^At]_[N-n, N] * rho_ex}
            """
    #     print(f"Inside Bn, t = {t}")
        if type(t) is np.ndarray:
            H_e = H_e[None, ...]
            t = t[:, None, None]
            if A is not None:
                A = A[None, ...]
            exp_func = loopy_expm
    #         exp_func = lambda a: taylor_expm(a, 100)
        else:
            exp_func = lin.expm
    #     print(f"Shape of H_e inside B_n = {H_e.shape}")
    #     print(f"Shape of A inside B_n = {A.shape}")

        ## First step compute matrix exponentials at time t
        e_iHet = exp_func(1j*H_e*t/hbar)
    #     print(f"e_iHet = {e_iHet}")
        if A is not None:
            e_At = exp_func(A*t)

    #     print(f"e_At = {e_At}")
        ## Navigate block array
        block_size = H_e.shape[-1]

        N = round(e_At.shape[-1]/block_size)
    #     print(f" A block indexed from {(N-order_n-1)} to {N-order_n}")
        Bn_array = (
            (
                -1j
                /
                hbar
                )**order_n
            *
            e_iHet
            @
            e_At[
                ...,
                (N-order_n-1)*block_size:(N-order_n)*block_size,
                (N-1)*block_size:
                ]
            @
            rho_ex
            )
        Bn = np.trace(Bn_array, axis1=-2, axis2=-1)
        if return_e_At_in_dict:
            return {'B_n':Bn, 'e_At':e_At}
        return Bn


    def sum_of_cumulants(self, _t, H_e, rho_ex, A):
        """ Returns sum of Cumulants calculates using the matrix theorm implemetation """
        B_2_dict = self.big_b_tilde(
                _t,
                order_n=2,
                H_e=H_e,
                rho_ex=rho_ex,
                A=A,
                e_At=None,
                return_e_At_in_dict=True)
        B_2 = B_2_dict['B_n']
        e_At = B_2_dict['e_At']
#         print(f"e_At = {e_At}")
        B_3 = self.big_b_tilde(
            _t,
            order_n=3,
            H_e=H_e,
            rho_ex=rho_ex,
#             A = A,
            e_At=e_At,
            )
        B_4 = self.big_b_tilde(
            _t,
            order_n=4,
            H_e=H_e,
            rho_ex=rho_ex,
            e_At=e_At,)
        B_5 = self.big_b_tilde(
            _t,
            order_n=5,
            H_e=H_e,
            rho_ex=rho_ex,
            e_At=e_At,)
        B_6 = self.big_b_tilde(
            _t,
            order_n=6,
            H_e=H_e,
            rho_ex=rho_ex,
            e_At=e_At,)
#         print(f"B_2 = {B_2}")
#         print(f"B_3 = {B_3}")
        exp_arg = (
            B_2
            +
            B_3
            ## Stop here fore 3rd order Cum. expansion
            +
            B_4
            -
            (B_2**2)/2
            ## Stop here for 4th order Cum. expansion
            +
            B_5
            -
            B_2*B_3
            ## Stope here fore 5th order Cum. expansion
            +
            B_6
            -
            B_2*B_4
            -
            (B_3**2)/2
            +
            (B_2**3)/3
            )

        return exp_arg


    def integrand(self, _t, omega, gamma, H_e, rho_ex, A, which_linespace='emission'):

        exp_arg = self.sum_of_cumulants(_t, H_e, rho_ex, A)

        ## Return integrand with time on last dimensions and omegas on
        ## first.
        if which_linespace == 'emission':
            omega += self.hbar_omega_eg
        elif which_linespace == 'absorption':
            omega -= self.hbar_omega_eg
        else:
            raise ValueError("which_linespace arg must specify 'absorption' or 'emission'")

        _integrand = np.real(
            np.exp(
                (-1j*omega[:, None] - gamma)*_t[None, :]
                +
                exp_arg[None, :])
                )
    #         print(f"integrand = {_integrand}")
        return _integrand

    def _flu_lineshape(
        self,
        omega,
        gamma,
        H_e,
        rho_ex,
        A,
        t_max=1000,
        t_points=100,
        return_integrand=False):
        """ Implement equation 31 from Anda without plugged in class instance attributes """

        ## Build t vector
        ts = np.linspace(0, t_max, t_points)*1e-15

        if return_integrand:
            return (ts, self.integrand(ts, omega, gamma, H_e, rho_ex, A))
        ## Integrate with trapazoid rule
        integral = inte.trapz(self.integrand(ts, omega, gamma, H_e, rho_ex, A), ts, axis=-1)

        ## Integrate with scipy quadriture function
    #     integral = integ.quad(integrand, 0, t_max*1e-15
        ## Integrate by direct Rieman sum
    #     integral = np.zeros(len(omega))
    #     for t in ts:
    #         integral += integrand(t)
    #     integral *= ts[1]-ts[0]
        return integral/(2*np.pi*hbar)

    def emission_lineshape(self,
        hbar_omegas,
        return_integrand=False):
        """ Implement equation 31 from Anda with parameters taken from
            class instance attributes """
        return self._flu_lineshape(
            omega=hbar_omegas/hbar,
            gamma=self.hbar_gamma/hbar,
            H_e=self.H_e,
            rho_ex=self.rho_e,
            A=self.A,
            t_max=self.integration_t_max,
            t_points=self.integration_t_points,
            return_integrand=return_integrand)


class multi_mode_anharmonic_emission(anharmonic_mat_exp_implementation):
    """ Implementation of the generalized solution to the displaced
        oscillator problem for arbitrary polynomial vibrational energy
        surfaces in both the ground and excited electronic states.
        Following the 2016 JCTC by Anda et al. (With some corrections
        for their cavalier treatment of units).

        Args:
        ~~~~~
            poly_prefs_g: (Array of length equal to polynomial order - 1)
                The prefacters to the perterbing potential surface,
                notated as lambda in the JCTC. They describe the vibrational
                potential surface for the electronic ground state minus the
                harmonic term. They are organized such that the [i] element is
                the ith order term of the polynomial
            poly_prefs_e: (same as above)
                Similarly, these are the lambda prefactors for the vibrational
                potential in the electronic excited state.
            basis_size: (int)
                The number of harmonic oscillator states used to represent the
                anharmonic eigenstates
            hbarw0: (float)
                The oscillatorion energy of the unperterbed harmonic oscillator
                potential. Any nonzero 'poly_prefs_g[2]' or 'poly_prefs_e[2]'
                will effectivly shift this value. Because of that subdlty, it
                is primarily used to dimensionalize/nondimensionalize various
                quantities like that displacement below.
            unitless_d: (float)
                nondimensionalized displacement between the equilibrium
                positions (minimum) of the vibrational potential energy
                surfaces of the ground and excited electronic states. For a
                harmonic system (), It is related to the Huang-Rys factor by
                    S = unitless_d^2 / 2
            T: (float)
                Temperature in Kelvin

        """

    def __init__(self,
        poly_prefs_g,
        poly_prefs_e,
        basis_size,
        hbarw0,
        hbar_gamma,
        unitless_d,
        T,
        integration_t_max=20,
        integration_t_points=600,
        # calc_matricies_on_init=True,
        A_mat_order=7,
        ):

        ## Lambdas will be expected to have n columns for an n-1 order polynomial
        ## and m rows for m different vibrational modes
        if poly_prefs_g.shape != poly_prefs_e.shape:
            raise ValueError('lambda_g and lambda_e must have same shape')
        self.poly_prefs_g = poly_prefs_g
        self.poly_prefs_e = poly_prefs_e

        self.basis_size = basis_size
        ## The number of vibrational energies should match the number of modes
        if len(hbarw0) != poly_prefs_g.shape[0]:
            raise ValueError('number of vib. energies much match number of rows in lambda arrays')
        self.hbarw0 = hbarw0

        self.hbar_gamma = hbar_gamma
        if len(unitless_d) != len(hbarw0):
            raise ValueError('must have same number of displacements as vibrational energies')
        self.unitless_d = unitless_d

        self.T = T

        self.integration_t_max = integration_t_max
        self.integration_t_points = integration_t_points

        self.num_modes = self.poly_prefs_g.shape[0]

        # if calc_matricies_on_init:
        self.calc_matricies(A_mat_order)

        ## Calculate 0-0 transition energy for fixing mysterios spectral offset
        self.hbar_omega_eg = self.calc_hbar_omega_eg(
            basis_size=self.basis_size,
            lambda_g_array=self.poly_prefs_g,
            lambda_e_array=self.poly_prefs_e,
            T=self.T
            )

    ## I think I need to rewrite
    ## calculation of hweg for shift correction
    ## - calc_matricies
    ##     - to calculate matricies for each mode, maybe store in a new dimention
    ## - lineshape to work with new dimention in matricies
    ## - integrand
    ##     - to sum cumulant expantions for each mode
    ##     - also to shift each mode by hw_eg (weird because thay should all be hte same...)
    def calc_hbar_omega_eg(self, basis_size, lambda_g_array, lambda_e_array, T):
        """ Needs density matrix defined , unlike single mode implementation"""
        density_matrix_e = self.rho_e

        hbar_omega_eg = np.zeros(self.num_modes, dtype='complex')
        for i in range(self.num_modes):
            for n in range(lambda_e_array.shape[1]):
                xk = self.x_tothe_k(n, basis_size)
                hbar_omega_eg[i] += (
                    (lambda_e_array[i, n] - lambda_g_array[i, n])
                    *
                    -np.trace(xk@density_matrix_e[i])
                    )
        return hbar_omega_eg
###########

    def calc_matricies(self, A_mat_order):

        self.H_e = np.empty((
            self.num_modes, self.basis_size, self.basis_size), dtype='complex')
        self.Delta = np.empty((
            self.num_modes, self.basis_size, self.basis_size), dtype='complex')
        self.A = np.empty((
            self.num_modes,
            self.basis_size*A_mat_order,
            self.basis_size*A_mat_order), dtype='complex')
        self.rho_e = np.empty((
            self.num_modes, self.basis_size, self.basis_size), dtype='complex')

        ## Iterate through mode indicies
        for i in range(self.num_modes):
            ## Ground state not needed for emission outside gap operater
            # self.H_g = self.vib_ham(self.poly_prefs_g, self.basis_size)
            self.H_e[i] = self.vib_ham(
                self.poly_prefs_e[i],
                self.basis_size
                )*self.hbarw0[i]

            self.Delta[i] = self.gap_fluc_op(
                basis_size=self.basis_size,
                lambda_g_array=self.poly_prefs_g[i],
                lambda_e_array=self.poly_prefs_e[i],
                T=self.T
                )*self.hbarw0[i]

            self.A_mat_order = A_mat_order
            self.A[i] = self.a_matrix(self.H_e[i], self.Delta[i], order=self.A_mat_order)

            self.rho_e[i] = self.rho(self.H_e[i], self.T)


    def calculate_cumulants(self, _t, H_e, rho_ex, A):

        if not hasattr(self, 'cum_sum'):
            # print('_t = ', _t)
            self.cum_sum = np.zeros((self.num_modes, len(_t),), dtype='complex')
            # print('self.cum_sum.shape = ', self.cum_sum.shape)
            # print('self.cum_sum.shape[0] = ', self.cum_sum.shape[0])
            ## Calculate for each mode
            for i in range(self.num_modes):
                self.cum_sum[i] = self.sum_of_cumulants(_t, H_e[i], rho_ex[i], A[i])
        else: pass


    def integrand(self, _t, omega, gamma, H_e, rho_ex, A, isolate_mode=None, which_linespace='emission'):

        self.calculate_cumulants(_t, H_e, rho_ex, A)
        ## Return integrand with time on last dimensions and omegas on
        ## first.

        if isolate_mode is None:
            ## Sum Cululants from each mode
            cum_sum = self.cum_sum.sum(axis=0)
            ## Sum w_eg from each mode
            hbar_omega_eg = self.hbar_omega_eg.sum()
        ## If we want a particular mode
        elif type(isolate_mode) is int:
            ## Get specific Cumulant sum
            cum_sum = self.cum_sum[isolate_mode]
            ## Get specific w_eg
            hbar_omega_eg = self.hbar_omega_eg[isolate_mode]

        ## Shift frequency axis to 0-0 line according to which lineshape we want
        ## According to calculations prescribed by Mukamel in the case
        ## where w_eg_0 (the energy difference between minima) is zero
        ## for plotting convinience (and fitting, since it is just an
        ## overall shift).
        if which_linespace == 'emission':
            omega += hbar_omega_eg
        elif which_linespace == 'absorption':
            omega -= hbar_omega_eg
        else:
            raise ValueError("which_linespace arg must specify 'absorption' or 'emission'")

        ## Build integrand array in frequency and time
        _integrand = np.real(
            np.exp(
                (-1j*omega[:, None] - gamma)*_t[None, :]
                +
                cum_sum[None, :])
                )
        return _integrand

