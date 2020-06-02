import numpy as np

# from ..calc import BEM_simulation_wrapper as bem
# from ..calc import fitting_misLocalization as fit
# from ..calc import coupled_dipoles as cp

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
            return 1/2 * np.real(
                np.exp(
                    (1j*(omega_m_omega_eq[:, None])*t[None, :]*1e-15)
                    -
                    g)
                )

        else:
            return 1/2 * np.real(
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
        if (type(self.hbar_gamma) is np.ndarray) or (type(self.hbar_gamma) is list):
            gamma = gamma[0]

        if type(omega_m_omega_eq) is np.ndarray:
            return np.pi**-1 * np.real(
                np.exp(
                    (1j*(omega_m_omega_eq[:, None])*t[None, :]*1e-15)
                    -
                    g)
                *
                np.exp(
                    (-gamma*t[None, :]*1e-15))
                )

        else:
            return np.pi**-1 * np.real(
                np.exp(1j*(omega_m_omega_eq)*t*1e-15 - g)
                *
                np.exp(
                    (-gamma*t[None, :]*1e-15))
                )



