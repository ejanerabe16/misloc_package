import numpy as np

from ..calc import BEM_simulation_wrapper as bem
from ..calc import fitting_misLocalization as fit
from ..calc import coupled_dipoles as cp

import scipy.constants as con
## Import physical constants
hbar = con.physical_constants['Planck constant over 2 pi in eV s'][0]
c = con.physical_constants['speed of light in vacuum'][0]*1e2 #cm/m
cm_per_nm = 1e-7
eps_water = 1.778

def sphere_fit_fun(hbarw,
    eps_inf,
    hbarw_p,
    hbargamma,
    a,
    eps_b=eps_water,
    model_cross_section=None):

    return model_cross_section(
        hbarw/hbar,
        eps_inf,
        hbarw_p/hbar,
        hbargamma/hbar,
        eps_b,
        a*cm_per_nm)


def sphere_dip_cost_fun(params, *args,
    eps_b=eps_water,
    data_scale_to_cgs=cm_per_nm**2,
    data_oom=1e-10,
    ):
    """ Cost function for fitting sphere dipole polarizability with a higher energy gaussian
        """
    ## Argumens contain independent and dependent varualbles.
    x=args[0]
    y=args[1]

    ## Define model of data
    yfit = (
        sphere_fit_fun(
            x,
            *params[:4],
            eps_b=eps_b,
            model_cross_section=cp.sigma_scat_ret_sphere
            )
        )

    y = y*(data_scale_to_cgs) ## to cm^2 from nm^2 (data from BEM)
    yfit = yfit ## in cm^2

    ## Rescale cost to order unity
    cost = (y-yfit) / data_oom

    return cost.ravel()

def sphere_dip_cost_fun__times_2(params, *args,
    eps_b=eps_water,
    data_scale_to_cgs=cm_per_nm**2,
    data_oom=1e-10,
    ):
    """ Cost function for fitting sphere dipole polarizability with a higher energy gaussian
        """
    ## Argumens contain independent and dependent varualbles.
    x=args[0]
    y=args[1]

    ## Define model of data
    yfit = (
        2*sphere_fit_fun(
            x,
            *params[:4],
            eps_b=eps_b,
            model_cross_section=cp.sigma_scat_ret_sphere
            )
        )

    y = y*(data_scale_to_cgs) ## to cm^2 from nm^2 (data from BEM)
    yfit = yfit ## in cm^2

    ## Rescale cost to order unity
    cost = (y-yfit) / data_oom

    return cost.ravel()


def sphere_Mie_cost_fun(params, *args,
    eps_b=eps_water,
    data_scale_to_cgs=cm_per_nm**2,
    data_oom=1e-10,
    ):
    """ Cost function for fitting sphere dipole polarizability with a higher energy gaussian
        """
    ## Argumens contain independent and dependent varualbles.
    x=args[0]
    y=args[1]

    ## Define model of data
    yfit = (
        sphere_fit_fun(
            x,
            *params[:4],
            eps_b=eps_b,
            model_cross_section=cp.sigma_scat_Mie_sphere
            )
        )

    y = y*(data_scale_to_cgs) ## to cm^2 from nm^2 (data from BEM)
    yfit = yfit ## in cm^2

    ## Rescale cost to order unity
    cost = (y-yfit) / data_oom

    return cost.ravel()


def sphere_TMatExp_cost_fun(params, *args,
    eps_b=eps_water,
    data_scale_to_cgs=cm_per_nm**2,
    data_oom=1e-10,
    ):
    """ Cost function for fitting sphere dipole polarizability with a higher energy gaussian
        """
    ## Argumens contain independent and dependent varualbles.
    x=args[0]
    y=args[1]

    ## Define model of data
    yfit = (
        sphere_fit_fun(
            x,
            *params[:4],
            eps_b=eps_b,
            model_cross_section=cp.sigma_scat_TMatExp_sphere
            )
        )

    y = y*(data_scale_to_cgs) ## to cm^2 from nm^2 (data from BEM)
    yfit = yfit ## in cm^2

    ## Rescale cost to order unity
    cost = (y-yfit) / data_oom

    return cost.ravel()


def hiengau_cost_func(
    params,
    *args,
    data_scale_to_cgs=cm_per_nm**2,
    data_oom=1e-10,
    model_cross_section=cp.sigma_scat_Mie_sphere):
    """ Cost function for fitting sphere dipole polarizability with a higher energy gaussian
        """
    ## Argumens contain independent and dependent varualbles.
    x=args[0]
    y=args[1]

    ## Parameteres;
    ##     eps_inf=params[0]
    ##     w_p=params[1]
    ##     gamma=params[2]
    ##     a = params[3]
    ##
    ## Gaussian parameters
    gau_amp = params[4]
    mu = params[5]
    std_dev = params[6]
    ## define a Gaussian and shrink it so that the amplitude is order unity
    high_en_gau = (
        gau_amp
        *
        np.exp(-(x-mu)**2./std_dev**2.)
        *
        data_oom
        )

    ## Define model of data
    yfit = (
        sphere_fit_fun(
            x,
            *params[:4],
            model_cross_section=model_cross_section)
        +
        high_en_gau
        )

    y = y*(data_scale_to_cgs) ## to cm^2 from nm^2 (data from BEM)
    yfit = yfit ## in cm^2

    ## Rescale cost to order unity
    cost = (y-yfit) / data_oom

    return cost.ravel()