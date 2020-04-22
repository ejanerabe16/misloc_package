""" for python 3


02/07/19:
    Updated to work in new folder structure for git, paths are now hardcoded
    and will later be set by instalation of Mislocalization package.


Stopped updating notes when I started using git.
"""
# import os
# import sys
# project_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(project_path)
import scipy.special as spl

from misloc_mispol_package import project_path

parameter_files_path = (
    project_path + '/param'
)
# sys.path.append(parameter_files_path)

import numpy as np
import yaml

phys_const_file_name = '/physical_constants.yaml'
opened_constant_file = open(
    parameter_files_path+phys_const_file_name,
    'r')
constants = yaml.load(opened_constant_file)
e = constants['physical_constants']['e']
c = constants['physical_constants']['c']  # charge of electron in statcoloumbs
hbar =constants['physical_constants']['hbar']
nm = constants['physical_constants']['nm']
n_a = constants['physical_constants']['nA']

# curly_yaml_file_name = '/curly_nrod_water_JC.yaml'

# print('reading parameters from {}'.format(
#     parameter_files_path+curly_yaml_file_name
#     )
# )

# opened_param_file = open(
#     parameter_files_path+curly_yaml_file_name,'r'
#     )
# parameters = yaml.load(opened_param_file)
# # print(curly_yaml_file_name)
# ## System background
# n_b = parameters['general']['background_ref_index']
# eps_b = n_b**2.

## Driving force
# ficticious_field_amp = parameters['general']['drive_amp']

# normalization = 1
# print('polarizability reduced by factor of {}'.format(normalization))
# print('coupling scaled up by by factor of {}'.format(normalization))



# ## prototyping a class structure for a polarizable object.

# class NanoParticle():
#     # this is a metalic, polarizable nanoparticle
#     # it is able to scatter incident light, and eventually will interacti with a molecule.

#     def __init__(self):
#         pass


## Adopted from old oscillator code
def fluorophore_mass(ext_coef, gamma, n_b):
    '''Derived at ressonance'''
    m = 4 * np.pi * e**2 * n_a  / (
            ext_coef * np.log(10) * c * n_b * gamma
            )
    return m

## Define polarizabilities in diagonal frames
def sparse_polarizability_tensor(mass, w_res, w, gamma_nr, a, eps_inf, eps_b):
    '''Define diagonal polarizability with single cartesien component derived
        from Drude model > Clausius-mosati.
        Assumes physical constants definded; e, c
    '''
    gamma_r = gamma_nr + (2*e**2./(3*mass*c**3.))*w**2.
    alpha_0_xx_osc = (e**2. / mass)/(w_res**2. - w**2. - 1j*gamma_r*w)
    alpha_0_xx_static = (a**3. * (eps_inf - 1*eps_b)/(eps_inf + 2*eps_b))
    # print('alpha_0_xx_static = ',alpha_0_xx_static)
    alpha_0_xx = alpha_0_xx_osc + alpha_0_xx_static

    if type(alpha_0_xx) is np.ndarray and alpha_0_xx.size > 1:
        alpha_0_xx = alpha_0_xx[..., None, None]

    alpha_0_ij = alpha_0_xx * np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ])

    return alpha_0_ij

def sparse_ellipsoid_polarizability(eps, eps_b, a_x, a_y, a_z):
    ''' '''
    def L_i(a, b, c):
        ''' assumes 'a' is ith radius for 'L_i'
        Don't think this will trivially converge'''
        q = np.linspace(0,10000,100000)*nm**2.
        ## these parameters led to ...
        ##... L_1 + L_2 + L_3 = ~.98 for my 44 x 20 x 20 nm rod
        ## not sure if they will
        fq = ( (a**2. + q) * (b**2. + q) * (c**2. + q) )**0.5
        integrand = 1/( (a**2. + q) * fq )
        integral = np.trapz(integrand, q)
        L_val = (a*b*c/2) * integral
        # print('L_val= ',L_val)
        return L_val

    def alpha_ii(a, b, c):
        alpha = a*b*c * (eps - eps_b)/(
            3*eps_b + 3*L_i(a,b,c)*(eps-eps_b)
            )
        return alpha

    alpha_1 = alpha_ii(a_x,a_y,a_z)
    alpha_2 = alpha_ii(a_y,a_z,a_x)
    alpha_3 = alpha_ii(a_z,a_x,a_y)

    alpha_ij = np.array([[alpha_1,      0.,      0.],
                         [     0., alpha_2,      0.],
                         [     0.,      0., alpha_3]])
    return alpha_ij

def drude_model(w, eps_inf, w_p, gamma):
    ''' '''
    eps = eps_inf - w_p**2./(w**2. + 1j*w*gamma)
    return eps

def sparse_ellipsoid_polarizability_drude(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_y, a_z):
    ''' '''
    return sparse_ellipsoid_polarizability(
        drude_model(w, eps_inf, w_p, gamma), eps_b, a_x, a_y, a_z)

def sigma_scat_spheroid(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_y, a_z):
    ''''''
    alpha = sparse_ellipsoid_polarizability_drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_y, a_z)

    sigma = (8*np.pi/3)*(w/c)**4.*(np.abs(alpha[0,0])**2.
        # + np.abs(alpha[1,1])**2.
        )
    # print('(8*np.pi/3)*(w/c)**4. = ',(8*np.pi/3)*(w/c)**4. )
    # print('(np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.) = ',
        # (np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.))
    return sigma


























################### retarded ellipsoid from Kong's notes

def sparse_ret_prolate_spheroid_polarizability(
    eps,
    eps_b,
    a_x,
    a_yz,
    w,
    isolate_mode=None):
    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        for the prolate spheroid (a_x < a_yz).

        'isolate_mode' takes args
            'long' : just x axis for prolate sphereoid (a_x > a_yz) or
                x and y axes for oblate spheroid (a_x < a_yz)
            'short' : y and x axes for prolate sphereoid (a_x > a_yz) or
                just z axis for oblate spheroid (a_x < a_yz)

        '''
    ### Define QS polarizability 'alphaR'
    def alphaR_ii(i, a_x, a_yz):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''

        alpha = ((a_x*a_yz**2.)/3) * (eps - eps_b)/(
            eps_b + L_i(i,a_x,a_yz)*(eps-eps_b)
            )

        return alpha

    ### Define static geometric factors 'L_i' and eccentricity 'ecc' required
    ### for quasistatic alpha 'alphaR'
    def L_i(i, a_x, a_yz):
        ''' '''
        def L_x(a_x, a_yz):
            e = ecc(a_x, a_yz)

            if a_x > a_yz:
                ## Use prolate result
                L = (1-e**2.)/e**3. * (-e + np.arctanh(e))
            elif a_x < a_yz:
                ## Use oblate spheroid result
                L = (1/e**2.)*(1- (np.sqrt(1-e**2.)/e)*np.arcsin(e))
            # print('L = ',L)
            return L

        def L_yz(a_x, a_yz):
            ''' 1 - L_x = 2*L_yx '''
            return (1 - L_x(a_x, a_yz))/2.

        if i == 1:
            L = L_x(a_x, a_yz)
        elif (i == 2) or (i == 3):
            L = L_yz(a_x, a_yz)

        return L

    def ecc(a_x, a_yz):
        return np.sqrt(np.abs(a_x**2. - a_yz**2.)/max([a_x, a_yz])**2.)

    ### Define retardation correction to alphaR
    def alphaMW_ii(i, a_x, a_yz):
        alphaR = alphaR_ii(i, a_x, a_yz)
        k = w*np.sqrt(eps_b)/c

        if i == 1:
            l_E = a_x
            D = D_x(a_x, a_yz)
        elif (i == 2) or (i == 3):
            l_E = a_yz
            D = D_yz(a_x, a_yz)

        alphaMW = alphaR/(
            1
            - (k**2./l_E) * D * alphaR
            - 1j * ((2*k**3.)/3) * alphaR
            )
        # print(f"alphaMW = {alphaMW}")
        return alphaMW

    ### Define dynamic geometric factors 'D_i' for alphaMW
    def D_x(a_x, a_yz):
        e = ecc(a_x, a_yz)
        if a_x > a_yz:
            ## Use prolate result
            D = 3/4 * (
                ((1+e**2.)/(1-e**2.))*L_i(1, a_x, a_yz) + 1
                )
        elif a_x < a_yz:
            ## Use oblate result
            D = 3/4 * ((1-2*e**2.)*L_i(1, a_x, a_yz) + 1)
        return D

    def D_yz(a_x, a_yz):
        e = ecc(a_x, a_yz)
        # print('e in D = ',e)
        if a_x > a_yz:
            D = (a_yz/(2*a_x))*(3/e * np.arctanh(e) - D_x(a_x,a_yz))
        elif a_x < a_yz:
            D = (a_yz/(2*a_x))*(
                3*np.sqrt(1-e**2.)/e * np.arcsin(e) - D_x(a_x,a_yz))
        # print('D = ',D)
        return D

    if a_x > a_yz:
        ## For prolate spheroid, assign long axis to be x
        alpha_11 = alphaMW_ii(1, a_x, a_yz)
        alpha_22 = alphaMW_ii(2, a_x, a_yz)
        alpha_33 = alphaMW_ii(3, a_x, a_yz)
    elif a_x < a_yz:
        ## For oblate spheroid, assign short axis to be z
        alpha_11 = alphaMW_ii(2, a_x, a_yz)
        alpha_22 = alphaMW_ii(2, a_x, a_yz)
        alpha_33 = alphaMW_ii(1, a_x, a_yz)

    if isolate_mode is None:
        alpha_ij = np.array([[ alpha_11,       0.,       0.],
                             [       0., alpha_22,       0.],
                             [       0.,       0., alpha_33]])

    elif isolate_mode is 'long':
        if a_x > a_yz:
            ## Keep only alpha_x for prolate
            alpha_ij = np.array([
                [ alpha_11,       0.,       0.],
                [       0.,       0.,       0.],
                [       0.,       0.,       0.]
                ])
        elif a_x < a_yz:
            ## Keep alpha_x and #alpha_y for oblate
            alpha_ij = np.array([
                [ alpha_11,       0.,       0.],
                [       0., alpha_22,       0.],
                [       0.,       0.,       0.]
                ])
    elif (isolate_mode is 'short') or (isolate_mode is 'trans'):
        if a_x > a_yz:
            alpha_ij = np.array([
                [       0.,       0.,       0.],
                [       0., alpha_22,       0.],
                [       0.,       0., alpha_33]
                ])
        elif a_x < a_yz:
            alpha_ij = np.array([
                [       0.,       0.,       0.],
                [       0.,       0.,       0.],
                [       0.,       0., alpha_33]
                ])

    # print(f"The output variable is {alpha_ij}.")
    return alpha_ij



def sparse_ret_prolate_spheroid_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz, isolate_mode=None):
    ''' '''
    return sparse_ret_prolate_spheroid_polarizability(
       drude_model(w, eps_inf, w_p, gamma), eps_b, a_x, a_yz, w, isolate_mode)

# For parameterization by spectra fit or modeling spectra
def sigma_prefactor(w, eps_b):
    """ added for debugging on 02/20/19 """
    n_b = np.sqrt(eps_b)
    prefac = (
        (8*np.pi/3)*(w * n_b/ c)**4.
        /(
        # 0.5
        # *
        n_b
        ) # copied from MNPBEM source
        )
    return prefac

def long_sigma_scat_ret_pro_ellip(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz):
    ''''''
    alpha = sparse_ret_prolate_spheroid_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[0,0])**2.
        )
    return sigma

def short_sigma_scat_ret_pro_ellip(w, eps_inf, w_p, gamma,
    eps_b, a_x, a_yz):
    ''''''
    alpha = sparse_ret_prolate_spheroid_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[1,1])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[1,1])**2.
        )
    # print('(8*np.pi/3)*(w/c)**4. = ',(8*np.pi/3)*(w/c)**4. )
    # print('(np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.) = ',
        # (np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.))
    return sigma

###################








## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Adapting the retarded ellipsoid section to be nonsingular for spheres
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sparse_ret_sphere_polarizability(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        '''
    ### Define QS polarizability 'alphaR'
    def alphaR_ii(i, a):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''

        alpha = ((a**3.)/3) * (eps - eps_b)/(
            eps_b + (1/3)*(eps-eps_b)
            )

        return alpha


    ### Define retardation correction to alphaR
    def alphaMW_ii(i, a):
        alphaR = alphaR_ii(i, a)
        k = w*np.sqrt(eps_b)/c

        alphaMW = alphaR/(
            1
            -
            ((k**2./a) * alphaR)
            -
            (1j * ((2*k**3.)/3) * alphaR)
            )

        return alphaMW

    alpha_11 = alphaMW_ii(1, a)
    alpha_22 = alphaMW_ii(2, a)
    alpha_33 = alphaMW_ii(3, a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def sparse_sphere_polarizability_TMatExp(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        '''
    ### Define QS polarizability 'alphaR'
    def alphaTME_ii(a):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''
        eps_r = eps / eps_b
        ka = w*np.sqrt(eps_b)/c * a

        alpha = (eps_r - 1)/(
            eps_r + 2
            -
            (6*eps_r - 12)*(ka**2./10)
            -
            1j*(2*ka**3./3)*(eps_r - 1)
            ) * a**3.

        return alpha

    alpha_11 = alphaTME_ii(a)
    alpha_22 = alphaTME_ii(a)
    alpha_33 = alphaTME_ii(a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def sparse_sphere_polarizability_Mie(
    eps,
    eps_b,
    a,
    w,
    isolate_mode=None,
    ):

    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
        '''
    ### Define QS polarizability 'alphaR'
    def alphaTME_ii(a):
        ''' returns components of alpha in diagonal basis with a_x denoting
            long axis
        '''
        eps_r = eps / eps_b
        m = np.sqrt(eps_r)
        k = w*np.sqrt(eps_b)/c
        x = k*a

        j1x = spl.spherical_jn(1,x)
        xj1x_prime = (
            spl.spherical_jn(1,x) +
            x*spl.spherical_jn(1,x, derivative=True)
            )

        j1mx = spl.spherical_jn(1,m*x)
        mxj1mx_prime = (
            spl.spherical_jn(1,m*x) +
            m*x*spl.spherical_jn(1,m*x, derivative=True)
            )

        def h1(x, der):
            return (
                spl.spherical_jn(1, x, derivative=der)
                +
                1j*spl.spherical_yn(1, x, derivative=der)
                )
        h1x = h1(x, False)
        xh1x_prime = (
            h1(x, False) +
            x*h1(x, True)
            )

        a_mie =(
            (m**2.*j1mx*xj1x_prime - j1x*mxj1mx_prime)
            /
            (m**2.*j1mx*xh1x_prime - h1x*mxj1mx_prime)
            )

        alpha = 1j*3/(2*k**3.)*a_mie

        return alpha

    alpha_11 = alphaTME_ii(a)
    alpha_22 = alphaTME_ii(a)
    alpha_33 = alphaTME_ii(a)

    alpha_tensor = distribute_sphere_alpha_components_into_tensor(
        alpha_11,
        alpha_22,
        alpha_33,
        isolate_mode,
        )

    return alpha_tensor


def distribute_sphere_alpha_components_into_tensor(
    alpha_11,
    alpha_22,
    alpha_33,
    isolate_mode):

    ## Reorganize matrix dimensions if multiple frequencies given
    if type(alpha_11) is np.ndarray and alpha_11.size > 1:
        alpha_11 = alpha_11[..., None, None]
    if type(alpha_22) is np.ndarray and alpha_22.size > 1:
        alpha_22 = alpha_22[..., None, None]
    if type(alpha_33) is np.ndarray and alpha_33.size > 1:
        alpha_33 = alpha_33[..., None, None]

    if isolate_mode == None:
        alpha_ij = (
            alpha_11 * np.array([
                [1.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_22 * np.array([
                [0.,0.,0.],
                [0.,1.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_33 * np.array([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,1.]
                ])
            )
    elif isolate_mode == 'long':
        alpha_ij = (
            alpha_11 * np.array([
                [1.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]
                ])
            )
    elif (isolate_mode == 'short') or (isolate_mode == 'trans'):
        alpha_ij = (
            alpha_22 * np.array([
                [0.,0.,0.],
                [0.,1.,0.],
                [0.,0.,0.]
                ])
            +
            alpha_33 * np.array([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,1.]
                ])
            )

    return alpha_ij


def sparse_TMatExp_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_sphere_polarizability_TMatExp(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )


def sparse_Mie_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_sphere_polarizability_Mie(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )


def sparse_ret_sphere_polarizability_Drude(w, eps_inf, w_p, gamma,
    eps_b, a, isolate_mode=None):
    ''' '''
    return sparse_ret_sphere_polarizability(
       drude_model(w, eps_inf, w_p, gamma),
       eps_b,
       a,
       w,
       isolate_mode,
       )

###################
## Define scattering crossections for the 3 sphere models
###################
def sigma_scat_ret_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_ret_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

def sigma_scat_Mie_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_Mie_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

def sigma_scat_TMatExp_sphere(w, eps_inf, w_p, gamma,
    eps_b, a,):
    ''''''
    alpha = sparse_TMatExp_sphere_polarizability_Drude(
        w, eps_inf, w_p, gamma, eps_b, a)

    ## result I had as of 02/19/19, don't remember justification
    # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
    #     np.abs(alpha[0,0])**2.
    #     )

    ## simple fix, changing k -> w*n/c
    sigma = sigma_prefactor(w, eps_b) * (
        np.abs(alpha[...,0,0])**2.
        )
    return sigma

# def short_sigma_scat_ret_sphere(w, eps_inf, w_p, gamma,
#     eps_b, a_x, a_yz):
#     ''''''
#     alpha = sparse_ret_sphere_polarizability_Drude(
#         w, eps_inf, w_p, gamma, eps_b, a_x, a_yz)

#     ## result I had as of 02/19/19, don't remember justification
#     # sigma = (8*np.pi/3)*(w/c)**4.*np.sqrt(eps_b)**(-1)*(
#     #     np.abs(alpha[1,1])**2.
#     #     )

#     ## simple fix, changing k -> w*n/c
#     sigma = sigma_prefactor(w, eps_b) * (
#         np.abs(alpha[1,1])**2.
#         )
#     # print('(8*np.pi/3)*(w/c)**4. = ',(8*np.pi/3)*(w/c)**4. )
#     # print('(np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.) = ',
#         # (np.abs(alpha[0,0])**2. + np.abs(alpha[1,1])**2.))
#     return sigma

###################



















############################################################################
### Coupling stuff...
############################################################################

### For generalized polarizabilities

def dipole_mags_gened(
    mol_angle,
    plas_angle,
    d_col,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=None,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    if E_d_angle == None:
        E_d_angle = mol_angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp


    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    alpha_1_p1 = alpha1_diag
    alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )
    # print('geometric_coupling_01 = ',geometric_coupling_01)
    # print('alpha_0 = ',alpha_0)
    # print('alpha_1 = ',alpha_1)
    # print('E_drive = ',E_drive)

    p0 = np.einsum('...ij,...j->...i',geometric_coupling_01 @ alpha_0, E_drive)
    p1 = np.einsum('...ij,...j->...i',alpha_1 @ G_d, p0)

    return [p0, p1]



def plas_dip_driven_by_mol(
    mol_angle,
    plas_angle,
    d_col,
    mol_dipole_mag,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha1_diag=None,
    n_b=None,
    drive_amp=1,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities.

        Returns dipole moment vecotrs as rows in array of shape
        (# of seperations, 3).
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    if E_d_angle == None:
        E_d_angle = mol_angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp

    ## Build 3D molecule dipole moments
    if mol_dipole_mag.ndim is not 1:
        raise TypeError(f"'mol_dipole_mag' is not dimension 1\n"+
            f"mol_dipole_mag.ndim = {mol_dipole_mag.ndim}")

    num_dips_for_calc = len(mol_dipole_mag)

    ## Creat diagonal polarizability for molecule
    alpha0_diag = np.zeros((num_dips_for_calc, 3, 3), dtype=np.complex_)
    alpha0_diag[..., 0, 0] = mol_dipole_mag/drive_amp
    ## Rotate molecule dipoles according to given angle
    alpha_0 = rotation_by(-phi_0) @ alpha0_diag @ rotation_by(phi_0)

    ## Rotate plasmon polarizability by given angle
    alpha_1 = rotation_by(-phi_1) @ alpha1_diag @ rotation_by(phi_1)

    ## Build coupling tensor
    G_d = G(drive_hbar_w, d_col, n_b)

    geometric_coupling_01 = np.linalg.inv(
        np.identity(3) - alpha_0 @ G_d @ alpha_1 @ G_d
        )

    p0 = np.einsum('...ij,...j->...i',alpha_0, E_drive)
    p1 = np.einsum('...ij,...j->...i',alpha_1 @ G_d, p0)

    return [p0, p1]



### older stuff in terms of effective masses and whatnot...
def rotation_by(by_angle):
    ''' need to vectorize '''
    if type(by_angle)==np.ndarray or type(by_angle)==list:
        R = np.zeros((by_angle.size,3,3))
        cosines = np.cos(by_angle)
        sines = np.sin(by_angle)
        R[:,0,0] = cosines
        R[:,0,1] = sines
        R[:,1,0] = -sines
        R[:,1,1] = cosines
        R[:,2,2] = 1
    else:
        R = np.array([[ np.cos(by_angle), np.sin(by_angle), 0],
                      [-np.sin(by_angle), np.cos(by_angle), 0],
                      [ 0,                0,                1]])
    return R


## define coupling diad
def G(drive_hbar_w, d_col, n_b):
    ''' assumes input arrays:
    drive_hbar_w : float
    and vectors,
    p1_hat.shape = (...,3)
    p2_hat.shape = (...,3)
    d_col.shape = (...,3) -> interpretable as ... number of row vectors
    091218: should naivly operate on last dimension as cartesien vector
    091218, 1629: realising this code computes p1 * G * p2, which is
                  a scalar, really I want G.  --->  ^
    '''

    d = vec_mag(d_col) ## returns shape = (...,1), preserves dimension
    n_hat = d_col/d ## returns shape = (...,3)

    w = drive_hbar_w/hbar
    k = w * n_b / c

    dyad = np.einsum('...i,...j->...ij',n_hat,n_hat)
    # print(f'dyad.shape = {dyad.shape}')

    ## If 1 seperation is given, check if multable frequencies given for spectrum
    # print(f'd.size = {d.size}')
    if d.size is not 1:
        d = d[...,None]
    elif d.size is 1 and (type(k) is np.ndarray):
        if k.size > 1:
            k = k.reshape(((k.size,)+(dyad.ndim-1)*(1,)))

    complex_phase_factor = np.exp(1j*k*d)

    ## add all piences together to calculate coupling
    g_dip_dip = (
        # normalization
        # *
        (
            complex_phase_factor*(
                (3.*dyad - np.identity(3)) * (1/d**3.- 1j*k/d**2.)
                -
                (dyad - np.identity(3)) * (k**2./d)
                )
            )
        )

    return g_dip_dip


### ^ requires
def vec_mag(row_vecs):
    '''replace last dimension of array with normalized verion
    '''
    # print(row_vecs)
    vector_magnitudes = np.linalg.norm(row_vecs, axis=(-1))[:,None]  # breaks if mag == 0, ok?
    return vector_magnitudes


def eV_to_Hz(energy):
    return energy/hbar


def uncoupled_p0(
    mol_angle,
    E_d_angle=None,
    alpha_0_p0=None,
    drive_amp=None,
    ):

    phi_0 = mol_angle ## angle of bf_p0

    # phi_1 = plas_angle ## angle of bf_p1
    # p1_hat = rotation_by(phi_1) @ np.array([1,0,0])

    # phi_d = sep_angle ## angle of bf_d

    if E_d_angle == None:
        E_d_angle = mol_angle
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*drive_amp

    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    p0_unc = np.einsum('...ij,...j->...i', alpha_0, E_drive)

    return [p0_unc]

# def uncoupled_p1(plas_angle, E_d_angle=None,
#     drive_hbar_w=parameters['general']['drive_energy']
#     ):

#     # drive_hbar_w = parameters['general']['drive_energy']
#     w = drive_hbar_w/hbar
#     # w_res = w
#     # gamma_nr = drude_damping_energy/hbar
#     # eps_inf =
#     n_b = parameters['general']['background_ref_index']
#     eps_b = n_b**2.

#     # phi_0 = mol_angle ## angle of bf_p0
#     # p0_hat = rotation_by(phi_0) @ np.array([1,0,0])

#     phi_1 = plas_angle ## angle of bf_p1
#     p1_hat = rotation_by(phi_1) @ np.array([1,0,0])

#     # phi_d = sep_angle ## angle of bf_d

#     if E_d_angle == None:
#         E_d_angle = plas_angle
#     E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*ficticious_field_amp

#     alpha_1_p1 = sparse_polarizability_tensor(
#         mass=parameters['plasmon']['fit_mass'],
#         w_res=parameters['plasmon']['fit_hbar_w0']/hbar,
#         w=w,
#         gamma_nr=parameters['plasmon']['fit_hbar_gamma']/hbar,
#         a = parameters['plasmon']['radius'],
#         eps_inf=parameters['plasmon']['eps_inf'],
#         eps_b=eps_b)
#     alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

#     p1_unc = np.einsum('...ij,...j->...i',alpha_1, E_drive)

#     return [p1_unc]












#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Scattering spectrum of coupled dipoles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def sigma_scat_coupled(
    dipoles_moments_per_omega,
    d_col,
    drive_hbar_w,
    n_b=None,
    E_0=None,
    ):
    """ Scattering spectrum of two coupled dipoles p_0 and p_1. Derived from
        Draine's prescription of the DDA.
        """

    omega = drive_hbar_w/hbar
    k = omega * n_b / c

    p_0, p_1 = dipoles_moments_per_omega(omega)

    # print(f'p_0, p_1 = {p_0, p_1}')

    G_d = G(drive_hbar_w, d_col, n_b)

    interference_term = np.sum((
        np.imag(p_0 * np.conj(np.einsum('...ij,...j->...i', G_d, p_1)))
        +
        np.imag(p_1 * np.conj(np.einsum('...ij,...j->...i', G_d, p_0)))
        ), axis=1)
    diag_term_0 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_0, axis=1 ))**2.
    diag_term_1 = (2 / 3) * k**3 * np.abs(np.linalg.norm( p_1, axis=1 ))**2.

    sigma = (
        (4 * np.pi * k  / np.abs(E_0)**2.)
        *
        (
            interference_term
            +
            diag_term_0
            +
            diag_term_1
            )
        )

    return [sigma, np.array(
        [interference_term, diag_term_0, diag_term_1,]
        )*(4 * np.pi * k  / np.abs(E_0)**2.)]


def dipole_moments_per_omega(
    mol_angle,
    plas_angle,
    d,
    E_d_angle=None,
    drive_hbar_w=None,
    alpha0_diag_of_omega=None,
    alpha1_diag_of_omega=None,
    n_b=None,
    drive_amp=None
    ):

    d_col = np.asarray(d).reshape((1, 3))

    alpha0_diag = alpha0_diag_of_omega(drive_hbar_w/hbar)
    alpha1_diag = alpha1_diag_of_omega(drive_hbar_w/hbar)

    p_0, p_1 = dipole_mags_gened(mol_angle,
        plas_angle,
        d_col,
        # E_d_angle=None,
        drive_hbar_w=drive_hbar_w,
        alpha0_diag=alpha0_diag,
        alpha1_diag=alpha1_diag,
        n_b=n_b,
        drive_amp=drive_amp
        )

    return p_0, p_1




if __name__ == "__main__":

     print("This file is not meant to be executed")




