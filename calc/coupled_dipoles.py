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


curly_yaml_file_name = '/curly_nrod_water_JC.yaml'

# print('reading parameters from {}'.format(
#     parameter_files_path+curly_yaml_file_name
#     )
# )

opened_param_file = open(
    parameter_files_path+curly_yaml_file_name,'r'
    )
parameters = yaml.load(opened_param_file)
# print(curly_yaml_file_name)
## System background
n_b = parameters['general']['background_ref_index']
eps_b = n_b**2.

## Plasmon Drude properties
# hbar_w_p = parameters['plasmon']['plasma_energy']
# drude_damping_energy = parameters['plasmon']['drude_damping_energy']
# eps_inf = parameters['plasmon']['eps_inf']

## Other plasmon properties
# a = parameters['plasmon']['radius']

## Driving force
ficticious_field_amp = parameters['general']['drive_amp']

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
def fluorophore_mass(ext_coef, gamma):
    '''Derived at ressonance'''
    m = 4 * np.pi * e**2 * n_a  / (
            ext_coef * np.log(10) * c * n_b * gamma
            )
    return m

## Define polarizabilities in diagonal frames
def sparse_polarizability_tensor(mass, w_res, w, gamma_nr, a, eps_inf, ebs_b):
    '''Define diagonal polarizability with single cartesien component derived
        from Drude model > Clausius-mosati.
        Assumes physical constants definded; e, c
    '''
    gamma_r = gamma_nr + (2*e**2./(3*mass*c**3.))*w**2.
    alpha_0_xx_osc = (e**2. / mass)/(w_res**2. - w**2. - 1j*gamma_r*w)
    alpha_0_xx_static = (a**3. * (eps_inf - 1*eps_b)/(eps_inf + 2*eps_b))
    # print('alpha_0_xx_static = ',alpha_0_xx_static)
    alpha_0_xx = alpha_0_xx_osc + alpha_0_xx_static

    alpha_0_ij = np.array([[alpha_0_xx, 0, 0],
                           [0,          0, 0],
                           [0,          0, 0]])

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

def sparse_ret_prolate_spheroid_polarizability(eps, eps_b, a_x, a_yz, w, isolate_mode=None):
    '''Follows Moroz, A. Depolarization field of spheroidal particles,
        J. Opt. Soc. Am. B 26, 517
        but differs in assuming that the long axis is x oriented
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
            L = (1-e**2.)/e**3. * (-e + np.arctanh(e))
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
        return np.sqrt((a_x**2. - a_yz**2.)/a_x**2.)

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

        return alphaMW

    ### Define dynamic geometric factors 'D_i' for alphaMW
    def D_x(a_x, a_yz):
        e = ecc(a_x, a_yz)
        D = 3/4 * (
            ((1+e**2.)/(1-e**2.))*L_i(1, a_x, a_yz) + 1
            )
        return D

    def D_yz(a_x, a_yz):
        e = ecc(a_x, a_yz)
        # print('e in D = ',e)
        D = (a_yz/(2*a_x))*(3/e * np.arctanh(e) - D_x(a_x,a_yz))
        # print('D = ',D)
        return D

    alpha_11 = alphaMW_ii(1, a_x, a_yz)
    alpha_22 = alphaMW_ii(2, a_x, a_yz)
    alpha_33 = alphaMW_ii(3, a_x, a_yz)

    if isolate_mode == None:
        alpha_ij = np.array([[ alpha_11,       0.,       0.],
                             [       0., alpha_22,       0.],
                             [       0.,       0., alpha_33]])
    elif isolate_mode == 'long':
        alpha_ij = np.array([[ alpha_11,       0.,       0.],
                             [       0.,       0.,       0.],
                             [       0.,       0.,       0.]])
    elif (isolate_mode == 'short') or (isolate_mode == 'trans'):
        alpha_ij = np.array([[       0.,       0.,       0.],
                             [       0., alpha_22,       0.],
                             [       0.,       0., alpha_33]])

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
        /(0.5*n_b) # copied from MNPBEM source
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





















############################################################################
### Coupling stuff...
############################################################################

### For generalized polarizabilities

def dipole_mags_gened(
    mol_angle,
    plas_angle,
    d_col,
    E_d_angle=None,
    drive_hbar_w=parameters['general']['drive_energy'],
    alpha0_diag=None,
    alpha1_diag=None,
    ):
    """ Calculate dipole magnitudes with generalized dyadic
        polarizabilities
        """

    # Initialize unit vector for molecule dipole in lab frame
    phi_0 = mol_angle ## angle of bf_p0 in lab frame

    # Initialize unit vecotr for molecule dipole in lab frame
    phi_1 = plas_angle ## angle of bf_p1 in lab frame

    if E_d_angle == None:
        E_d_angle = mol_angle
    # rotate driving field into lab frame
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*ficticious_field_amp


    alpha_0_p0 = alpha0_diag
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    alpha_1_p1 = alpha1_diag
    alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

    G_d = G(drive_hbar_w, d_col)

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
def G(drive_hbar_w, d_col):
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

    d = d[...,None]
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
    drive_hbar_w=parameters['general']['drive_energy'],
    n_b=parameters['general']['background_ref_index']
    ):
    # drive_hbar_w = parameters['general']['drive_energy']
    w = drive_hbar_w/hbar
    # w_res = w
    # gamma_nr = drude_damping_energy/hbar
    # eps_inf =

    eps_b = n_b**2.

    phi_0 = mol_angle ## angle of bf_p0
    p0_hat = rotation_by(phi_0) @ np.array([1,0,0])

    # phi_1 = plas_angle ## angle of bf_p1
    # p1_hat = rotation_by(phi_1) @ np.array([1,0,0])

    # phi_d = sep_angle ## angle of bf_d

    if E_d_angle == None:
        E_d_angle = mol_angle
    E_drive = rotation_by(E_d_angle) @ np.array([1,0,0])*ficticious_field_amp

    mole_eff_mass = fluorophore_mass(
        ext_coef=parameters['fluorophore']['extinction_coeff'],
        gamma=parameters['fluorophore']['mass_gamma']/hbar
        )
    alpha_0_p0 = sparse_polarizability_tensor(
        mass=mole_eff_mass,
        w_res=parameters['fluorophore']['res_energy']/hbar,
        w=w,
        gamma_nr=parameters['fluorophore']['test_gamma']/hbar,
        a=0,
        eps_inf=1,
        ebs_b=1)
    alpha_0 = rotation_by(-phi_0) @ alpha_0_p0 @ rotation_by(phi_0)

    p0_unc = np.einsum('...ij,...j->...i',alpha_0, E_drive)

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
#         ebs_b=eps_b)
#     alpha_1 = rotation_by(-phi_1) @ alpha_1_p1 @ rotation_by(phi_1)

#     p1_unc = np.einsum('...ij,...j->...i',alpha_1, E_drive)

#     return [p1_unc]


if __name__ == "__main__":

     print("This file is not meant to be executed")




