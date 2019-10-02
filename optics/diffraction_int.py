from __future__ import print_function
from __future__ import division

import numpy as np


def observation_points(x_min, x_max, y_min, y_max, points):
    ''' Returns list of (x,y) points for input into integrand,
        and meshgrid for plotting later.
    '''
    X, Y = np.mgrid[x_min:x_max:points*1j, y_min:y_max:points*1j]
    list_of_points = np.vstack([X.ravel(), Y.ravel()]).T
    return np.array([list_of_points, X, Y])

def b_arctan(y,x):
    ''' Defines aximuthal angle from x-axis increasing to 2pi counterclockize
        for cylindrical coordinatess
    '''
    return np.arctan2(-y,-x) + np.pi

def refract_sph_coords(spherical_coords, obj_f, tube_f):
    ''' Converts spherical coordinates used for scattered field into
        coordinates used for refracted fields. In the notation of
        the microspheres paper, this function takes (alpha_1, beta_1)
        and generates the associated (alpha, beta) given the two focal
        lengths f_1 and f.
    '''
    refracted_alpha = np.arcsin((obj_f/tube_f)*np.sin(spherical_coords[:,0]))
    refracted_beta = 2*np.pi - spherical_coords[:,1]  # flip sign of phi
    refracted_coords = np.hstack(
        (refracted_alpha[:,None], refracted_beta[:,None])
        )
    return refracted_coords
        ## should be shape of spherical_coords

def vectorize_coordinates(scattered_coords, obser_pts, obj_f, tube_f):
    ''' Returns alpha, beta, rho, and phi in with arrays properly structered
        for integration with trapazoid rule.
        All arrays structured with shape=(observation_pts, 3, aperture_pts)
    '''
    alpha_sca = scattered_coords[:,0]  # thetas
    alpha_ref = np.arcsin((obj_f/tube_f)*np.sin(alpha_sca))

    beta_sca = scattered_coords[:,1]  # phis
    beta_ref = 2*np.pi - beta_sca
    # print(obser_pts.shape)
    rho = np.sqrt(np.sum(obser_pts**2., axis=1))  # returns row
    phi = b_arctan(obser_pts[:,1],obser_pts[:,0])  # returns row

    # print(rho.shape, phi.shape, alpha_ref.shape, beta_ref.shape)

    alpha_sca = alpha_sca[None,None,:]
    beta_sca = beta_sca[None,None,:]

    alpha_ref = alpha_ref[None,None,:]
    beta_ref = beta_ref[None,None,:]

    rho = rho[:,None,None]
    phi = phi[:,None,None]
    # print(phi)
    return [alpha_sca, beta_sca, alpha_ref, beta_ref, rho, phi]

# #################
def refract_field(scattered_E, scattered_coords, obj_f, tube_f):
    ''' E_ref = sqrt( cos(alpha)/cos(alpha_1) ) * E_sca
    '''
    alpha_sca = scattered_coords[:,0]
    alpha_ref = refract_sph_coords(scattered_coords, obj_f, tube_f)[:,0]
    ## columate angels
    alpha_sca = alpha_sca[:,None]
    alpha_ref = alpha_ref[:,None]
    refracted_E = np.sqrt(np.cos(alpha_ref)/np.cos(alpha_sca)) * scattered_E
    vectorized_E_ref = refracted_E.T[None,...]
    return vectorized_E_ref
        # vectorized by shape=(observation_pts, 3, aperture_pts)
# #################

def generate_integrand(
    scattered_E,
    scattered_sph_coords,
    obser_pts,
    z,
    obj_f,
    tube_f,
    k
    ):

    refracted_coords = refract_sph_coords(
        scattered_sph_coords,
        obj_f,
        tube_f
        )

    refracted_E = refract_field(
        scattered_E,
        scattered_sph_coords,
        obj_f,
        tube_f)

    (alpha_sca,
        beta_sca,
        alpha_ref,
        beta_ref,
        rho,
        phi) = vectorize_coordinates(
            scattered_sph_coords,
            obser_pts,
            obj_f,
            tube_f
            )

    ## Scalar prefactor
    ## ---------------_
    ## s = -i k e^{i k f_tube} / (2 π f_obj)
    ##
    ## Has units of k / f ~ (length)^-2
    s_prefactor = -1j*k*np.exp(1j*k*tube_f)/(2*np.pi*obj_f)

    ## Vector prefactor
    ## ----------------
    ## no units
    v_prefactor_denom = (
        (tube_f/obj_f)**2.-np.sin(alpha_sca)**2.
        )**0.5

    v_prefactor_denom_no_sing = v_prefactor_denom.copy()
    v_prefactor_denom_no_sing[np.where(v_prefactor_denom==0)] = 1

    v_prefactor = np.cos(alpha_sca)/v_prefactor_denom_no_sing
    v_prefactor[np.where(v_prefactor_denom==0)] = 1

    ## Complex exponential
    ## -------------------
    ## no units
    c_exp = np.exp(
        1j*k*(
            rho*np.sin(alpha_ref)*np.cos(beta_ref-phi)
            +
            z*np.cos(alpha_ref)
            )
        )

    ## units of 1/l^2
    integrand = s_prefactor*v_prefactor*refracted_E*c_exp

    return integrand

def perform_integral(
        scattered_E,
        scattered_sph_coords,
        obser_pts,
        z,
        obj_f,
        tube_f,
        k,
        alpha_1_max,
        ):

    integrand = generate_integrand(
        scattered_E,
        scattered_sph_coords,
        obser_pts,
        z,
        obj_f,
        tube_f,
        k
        )

    ## Define differential area element
    ## --------------------------------
    number_of_lens_points = scattered_sph_coords.shape[0]
    ## Area element has units of f^2 ~ length
    lens_surface_area = obj_f**2. * 2*np.pi * (1-np.cos(alpha_1_max))
    area_per_point = lens_surface_area/number_of_lens_points

    weighted_terms_for_sum = area_per_point*integrand
    result_of_integral = np.sum(weighted_terms_for_sum, axis=2)

    return result_of_integral
