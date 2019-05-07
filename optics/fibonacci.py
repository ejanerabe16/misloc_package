'''
https://arxiv.org/pdf/0912.4540.pdf
'''
from __future__ import print_function
from __future__ import division

import numpy as np

def sphere_to_cart(thetas, phis, rs):
    '''take set of spherical coordinates and give cartisien'''
    # check that the length of thetas, phi, and rs are the same
    carts = []
    for i in range(len(thetas)):
        x = rs[i]*np.sin(thetas[i])*np.cos(phis[i])
        y = rs[i]*np.sin(thetas[i])*np.sin(phis[i])
        z = rs[i]*np.cos(thetas[i])
        carts.append([x,y,z])
    return np.array(carts)

def cart_to_sphere(x, y, z):
    ''' takes rows of cartesien coords and returns rows of (r, theta, phi)
    '''
    r = ( x**2. + y**2. + z**2. )**(0.5)

    theta = np.arctan2(
        ( x**2. + y**2. )**0.5,
        z
        )

    phi = np.arctan2(-y,-x) + np.pi

    sph_points = np.array([r, theta, phi])
    return sph_points

def fib_alg_k_filter(num_points=100, max_ang=np.pi):
    ''' generate evenly distributed points of sphere
    using the fibonacci grid algorithm
    '''

    ## determine total points across sphere if num_points in portion defined
    ## by max_ang
    num_points = round(num_points*2./(1-np.cos(max_ang)))
    if num_points %2 == 0:
        print('given even number of points for Fibonacci lattice, \n',
            ' adding 1...')
        num_points += 1

    n = (num_points-1)/2
    iter_across = np.arange(-n,n+1)
    golden_angle = np.sqrt(2./(3.-np.sqrt(5.)))

    points_on_s = []
    for i in iter_across:
        theta = np.pi/2 + np.arcsin(2.*i/(2.*n+1.))
        if theta > max_ang: break
        phi = 2*np.pi*(i%golden_angle)/golden_angle
        points_on_s.append([theta, phi])
    ans = np.array(points_on_s)
    # print(np.array(points_on_s))
    return ans
