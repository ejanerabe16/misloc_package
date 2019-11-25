from __future__ import print_function
from __future__ import division

import sys
import os
import yaml

import numpy as np
import scipy.optimize as opt
from scipy import interpolate
import scipy.io as sio
import scipy.special as spf


from ..optics import diffraction_int as diffi
from ..optics import fibonacci as fib

## Read parameter file to obtain physical properties
## of molecule and plasmon, molecule and imaging system.
from misloc_mispol_package import project_path

parameter_files_path = (
    project_path + '/param'
)

# curly_yaml_file_name = '/curly_nrod_water_JC.yaml'
# parameters = yaml.load(open(parameter_files_path+curly_yaml_file_name,'r'))


# modules_path = project_path + '/solving_problems/modules'
# sys.path.append(modules_path)

from . import fitting_misLocalization as fit

## plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl
# get_ipython().magic('matplotlib inline')

# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# mpl.rcParams['text.usetex'] = True

## colorbar stuff
from mpl_toolkits import axes_grid1

## Import analytic expressions for the focused fields from a point dipole.
from ..optics import anal_foc_diff_fields as afi

## Import computational solution to two coupled oscillators of arbitrarty
## anisotropic polarizabilities.
# import coupled_dipoles as cp

## Import physical constants from yaml file.
phys_const_file_name = '/physical_constants.yaml'
opened_constant_file = open(
    parameter_files_path+phys_const_file_name,
    'r')
constants = yaml.load(opened_constant_file)
e = constants['physical_constants']['e']
c = constants['physical_constants']['c']  # charge of electron in statcoloumbs
hbar = constants['physical_constants']['hbar']
nm = constants['physical_constants']['nm']
n_a = constants['physical_constants']['nA']   # Avogadro's number
m_per_nm = constants['physical_constants']['nm']
# Z_o = 376.7303 # impedence of free space in ohms (SI)

## Define some useful constants from defined parameters
# n_b = parameters['general']['background_ref_index']
# eps_b = n_b**2.
# a = parameters['plasmon']['radius']



#######################################################################
## Optics stuff.
# sensor_size = parameters['optics']['sensor_size']*nm
# # height = 2*mm  # also defines objective lens focal length
# # height = parameters['optics']['self.obj_f_len']
# resolution = parameters['optics']['sensor_pts']  # image grid resolution
# ## Build image sensor
# eye = diffi.observation_points(
#     x_min= -sensor_size/2,
#     x_max= sensor_size/2,
#     y_min= -sensor_size/2,
#     y_max= sensor_size/2,
#     points= resolution
#     )

# ## Experimental parameters
# magnification = parameters['optics']['magnification']
# numerical_aperture = parameters['optics']['numerical_aperture']
# max_theta = np.arcsin(numerical_aperture) # defines physical aperture size

# ## numerical parameters for calculation of scattered field
# lens_points = parameters['optics']['lens_points']

# # obj_f = 1.*mm  # still dont know what this is supposed to be
# obj_f = parameters['optics']['obj_f_len']

# self.tube_f = magnification * obj_f

## calculate dipole magnitudes
# drive_energy_eV = parameters['general']['drive_energy'] ## rod long mode max at 1.8578957289256757 eV
# omega_drive = drive_energy_eV/hbar  # driving frequency


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup warning tracker, not sure how useful this is in module...
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import traceback
import warnings
import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup warning tracker, not sure how useful this is in module...
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## import matlab engine to run BEM
import matlab
import matlab.engine

# Import from fit.
# def fixed_ori_mol_placement(x_min=0, x_max=500, y_min=0, y_max=500, mol_grid_pts_1D = 3, mol_angle=0):
#     locations = diffi.observation_points(x_min, x_max, y_min, y_max, points=mol_grid_pts_1D)[0]
#     locations = np.hstack((locations,np.zeros((locations.shape[0],1))))

#     mol_linspace_pts = mol_grid_pts_1D
# #     random_mol_angles= (np.random.random(mol_linspace_pts**2)*np.pi*2)
#     return [locations, mol_angle]

class Simulation(fit.DipoleProperties):
    """Runs BEM simulation

    Collects focused+diffracted far-field information from molecules nearby a nanorod
    by MNPBEM calculation in an invisible instance of Matlab.

    Args:
        locations: list of 3D cartesien coordinates of molecules.
            One simulation is run per location.
        mol_angle: Angle on the molecule's dipole moment relatice to the x-axis
        plas_angle: Angle of the long-axis of the nanorod relative to the x-axis

    Attributes:
        mol_too_close: named confusingly, this method is built to filter locations
            for points within the fluorescence quenching region around the nanorod,
            assigned to be 10 nm from the surface of the fit ellipsoid.
            Returns: VALID locations OUTSIDE this region.
        calculate_BEM_fields: runs simulatins and stores results in attributes;
            bem_E: Focused and diffracted electric field
            BEM_images:
            Check method docstring for availible BEM simulation options.

    """

    ## set up inverse mapping from observed -> true angle for signle molecule in the plane.

    def __init__(self,
        locations,
        mol_angle=0,
        plas_angle=np.pi/2,
        obs_points=None,
        simulation_type='disk',
        param_file=None,
        ):

        ## Get all of the system specific attributes
        fit.DipoleProperties.__init__(self, param_file=param_file)

        self.simulation_type = simulation_type

        # self.n_b = n_b ## vacuum
        self.mol_locations = locations
        self.mol_angles = mol_angle
        self.rod_angle = plas_angle

        #### Filtering out molecules in region of fluorescence quenching
        self.el_a = self.a_long_meters/nm
        self.el_c = self.a_short_meters/nm
        self.quel_a = self.el_a + self.fluo_quench_range ## define quenching region
        self.quel_c = self.el_c + self.fluo_quench_range

        self.input_x_mol = locations[:,0]
        self.input_y_mol = locations[:,1]

        self.pt_is_in_ellip = self.mol_too_close()
        ## select molecules outside region,
        self.mol_locations = locations[self.pt_is_in_ellip]
        ## select molecule angles if listed per molecule,
        if type(mol_angle)==np.ndarray and mol_angle.shape[0]>1:
            self.mol_angles = mol_angle[self.pt_is_in_ellip]
        else: self.mol_angles = mol_angle

        self.default_plot_limits = [
            np.min(self.mol_locations)
            -
            (
                (np.max(self.mol_locations)-np.min(self.mol_locations))*.1
                ),
            np.max(self.mol_locations)
            +
            (
                (np.max(self.mol_locations)-np.min(self.mol_locations))*.1
                )
            ]


        if obs_points is None:
            ## Load from param file
            self.parameters = fit.load_param_file(param_file)

            sensor_size = self.parameters['optics']['sensor_size']*nm
            # height = 2*mm  # also defines objective lens focal length
            # height = self.parameters['optics']['obj_f_len']
            resolution = self.parameters['optics']['sensor_pts']  # image grid resolution
            ## Build image sensor
            self.obs_points = diffi.observation_points(
                x_min= -sensor_size/2,
                x_max= sensor_size/2,
                y_min= -sensor_size/2,
                y_max= sensor_size/2,
                points= resolution
                )

        else:
            self.obs_points = obs_points

        ## Load microscope parameters
        ## Experimental parameters
        self.magnification = self.parameters['optics']['magnification']
        self.numerical_aperture = self.parameters['optics']['numerical_aperture']
        self.max_theta = np.arcsin(self.numerical_aperture) # defines physical aperture size

        ## numerical parameters for calculation of scattered field
        self.lens_points = self.parameters['optics']['lens_points']

        # obj_f = 1.*mm  # still dont know what this is supposed to be
        self.obj_f = self.parameters['optics']['obj_f_len']
        self.tube_f = self.magnification * self.obj_f

        # ## Load Drive energy
        # self.drive_energy_eV = parameters['general']['drive_energy']
        #     ## rod long mode max at 1.8578957289256757 eV

    def mol_too_close(self):
        ''' Returns molecule locations that are outside the
            fluorescence quenching zone, defined as 10 nm from surface
            of fit spheroid.
            '''
        rotated_x = (
            np.cos(self.rod_angle)*self.input_x_mol
            +
            np.sin(self.rod_angle)*self.input_y_mol
            )
        rotated_y = (
            -np.sin(self.rod_angle)*self.input_x_mol
            +
            np.cos(self.rod_angle)*self.input_y_mol
            )
        long_quench_radius = self.quel_a
        short_quench_radius = self.quel_c
        rotated_ellip_eq = (
            rotated_x**2./long_quench_radius**2
            +
            rotated_y**2./short_quench_radius**2
            )
        return (rotated_ellip_eq > 1)


    def calculate_BEM_fields(self):
        """ Runs BEM simulation in Matlab using parameters initialized
            parameters.

            Current options for BEM simulations are
            selected by the class instance variable 'simulation_type';
                - 'disk' : disk with layer substrate, BROKEN.
                - 'bare_disk_JC' : Gold disk in water, JC data.
                - 'bare_disk_Drude' : Gold disk in water, Drude model.
                    built in to BEM.
            """
        if hasattr(self, 'BEM_images'):
            return self.BEM_images

        ## start background matlab instance before looping through images.
        print('starting Matlab...')
        eng = matlab.engine.start_matlab()

        # Add BEM to path
        eng.addpath(
            eng.genpath(project_path+'/matlab_bem', nargout=1),
            nargout=0)

        if self.simulation_type == 'disk':
            mnpbem_sim_fun = eng.BEM_CurlyDisk_dipDrive_E
        elif self.simulation_type == 'bare_disk_JC':
            ## Just disk in water, included in package files.
            ## File reads JC data built in to BEM currently.
            mnpbem_sim_fun = eng.CurlyDiskJC_NoSub_dipDrive_E
        elif self.simulation_type == 'bare_disk_Drude':
            ## Just disk in water, included in package files.
            ## File reads JC data built in to BEM currently.
            mnpbem_sim_fun = eng.CurlyDiskDrudeNoSub_dipDrive_E
        elif self.simulation_type == 'rod':
            mnpbem_sim_fun = eng.AuNR_dipDrive_E

        # Initialize coordinates of points on hemisphere for field BEM field
        # calculation.
        sphere_points = fib.fib_alg_k_filter(
            num_points=self.lens_points,
            max_ang=self.max_theta
            )
        # Convert spherical coordinates to Caresian.
        cart_points_on_sph = fib.sphere_to_cart(
            sphere_points[:,0],
            sphere_points[:,1],
            self.obj_f*np.ones(np.shape(sphere_points[:,0]))
            )

        ## convert lens-integration coordinates to matlab variable
        matlab_cart_points_on_sph = matlab.double(cart_points_on_sph.tolist())

        # Setup values for field calculation
        drive_energy = self.drive_energy_eV
        number_of_molecules = self.mol_locations.shape[0]

        # Initialize outputs.
        self.BEM_images = np.zeros(
            (number_of_molecules, self.obs_points[0].shape[0])
            )
        self.bem_E = np.zeros(
            (number_of_molecules, self.obs_points[0].shape[0], 3),
            dtype=np.complex_
            )

        # Loop through molecules and run BEM calculation in MATLAB
        for i in range(number_of_molecules):
            print('{}th molecule'.format(int(i+1)))
            mol_location = self.mol_locations[i]
            if np.atleast_1d(self.mol_angles).shape[0] == (
                self.mol_locations.shape[0]
                ):
                mol_angle = np.atleast_1d(self.mol_angles)[i]
            elif np.atleast_1d(self.mol_angles).shape[0] == 1:
                mol_angle = self.mol_angles
            mol_orientation = [
                np.cos(mol_angle),
                np.sin(mol_angle),
                0.,
            ]


            # print(f"\nmol_location input to BEM function:\n",
            #     mol_location
            #     )
            # print(f"\nmol_orientation input to BEM function:\n",
            #     mol_orientation
            #     )
            # print(f"\nmatlab_cart_points_on_sph input to BEM function:\n",
            #     matlab_cart_points_on_sph
            #     )

            self.matlab_cart_points_on_sph = matlab_cart_points_on_sph
            # Run BEM calculation, return fields and coords.
            [E, sph_p] = mnpbem_sim_fun(
                matlab.double(list(mol_location)),
                drive_energy,
                matlab.double(list(mol_orientation)),
                matlab_cart_points_on_sph,
                float(self.eps_b),
                nargout=2)

            # Format outputs for np
            self.BEM_scattered_E = np.asarray(E)
                # print('self.BEM_scattered_E.shape = ',self.BEM_scattered_E.shape)
                # BEM_scattered_H = np.asarray(H)
                # print('BEM_scattered_H.shape = ',BEM_scattered_H.shape)
            cart_sphere_points = np.asarray(sph_p)

            sph_sph_points = fib.cart_to_sphere(
                cart_sphere_points[:,0],
                cart_sphere_points[:,1],
                cart_sphere_points[:,2]
                ).T

            thetas_and_phis = sph_sph_points[:,1:]

            # Calculate focused+diffracted fields
            print('calculating diffracted fields')
            diffracted_E_field = diffi.perform_integral(
                scattered_E=self.BEM_scattered_E,
                scattered_sph_coords=thetas_and_phis,
                obser_pts=self.obs_points[0]*np.array([[1,-1]]),
                z=0,
                obj_f=self.obj_f,
                tube_f=self.tube_f,
                k=(self.drive_energy_eV/hbar)*(np.sqrt(self.eps_b))/c,
                alpha_1_max=self.max_theta
                )

            diffracted_power_flux = np.real(
                np.sum(
                    np.multiply(diffracted_E_field,
                        np.conj(diffracted_E_field)
                        ),
                    axis=-1
                    )
                )

            self.bem_E[i] = diffracted_E_field
            self.BEM_images[i] = diffracted_power_flux

        eng.exit()

        return self.BEM_images


class SimulatedExperiment(Simulation, fit.MolCoupNanoRodExp):
    """ Give BEM simulation class instance same attributes as Model Exp
        class for easy plotting.
        """
    def __init__(self,
        locations,
        mol_angle=0,
        plas_angle=np.pi/2,
        obs_points=None,
        simulation_type='disk',
        **kwargs
        ):

        fit.CoupledDipoles.__init__(self,
            # obs_points=obs_points,
            **kwargs
            )

        Simulation.__init__(self,
            locations,
            mol_angle,
            plas_angle,
            obs_points,
            simulation_type=simulation_type,
            **kwargs
            )
        ## Get drive intensity
        fit.BeamSplitter.__init__(self, **kwargs)

    def plot_mispol_map_wMisloc(self,
        plot_limits=None,
        given_ax=None,
        **kwargs,
        ):

        return fit.MolCoupNanoRodExp.plot_mispol_map_wMisloc(
            self,
            plot_limits=plot_limits,
            given_ax=given_ax,
            plot_ellipse=False,
            **kwargs
            )

    def plot_image(self, image_idx, ax=None, images=None):

        if images is None:
            image = self.BEM_images[image_idx]

        if ax is None:
            plt.figure(figsize=(3,3),dpi=600)
            plt.pcolor(
                self.obs_points[-2]/m_per_nm,
                self.obs_points[-1]/m_per_nm,
                image.reshape(self.obs_points[-2].shape),
                )
            plt.colorbar()
        else:
            ax.contour(self.obs_points[-2]/m_per_nm,
                self.obs_points[-1]/m_per_nm,
                image.reshape(self.obs_points[-2].shape),
                cmap='Greys',
                linewidths=0.5,
                )
        plt.title(r'$|E|^2/|E_\mathrm{inc}|^2$')
        plt.xlabel(r'$x$ [nm]')
        plt.ylabel(r'$y$ [nm]')

        return plt.gca()

    def make_image_noisy(self, PEAK=1):
        if not hasattr(self, 'BEM_images'):
            image = self.calculate_BEM_fields()
        else:
            image = self.BEM_images

        if image.ndim == 2:
            ## Assuming a set of raveled images,
            ## Normalize each independently
            max_for_norm = image.max(axis=1)[:,None]
        elif image.ndim == 1:
            max_for_norm = image.max()

        normed_image = image/max_for_norm
        noised_normded_image_data = (
            np.random.poisson(normed_image/255.0* PEAK) / (PEAK) *255
            )
        noised_image_data = noised_normded_image_data*max_for_norm

        return noised_image_data



class NoisySimulatedExp(SimulatedExperiment):
    """ Replicate the SimulatedExperiment class, but add noise
        generation for experiment emulation
        """

    def __init__(self,
        locations,
        mol_angle=0,
        plas_angle=np.pi/2,
        obs_points=None,
        simulation_type='disk',
        ):


        SimulatedExperiment.__init__(
            self,
            locations=locations,
            mol_angle=mol_angle,
            plas_angle=plas_angle,
            obs_points=obs_points,
            simulation_type=simulation_type
            )


def save_sim_exp_inst(sim_exp_instance, data_dir_name=None):

    path_to_data = (
        project_path
        +
        "/data/"
        +
        data_dir_name
        )

    os.mkdir( path_to_data )
#     print(path_to_data+'/BEM_images.txt')
    np.savetxt(path_to_data+
        '/BEM_images.txt',sim_exp_instance.BEM_images)
    np.savetxt(path_to_data+
        '/mol_locations.txt',sim_exp_instance.mol_locations)
    np.savetxt(path_to_data+
        '/default_plot_limits.txt',sim_exp_instance.default_plot_limits)
    np.savetxt(path_to_data+
        '/mol_angles.txt',np.atleast_1d(sim_exp_instance.mol_angles))
    np.savetxt(path_to_data+
        '/rod_angle.txt',np.atleast_1d(sim_exp_instance.rod_angle))
    np.savetxt(path_to_data+
        '/obs_points[0].txt',np.atleast_1d(sim_exp_instance.obs_points[0]))
    np.savetxt(path_to_data+
        '/obs_points[1].txt',np.atleast_1d(sim_exp_instance.obs_points[1]))
    np.savetxt(path_to_data+
        '/obs_points[2].txt',np.atleast_1d(sim_exp_instance.obs_points[2]))

    np.savetxt(path_to_data+
        '/mispol_angle.txt',np.atleast_1d(sim_exp_instance.mispol_angle))


class LoadedSimExp(SimulatedExperiment):

    def __init__(self,
        data_dir_name,
        param_file,
        ):

        fit.DipoleProperties.__init__(self,
#             isolate_mode=isolate_mode,
#             drive_energy_eV=drive_energy_eV,
            param_file
            )

        self.path_to_data = (
            project_path
            +
            "/data/"
            +
            data_dir_name
            )


        self.BEM_images = np.loadtxt(self.path_to_data+'/BEM_images.txt')
        # self.trial_images = np.loadtxt(self.path_to_data+'/BEM_images.txt')
        self.mol_locations = np.loadtxt(self.path_to_data+'/mol_locations.txt')
        self.default_plot_limits = np.loadtxt(
            self.path_to_data+'/default_plot_limits.txt')
        self.mol_angles = np.loadtxt(self.path_to_data+'/mol_angles.txt')
#         self.angles = self.mol_angles
        self.rod_angle = np.atleast_1d(
            np.loadtxt(self.path_to_data+'/rod_angle.txt'))[0]

        obs_points_0 = np.loadtxt(self.path_to_data+'/obs_points[0].txt')
        obs_points_1 = np.loadtxt(self.path_to_data+'/obs_points[1].txt')
        obs_points_2 = np.loadtxt(self.path_to_data+'/obs_points[2].txt')

        self.obs_points = np.array([obs_points_0, obs_points_1, obs_points_2])

        self.mispol_angle = np.loadtxt(self.path_to_data+'/mispol_angle.txt')


def save_fit_inst(fit_instance, exp_instance, data_dir_name):
    path_to_data = (
        project_path
        +
        "/data/"
        +
        data_dir_name
        )
    os.mkdir( path_to_data )

    np.savetxt(
        path_to_data+'/model_fit_results.txt',fit_instance.model_fit_results)

    # Directory for exp instance
#     os.mkdir( path_to_data+'/exp_instance' )
    save_sim_exp_inst(exp_instance, data_dir_name+'/exp_instance')


class LoadedFit(fit.FitModelToData, fit.PlottingStuff):

    def __init__(self,
        data_dir_name,
        param_file,
        ):

        fit.DipoleProperties.__init__(self,
#             isolate_mode=isolate_mode,
#             drive_energy_eV=drive_energy_eV,
            param_file=param_file
            )
        ## get plot NP radii and quenching zone
        fit.PlottingStuff.__init__(self,
#             isolate_mode=isolate_mode,
#             drive_energy_eV=drive_energy_eV,
            param_file=param_file
            )


        self.path_to_data = (
            project_path
            +
            "/data/"
            +
            data_dir_name
            )


        self.model_fit_results = np.loadtxt(
            self.path_to_data+'/model_fit_results.txt'
            )

        self.loaded_sim_exp_instance = LoadedSimExp(
            data_dir_name+'/exp_instance',
            param_file=param_file
            )

    def plot_fit_results_as_quiver_map(
        self,
        fitted_exp_instance=None,
        plot_limits=None,
        given_ax=None,
        draw_quadrant=True,
        ):

        if fitted_exp_instance is None:
            fitted_exp_instance = self.loaded_sim_exp_instance

        if type(plot_limits) == np.ndarray:
            plot_limits.to_list()

        return fit.FitModelToData.plot_fit_results_as_quiver_map(
            self,
            fitted_exp_instance=fitted_exp_instance,
            plot_limits=plot_limits,
            given_ax=given_ax,
            draw_quadrant=draw_quadrant,
            )

def fig5(
    sim_instance, fit_model_instance,
    quiv_ax_limits=[-25,150,-25,150],
    quiv_tick_list=np.linspace(0,125,6),
    quiv_ticklabel_list=[r'$0$', r'$25$',r'$50$',r'$75$',r'$100$',r'$125$'],
    fig_size=(6.5, 2.9),
    draw_quadrant=True,
    ):

    cbar_width = 0.15
    plot_width = 4
    widths = [cbar_width, plot_width, plot_width, cbar_width]
    heights = [1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    # paper_fig, paper_axs = plt.subplots(
    #     nrows=1,
    #     ncols=4,
    # #     sharey=True,
    #     figsize=fig_size,
    #     dpi=300,
    #     constrained_layout=True,
    #     gridspec_kw=gs_kw,
    #     # sharey='row',
    #     )

    paper_fig = plt.figure(
        figsize=fig_size,
        dpi=300,
        constrained_layout=True,
        )
    spec = mpl.gridspec.GridSpec(
        nrows=1,
        ncols=4,
        figure=paper_fig,
        **gs_kw,
        )

    # Add 4 axes to figure
    paper_axs = []
    paper_axs.append(paper_fig.add_subplot(spec[0, 0]))
    paper_axs.append(paper_fig.add_subplot(spec[0, 1]))
    paper_axs.append(
        paper_fig.add_subplot(
            spec[0, 2],
            sharey=paper_axs[1]
            )
        )
    paper_axs.append(
        paper_fig.add_subplot(spec[0, 3], sharey=paper_axs[0])
        )


    paper_axs[1] = sim_instance.plot_mispol_map_wMisloc(
        given_ax=paper_axs[1],
        draw_quadrant=draw_quadrant,
        )

    # Place left ticks and labels on the right
    paper_axs[1].yaxis.tick_right()
    paper_axs[1].yaxis.set_label_position("right")
    for tk in paper_axs[1].get_yticklabels():
        tk.set_visible(True)
    # paper_axs[1].set_title(None)

    paper_axs[2] = fit_model_instance.plot_fit_results_as_quiver_map(
        sim_instance, given_ax=paper_axs[2],
        draw_quadrant=draw_quadrant)
    # paper_axs[2].set_title(None)

    fit_model_instance.build_colorbar(
        paper_axs[0],
        r'PBD polarization angle [$\Delta^\circ$]',
        fit.PlottingStuff.curlycm
        )

    fit_model_instance.build_colorbar(
        paper_axs[3],
        'Molecule angle by model fit [deg]',
        fit.PlottingStuff.curlycm
        )

    paper_axs[0].yaxis.tick_left()
    paper_axs[0].yaxis.set_label_position("left")

    # Build legends
    def loc_map_legend(ax, loc_label='fit localization'):
        legend_elements = [
            mpl.lines.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=loc_label,
                markerfacecolor=fit.PlottingStuff.a_shade_of_green,
                markersize=10
                ),
            mpl.lines.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label='molecule location',
                markerfacecolor='black',
                markersize=8
                ),
            ]

        ax.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(1,1.11),
    #         ncol=2, mode="expand",
            fontsize=8,
            framealpha=1,
    #         loc=1
            )

    # Plot Legend
    # loc_map_legend(paper_axs[2], loc_label='fit localization')

    # Title
    paper_axs[1].set_title(None)
    paper_axs[1].set_title('(a)', loc='left')
    paper_axs[2].set_title(None)
    paper_axs[2].set_title('(b)', loc='left')



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Quick and dirty fixes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # rotate right colorbar label
    paper_axs[3].set_ylabel(None)
    # paper_axs[3].yaxis.set_label_position("right")
    paper_axs[3].set_ylabel(
        r'Model fit molecule polarization [$\Delta^\circ$]',
        rotation=270,va="bottom")

    for cbar_ax in [paper_axs[3],paper_axs[0]]:
    #     cbar_ax.yaxis.set_ticks(np.linspace(0,np.pi/2,10))
    #     cbar_ax.yaxis.set_ticklabels(
    #         [r'$0$', r'$10$',r'$20$',r'$30$',r'$40$',r'$50$',r'$60$',r'$70$',r'$80$',r'$90$',]
    #         )
        cbar_ax.yaxis.set_ticks(np.linspace(0,np.pi/2,3))
        cbar_ax.yaxis.set_ticklabels(
            [r'$0$', r'$45$',r'$90$',]
            )

    # Redo quiver ticks
    for quiver_ax in [paper_axs[i+1] for i in range(2)]:
        quiver_ax.tick_params(direction='in'),


        # set axis equal
        # quiver_ax.axis('equal')
        # quiver_ax.set_ylim(-10,165)
        quiver_ax.axis(quiv_ax_limits)
        # quiver_ax.set_ylim(quiv_ax_limits[:2])

        quiver_ticks = np.linspace(0,125,5)
        for quiver_axis in [quiver_ax.yaxis, quiver_ax.xaxis]:
            quiver_axis.set_ticks(quiv_tick_list)
            quiver_axis.set_ticklabels(
                quiv_ticklabel_list
                )

        # fix labels
        quiver_ax.set_xlabel(r'$x$ position [nm]')
        quiver_ax.set_ylabel(r'$y$ position [nm]')

    paper_axs[2].set_ylabel(None)
    # paper_axs[2].set_yticklabels([None])
    paper_axs[2].tick_params(labelleft=False)

    # Things I tried to get yticklabels to show up with sharey
    # paper_axs[1].yaxis.set_tick_params(
    #     which='both',
    #     labelright=True,
    #     )
    # plt.setp(paper_axs[1].get_yticklabels(), visible=True)
    # for tk in paper_axs[1].get_yticklabels():
    #     print(tk)
    #     tk.set_visible(True)

    return paper_fig

def map_angles_to_first_quad(angles):
    angle_in_first_quad = np.arctan(
        np.abs(np.sin(angles))
        /
        np.abs(np.cos(angles))
        )
    return angle_in_first_quad


if __name__ == "__main__":

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Example how to run simulations, with Gaussian localizations and PBD
    # polarizations
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    locations, angles = fit.fixed_ori_mol_placement(
        mol_grid_pts_1D=2, x_max=75, y_max=75)
    simTestInst_few_mol = SimulatedExperiment(
        locations, mol_angle=np.pi/2)
    simTestInst_few_mol.trial_images = (
        simTestInst_few_mol.calculate_BEM_fields()
        )
    # Plot
    simTestInst_few_mol.plot_mispol_map_wMisloc()

    ## Initial guess defaults to position of maximum intensity.
    test_BEM_fit_instance_few_mol = fit.FitModelToData(
        simTestInst_few_mol.BEM_images,
    #     ini_guess=simTestInst_few_mol.mol_locations
        )
    test_BEM_fit_few_mol = (
        test_BEM_fit_instance_few_mol.fit_model_to_image_data()
        )

    ## Plot fit results
    test_BEM_fit_instance_few_mol.plot_fit_results_as_quiver_map(
        simTestInst_few_mol
        )