function [ e, sph_points ] = SphereDrude_NoSub_dipDrive_E( ...
    mol_location, ...
    drive_ene, ...
    mol_or, ...
    sph_points, ...
    eps_b_input)
%% convert inputs from cgs to nm-grams-seconds

nm_per_cm = 1e7;
% sph_radius = sph_radius * nm_per_cm;
% cc_sep = cc_sep * nm_per_cm;
% sph_points = sph_points .* nm_per_cm;
sph_points = (sph_points .* nm_per_cm);

%  options for BEM simulation
op_ret = bemoptions( 'sim', 'ret', 'interp', 'curv' );
op = op_ret;
units;

%  table of dielectric functions
% quartz_n = 1.455; % Approximate result from refractiveindex.info
water_eps = 1.778;

epstab = { ...
    epsconst( water_eps ), ...
    epsdrude( 'Au' ), ...
%     epsconst( quartz_n^2 ) ...
    };

%% Sphere
%  diameter of sphere
sph_radius = 40;
diameter = ( sph_radius * 2 ) ; 

% center-center sphere-dipole seperation
% sep = cc_sep;  % 50 - ...

% phi=linspace(0,2*pi,32);
% theta=[linspace(0,14*pi/16,28) linspace(14*pi/16, pi, 8)];
% p1 = trispheresegment( phi, theta, diameter );
% p1 = rot(p1, -90, [0,1,0]);

p = trisphere( 300, diameter);
p = comparticle( epstab, { p }, [ 2, 1; ], 1, op );


%%  dipole oscillator
%  dipole transition energy tuned to plasmon resonance frequency
%    plasmon resonance extracted from DEMODIPRET7
enei = eV2nm / drive_ene;  % lambda = 1240 / 'drive_ene'

%  compoint, places dipole at origin
pt = compoint( p, mol_location );

%  dipole excitation
dip = dipole( pt, mol_or, op );

%%  BEM simulation
%  set up BEM solver
bem = bemsolver( p, op, 'waitbar', 0);
%  surface charge
sig = bem \ dip( p, enei );

%%  computation of electric field on sphere 
% REWRITE THIS SECTION!!!
%for diffracted field calculation

x_sph = sph_points(:,1); 
y_sph = sph_points(:,2);
z_sph = sph_points(:,3);
%  object for electric field
%    MINDIST controls the minimal distance of the field points to the
%    particle boundary

emesh = meshfield( p, x_sph(:), y_sph(:), z_sph(:), op, 'mindist', 0.5 );

%  induced and incoming electric field
% [ e_ind, h_ind ] = field( emesh_sph, sig );
% [ e_dip, h_dip ] = field( emesh_sph, dip.field( emesh_sph.pt, enei ) );
e = emesh( sig ) + emesh( dip.field( emesh.pt, enei ) );

end
