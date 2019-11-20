function [ e, sph_points ] = CurlyDiskJC_NoSub_dipDrive_E( ...
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
quartz_n = 1.455; % Approximate result from refractiveindex.info
water_eps = 1.778;

epstab = { ...
    epsconst( water_eps ), ...
    epstable( 'gold.dat' ), ...
    epsconst( quartz_n^2 ) ...
    };

%% Build disk
%  polygon for disk
poly = polygon( 25, 'size', [ 80, 80 ] );
%  edge profile for nanodisk
%    MODE '01' produces a rounded edge on top and a sharp edge on bottom,
%    MIN controls the lower z-value of the nanoparticle 
edge = edgeprofile( 30, 11, 'mode', '21', 'min', 1e-3 );
%  extrude polygon to nanoparticle
p = tripolygon( poly, edge );

%% Shift particle down so center is at z=0
p = shift(p, [0, 0, -15])
%  set up COMPARTICLE objects
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
