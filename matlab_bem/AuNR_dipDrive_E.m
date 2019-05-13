function [ e, sph_points ] = AuNR_dipDrive_E( ...
    mol_location, ...
    drive_ene, ...
    mol_or, ...
    sph_points,...
    eps_b_input ...
    )
%% convert inputs from cgs to nm-grams-seconds

nm_per_cm = 1e7;
% sph_radius = sph_radius * nm_per_cm;
% cc_sep = cc_sep * nm_per_cm;
sph_points = sph_points .* nm_per_cm;

%%  initialization

%  options for BEM simulation
op_ret = bemoptions( 'sim', 'ret', 'interp', 'curv' );
op = op_ret;
units;

%  table of dielectric functions
% background_eps = 1.52^2;
background_eps =  eps_b_input ; % SHOULD BE 1.778
epstab_drude = { epsconst( background_eps ), epsdrude( 'gold' ) }; 
% epsrab_data =  { epsconst( background_eps ), epstable( 'gold.dat' ) };
eps_use_for_fields = epstab_drude;

p_z = trirod( 40, 88, [ 12, 12, 12 ] );
p_y = rot(p_z, 90, [1,0,0]);
p = comparticle( eps_use_for_fields, { p_y }, [ 2, 1 ], 1, op );
% figure()
% plot(p)
% axis on

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
%for diffracted field calculation

x_sph = sph_points(:,1); 
y_sph = sph_points(:,2);
z_sph = sph_points(:,3);
%  object for electric field
%    MINDIST controls the minimal distance of the field points to the
%    particle boundary

emesh_sph = meshfield( p, x_sph(:), y_sph(:), z_sph(:), op, 'mindist', 0.5 );

%  induced and incoming electric field
[ e_ind, h_ind ] = field( emesh_sph, sig );
[ e_dip, h_dip ] = field( emesh_sph, dip.field( emesh_sph.pt, enei ) );

e = e_ind + e_dip;
% h = h_ind + h_dip; 

end
