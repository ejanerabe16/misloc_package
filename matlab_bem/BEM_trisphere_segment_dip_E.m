function [ e, sph_points ] = BEM_trisphere_segment_dip_E( 
    eps_b, sph_radius, cc_sep, ...
    drive_ene, dip_or, sph_points)
%% convert inputs from cgs to nm-grams-seconds

nm_per_cm = 1e7;

sph_radius = sph_radius * nm_per_cm;
cc_sep = cc_sep * nm_per_cm;
sph_points = sph_points .* nm_per_cm;

%%  initialization

%  options for BEM simulation
op_ret = bemoptions( 'sim', 'ret', 'interp', 'curv' );
op = op_ret;
units;

%  table of dielectric functions
% background_eps = 1.52^2;
background_eps = eps_b; % 1.0
epstab_drude = { epsconst( background_eps ), epsdrude( 'gold' ) }; 
% epsrab_data =  { epsconst( background_eps ), epstable( 'gold.dat' ) };
eps_use_for_fields = epstab_drude;

%  diameter of sphere
diameter = ( sph_radius * 2 ) ;  % 80

% center-center sphere-dipole seperation
sep = cc_sep;  % 50 - ...

if sep < diameter
    % sphere
    phi=linspace(0,2*pi,32);
    theta=[linspace(0,14*pi/16,28) linspace(14*pi/16, pi, 8)];
    p1 = trispheresegment( phi, theta, diameter );
    p1 = rot(p1, -90, [0,1,0]);
else
    p1 = trisphere( 576, diameter);
end
    
% shift to the left, so fluo is at center
pshifted = shift(p1, [-sep, 0, 0]);
%  initialize sphere
p = comparticle( eps_use_for_fields, { pshifted }, [ 2, 1], 1, op );

%%  dipole oscillator
%  dipole transition energy tuned to plasmon resonance frequency
%    plasmon resonance extracted from DEMODIPRET7
enei = eV2nm / drive_ene;  % lambda = 1240 / 2.6 eV

%  compoint, places dipole at origin
pt = compoint( p, [  0 , 0, 0] );

%  dipole excitation
dip = dipole( pt, dip_or, op );

%%  BEM simulation
%  set up BEM solver
bem = bemsolver( p, op, 'waitbar',0);
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
