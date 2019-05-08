%  DEMOSPECRET17 - Light scattering of metallic nanorod.
%    For a metallic nanorod and an incoming plane wave, this program
%    computes the scattering cross section for different light wavelengths
%    using the full Maxwell equations and an iterative BEM solver.
%
%  Runtime on my computer:  35 minutes

%%  initialization
%  options for BEM simulation
op = bemoptions( 'sim', 'ret' );
%  use iterative BEM solver
%    Output flag controls information about number of iterations and timing
%    of matrix evaluations.  For comparison you might also like to run the
%    program w/o iterative solvers by commenting the next line.
op.iter = bemiter.options( 'output', 1 );

%  table of dielectric functions
epstab = { epsconst( 1 ), epstable( 'silver.dat' ) };
%  nanorod
%    diameter 20 nm, length 800 nm, number of boundary elements 7378 
l1 = 200;
l2 = 175;

p1_z = trirod( 40, l1, [ 12, 12, 24 ] );
p1_x = rot(p1_z, 90, [0,1,0]);
p1 = shift( p1_x, [ -l1/2 - 1, 0, 0] );
     
p2_x = rot(trirod( 40 ,l2 , [12, 12, 24 ] ), 90, [0,1,0]);
p2 = shift( p2_x, [ +l2/2 + 1, 0, 0] );

p = comparticle( epstab, { p1, p2 }, [ 2, 1 ; 2, 1], 1, 2, op );

figure()
plot(p)


%%  BEM simulation
%  set up BEM solver
bem = bemsolver( p, op );

%  plane wave excitation
exc = planewave( [ 1, 0, 0], [0,1,0], op );
%  light wavelength in vacuum
%    we use a relatively small number of wavelengths to speed up the
%    simulations
% enei = linspace( 500, 900, 20 );
enei = linspace( 1240/1.55, 1240/1.4, 5 );
%  allocate scattering and extinction cross sections
sca = zeros( length( enei ), 2 );
ext = zeros( length( enei ), 2 );

multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
%  loop over wavelengths
for ien = 1 : length( enei )
  %  During the BEM evaluation Matlab keeps a copy of the BEM object.
  %  In case of restricted memory  it thus might be a good idea to clear
  %  all auxiliary matrices in BEM before calling the initialization.
  bem = clear( bem );
  %  initialize BEM solver
  bem = bem( enei( ien ) );
  %  surface charge
  [ sig, bem ] = solve( bem, exc( p, enei( ien ) ) );
  %  scattering and extinction cross sections
  sca( ien, : ) = exc.sca( sig );
  ext( ien, : ) = exc.ext( sig );
  
  multiWaitbar( 'BEM solver', ien / numel( enei ) );
end
%  close waitbar
multiWaitbar( 'CloseAll' );


%%  final plot
units;
eV_sca = eV2nm ./ enei;
figure()
plot(eV_sca, sca, 'o-'  );  hold on;

xlabel( 'hbar*w (eV)' );
ylabel( 'Scattering cross section (nm^2)' );

% %  print matrix compression and timing for H-matrix manipulation
% hinfo( bem );
% %  plot rank of low-rank matrices 
% plotrank( bem.G2 );
% colorbar;

hold off;
