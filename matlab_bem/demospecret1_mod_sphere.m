%  DEMOSPECRET1 - Light scattering of metallic nanosphere.
%    For a metallic nanosphere and an incoming plane wave, this program
%    computes the scattering cross section for different light wavelengths
%    using the full Maxwell equations, and compares the results with Mie
%    theory.
%
%  Runtime on my computer:  7.4 sec.

%%  initialization
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' );

%  table of dielectric functions
quartz_n = 1.455; % Approximate result from refractiveindex.info
water_eps = 1.778;

epstab = { ...
    epsconst( water_eps ), ...
    epsdrude( 'Au' ), ...
    };

%  diameter of sphere
diameter = 80;
%  initialize sphere
p = comparticle( epstab, { trisphere( 300, diameter ) }, [ 2, 1 ], 1, op );
figure()

plot(p, 'EdgeColor', 'b')
axis on
%%  BEM simulation
%  set up BEM solver
bem = bemsolver( p, op );

%  plane wave excitation
exc = planewave( [ 1, 0, 0; 0, 1, 0; 0, 0, 1 ], ...
    [ 0, 0, 1; 0, 0, 1; 0, 1, 0 ], op );
%  light wavelength in vacuum
enei = linspace( 450, 650, 150 );
%  allocate scattering and extinction cross sections
sca = zeros( length( enei ), 3 );
ext = zeros( length( enei ), 3 );

multiWaitbar( 'BEM solver', 0, 'Color', 'g', 'CanCancel', 'on' );
%  loop over wavelengths
for ien = 1 : length( enei )
  %  surface charge
  sig = bem \ exc( p, enei( ien ) );
  %  scattering and extinction cross sections
  sca( ien, : ) = exc.sca( sig );
  ext( ien, : ) = exc.ext( sig );
  
  multiWaitbar( 'BEM solver', ien / numel( enei ) );
end
%  close waitbar
multiWaitbar( 'CloseAll' );

%%  final plot
figure
plot( enei, sca, 'o-'  );  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering cross section (nm^2)' );


%% Save spectra
save('DrudeSphere80nm_inWater.mat', 'enei', 'sca')
