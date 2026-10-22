# Makefile.PL for PDL::PocketFFT — builds the PDL::PP module from pocketfft.pd
# and links the bundled pocketfft_lib.c (3-clause BSD) into it.
use strict;
use warnings;
use ExtUtils::MakeMaker;
use PDL::Core::Dev;

my $package = ['pocketfft.pd', 'PocketFFT', 'PDL::PocketFFT'];
my %args = pdlpp_stdargs($package);
$args{OBJECT} .= ' pocketfft_lib$(OBJ_EXT)';
$args{INC}    .= ' -I.';

WriteMakefile(
    %args,
    NAME             => 'PDL::PocketFFT',
    VERSION_FROM     => 'pocketfft.pd',
    ABSTRACT         => 'FFT of any length for PDL, by the bundled pocketfft C library',
    LICENSE          => 'perl',
    MIN_PERL_VERSION => '5.016',
    CONFIGURE_REQUIRES => { 'PDL' => '2.088', 'ExtUtils::MakeMaker' => 0 },
    PREREQ_PM          => { 'PDL' => '2.088' },
    TEST_REQUIRES      => { 'Test::More' => '0.88' },
    clean => { FILES => 'PocketFFT.pm PocketFFT.xs PocketFFT$(OBJ_EXT) PocketFFT.c pocketfft_lib$(OBJ_EXT) pp-*.c' },
);

sub MY::postamble { pdlpp_postamble($package) }
