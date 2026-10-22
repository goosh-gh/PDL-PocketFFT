#!/usr/bin/env perl
# bench_fftnd_pocketfft_vs_fftn_fftw3.pl
#
# Compares n-dimensional FFTs available to PDL, in speed and accuracy, each
# with PDL's auto-pthreading off (1 thread) and on (all cores):
#   fftn   : PDL::FFT::fftnd (PDL's own fftn.c), complex, all dims; PDL::FFT::fft for rank 1
#   pocket : PDL::PocketFFT cfftnd / rfftnd (bundled pocketfft C)
#   fftw3  : PDL::FFTW3 fftn / rfftn (FFTW 3, FFTW_ESTIMATE plans, cached)
#
# kind: complex / real return a new array; cplx-ip transforms a copy of the
# input in place (pocket: $z->inplace->cfftnd, fftw3: output = input), the way
# PDL::FFT works, so no new array is allocated in the timed call.
#
# Cases are full-rank 2-D and 3-D transforms, and rank-1 transforms broadcast
# over many rows (EEG: samples x channels x epochs), where the work is split
# over threads by PDL's broadcasting.
#
# Columns (as in bench_fft_pocketfft_vs_fftn_fftw3.pl):
#   first_ms : the first call (includes FFTW's planning)
#   med_ms   : median of repeated calls
#   spec_err : error against the exact spectrum of a sum of plane waves with
#              integer wavevectors (phases from (t*k mod n), exact to rounding),
#              max|X - X_exact| / (N * max amplitude), N = product of the
#              transformed lengths
#   rt_err   : max|inverse(forward(x)) - x| / max|x|
#
# PDL::FFT runs in a forked child process (fftn.c can abort the process on
# lengths with more than 11 factors); its rows show "abort" or the first line
# it printed to stderr.
#
# Usage:
#   perl -I<PDL-PocketFFT>/blib/lib -I<PDL-PocketFFT>/blib/arch \
#        bench_fftnd_pocketfft_vs_fftn_fftw3.pl [--quick] [--budget 0.5] [--threads 0,8]
#                                               [--fftn-timeout 120] [--out results.tsv]
#     --threads       comma-separated auto-pthread targets to run (default: 0 and the
#                     number of CPUs)
#     --quick         skip the cases with more than 2^22 points
use strict;
use warnings;
use Getopt::Long;
use Time::HiRes qw(time);
use POSIX ();
use PDL;
use PDL::FFT ();
use PDL::PocketFFT qw(cfftnd icfftnd rfftnd irfftnd);
use PDL::FFTW3 ();

sub ncpu {
    for my $cmd ('sysctl -n hw.ncpu 2>/dev/null', 'nproc 2>/dev/null') {
        my $n = `$cmd`;
        return $1 if defined $n && $n =~ /(\d+)/;
    }
    return 1;
}

my %opt = (budget => 0.5, 'fftn-timeout' => 120, threads => '0,' . ncpu());
GetOptions(\%opt, 'quick', 'budget=f', 'threads=s', 'out=s', 'fftn-timeout=i') or die "bad options\n";
my @threads = split /,/, $opt{threads};
set_autopthread_size(1);     # let PDL split anything above 1M elements (its default threshold)

my $PI = 3.14159265358979323846;

# cases: [label, [dims], rank]
my @cases = (
    ['2-D',            [512, 512],        2],
    ['2-D',            [1000, 1000],      2],
    ['2-D',            [2048, 2048],      2],
    ['2-D',            [4096, 4096],      2],
    ['2-D prime',      [1009, 1009],      2],
    ['3-D',            [64, 64, 64],      3],
    ['3-D',            [100, 100, 100],   3],
    ['3-D',            [256, 256, 256],   3],
    ['EEG epochsxch',  [1000, 64, 200],   1],   # samples x channels x epochs, rank 1
    ['EEG epochsxch',  [1024, 64, 200],   1],
);
@cases = grep { my $N = 1; $N *= $_ for @{ $_->[1] }; $N <= 4194304 } @cases if $opt{quick};

sub median_of {
    my @s = sort { $a <=> $b } @_;
    return @s % 2 ? $s[$#s / 2] : 0.5 * ($s[@s / 2 - 1] + $s[@s / 2]);
}

sub time_calls {
    my ($setup, $run) = @_;
    my @args = $setup->();
    my $t0 = time;
    my $res = $run->(@args);
    my $first = (time - $t0) * 1e3;
    return ($first, $first, 1, $res) if $first > 4e3 * $opt{budget};
    my @t;
    my $spent = 0;
    while (@t < 3 || ($spent < $opt{budget} && @t < 50)) {
        @args = $setup->();
        $t0 = time;
        $run->(@args);
        my $dt = time - $t0;
        push @t, $dt * 1e3;
        $spent += $dt;
    }
    return ($first, median_of(@t), scalar(@t), $res);
}

# phase of one plane wave along the transformed dims, broadcast to @dims
sub plane_phase {
    my ($dims, $rank, $k) = @_;
    my $ph = zeroes(double, @$dims);
    for my $d (0 .. $rank - 1) {
        my $n = $dims->[$d];
        my $t = (sequence(longlong, $n) * $k->[$d]) % $n;
        my @shape = (1) x @$dims;
        $shape[$d] = $n;
        $ph += (2 * $PI * $t->double / $n)->reshape(@shape);
    }
    return $ph;
}

# wavevectors: fixed fractions of each length, distinct, avoiding 0 and n/2 in dim 0
sub wavevectors {
    my ($dims, $rank) = @_;
    my @f = ([0.013, 0.21], [0.0707, 0.333], [0.21, 0.07], [0.333, 0.45]);
    my @k;
    for my $fr (@f) {
        my @kv = map { my $n = $dims->[$_]; my $v = int($n * $fr->[$_ % 2]); $v = 1 if $_ == 0 && ($v == 0 || 2 * $v == $n); $v } 0 .. $rank - 1;
        push @k, \@kv;
    }
    my %seen;
    return grep { !$seen{join ',', @$_}++ } @k;
}

# complex signal and exact spectrum; the amplitude varies over the broadcast dims
sub complex_signal {
    my ($dims, $rank) = @_;
    my @k = wavevectors($dims, $rank);
    my $z  = zeroes(cdouble, @$dims);
    my $Xt = zeroes(cdouble, @$dims);
    my $N = 1;
    $N *= $dims->[$_] for 0 .. $rank - 1;
    my @bdims = @$dims[$rank .. $#$dims];
    my $amax = 0;
    for my $i (0 .. $#k) {
        my $r   = @bdims ? sequence(@bdims) : pdl(0);
        my $amp = czip(cos(0.7 * $i + 0.011 * $r), sin(1.3 * $i + 0.005 * $r)) * (1 + 0.1 * $i);
        my $wave = exp(czip(zeroes(@$dims), plane_phase($dims, $rank, $k[$i])));
        # (reshape would change $amp itself, which is still needed below; dummy makes a view)
        my $ampb = $amp;
        $ampb = $ampb->dummy(0) for 1 .. $rank;
        $z += $wave * $ampb;
        my $sl = join(',', map({ "($_)" } @{ $k[$i] }), (':') x @bdims);
        $Xt->slice($sl) .= $N * $amp;
        my $m = $amp->abs->max->sclr;
        $amax = $m if $m > $amax;
    }
    return ($z, $Xt, $amax, $N);
}

sub real_signal {
    my ($dims, $rank) = @_;
    my @k = wavevectors($dims, $rank);
    my $x = zeroes(double, @$dims);
    my @hd = @$dims;
    $hd[0] = int($hd[0] / 2) + 1;
    my $Xt = zeroes(cdouble, @hd);
    my $N = 1;
    $N *= $dims->[$_] for 0 .. $rank - 1;
    for my $i (0 .. $#k) {
        my $amp = 1 + 0.1 * $i;
        my $phi = 0.4 * $i;
        $x += $amp * cos(plane_phase($dims, $rank, $k[$i]) + $phi);
        # cos -> half amplitude at +k; dim 0 of k is in 1 .. n0/2-1, so only +k lands in the half spectrum
        my $sl = join(',', map({ "($_)" } @{ $k[$i] }), (':') x (@$dims - $rank));
        $Xt->slice($sl) .= ($N / 2) * $amp * exp(czip(0, $phi));
    }
    return ($x, $Xt, 1 + 0.1 * $#k, $N);
}

sub relerr { my ($p, $q, $scale) = @_; return ($p - $q)->abs->max->sclr / $scale }

my @rows;
sub row {
    my %r = @_;
    push @rows, \%r;
    printf "%-14s %-14s %4s %3s  %-6s %-7s %10s %10s %10s %10s  %s\n",
        @r{qw(label dims rank threads lib kind)},
        map({ defined $_ ? sprintf('%.2f', $_) : '-' } @r{qw(first_ms med_ms)}),
        map({ defined $_ ? sprintf('%.1e', $_) : '-' } @r{qw(spec_err rt_err)}),
        $r{status} // 'ok';
}

printf "PDL %s, PDL::PocketFFT %s (block %d), PDL::FFTW3 %s, perl %vd, threads %s, budget %.2f s\n",
    $PDL::VERSION, $PDL::PocketFFT::VERSION, $PDL::PocketFFT::BLOCK, $PDL::FFTW3::VERSION, $^V,
    join('/', @threads), $opt{budget};
printf "%-14s %-14s %4s %3s  %-6s %-7s %10s %10s %10s %10s  %s\n",
    qw(case dims rank thr lib kind first_ms med_ms spec_err rt_err status);

for my $c (@cases) {
    my ($label, $dims, $rank) = @$c;
    my %base = (label => $label, dims => join('x', @$dims), rank => $rank);
    my ($z, $Xt, $amax, $N) = complex_signal($dims, $rank);
    my $zmax = $z->abs->max->sclr;
    my ($x, $Xr, $rmax) = real_signal($dims, $rank);
    my $xmax = $x->abs->max->sclr;

    for my $th0 (@threads) {
        my $th = $th0;
        set_autopthread_targ($th);
        my %b = (%base, threads => $th);
        # PDL 2.104: an ndarray last processed under another auto-pthread target
        # croaks ("Cannot magick non-broadcasted dims") in operations with active
        # dims (sumover, the transforms) after set_autopthread_targ changes it.
        # Fresh copies made under the current target do not.
        my ($z, $x) = ($z->copy, $x->copy);

        # PDL::FFT in a child process: fftnd for full-rank cases, fft (dim 0,
        # broadcasting) for rank 1; other ranks are skipped
        if ($rank == @$dims || $rank == 1) {
            my ($fwd, $inv) = $rank == @$dims ? (\&PDL::FFT::fftnd, \&PDL::FFT::ifftnd)
                                              : (\&PDL::FFT::fft,   \&PDL::FFT::ifft);
            my $errfile = "/tmp/bench_fftnd_stderr.$$";
            pipe(my $rd, my $wr) or die "pipe: $!";
            my $pid = fork() // die "fork: $!";
            if ($pid == 0) {
                close $rd;
                open STDERR, '>', $errfile or die "$errfile: $!";
                alarm $opt{'fftn-timeout'};
                my ($first, $med, $reps, $res) = time_calls(
                    sub { ($z->re->copy, $z->im->copy) },
                    sub { $fwd->($_[0], $_[1]); [$_[0], $_[1]] });
                my $X  = czip(@$res);
                my $se = relerr($X, $Xt, $N * $amax);
                my ($r2, $i2) = ($X->re->copy, $X->im->copy);
                $inv->($r2, $i2);
                my $re = relerr(czip($r2, $i2), $z, $zmax);
                print $wr join("\t", $first, $med, $reps, $se, $re), "\n";
                close $wr;
                POSIX::_exit(0);
            }
            close $wr;
            my $line = <$rd>;
            close $rd;
            waitpid($pid, 0);
            my $st = $?;
            my $msg = '';
            if (open my $ef, '<', $errfile) {
                my $l = <$ef>;
                close $ef;
                if (defined $l) { chomp $l; $l =~ s/^Error:\s*//; $msg = " [stderr: $l]" }
            }
            unlink $errfile;
            if (defined $line && $st == 0) {
                chomp $line;
                my ($first, $med, $reps, $se, $re) = split /\t/, $line;
                row(%b, lib => 'fftn', kind => 'complex', first_ms => $first, med_ms => $med,
                    spec_err => $se, rt_err => $re, status => ($se > 1e-6 ? 'WRONG' : 'ok') . $msg);
            } else {
                my $sig = $st & 127;
                my $why = $sig == 14 ? "timeout (> $opt{'fftn-timeout'} s)"
                        : $sig       ? "abort (signal $sig)"
                        :              'exit ' . ($st >> 8);
                row(%b, lib => 'fftn', kind => 'complex', status => $why . $msg);
            }
        }

        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($z) }, sub { cfftnd($_[0], $rank) });
            row(%b, lib => 'pocket', kind => 'complex', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xt, $N * $amax), rt_err => relerr(icfftnd($X, $rank), $z, $zmax));
        }
        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($z) }, sub { PDL::FFTW3::fftn($_[0], $rank) });
            row(%b, lib => 'fftw3', kind => 'complex', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xt, $N * $amax), rt_err => relerr(PDL::FFTW3::ifftn($X, $rank), $z, $zmax));
        }
        # in place, as PDL::FFT is measured: the input copy is made outside the timing
        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($z->copy) }, sub { $_[0]->inplace->cfftnd($rank) });
            row(%b, lib => 'pocket', kind => 'cplx-ip', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xt, $N * $amax), rt_err => relerr(icfftnd($X, $rank), $z, $zmax));
        }
        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($z->copy) },
                sub { my $w = $_[0]; PDL::FFTW3::fftn($w, $rank, $w); $w });
            row(%b, lib => 'fftw3', kind => 'cplx-ip', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xt, $N * $amax), rt_err => relerr(PDL::FFTW3::ifftn($X, $rank), $z, $zmax));
        }
        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($x) }, sub { rfftnd($_[0], $rank) });
            row(%b, lib => 'pocket', kind => 'real', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xr, $N * $rmax),
                rt_err => relerr(irfftnd($X, $rank, $dims->[0]), $x, $xmax));
        }
        {
            my ($first, $med, $reps, $X) = time_calls(sub { ($x) }, sub { PDL::FFTW3::rfftn($_[0], $rank) });
            row(%b, lib => 'fftw3', kind => 'real', first_ms => $first, med_ms => $med,
                spec_err => relerr($X, $Xr, $N * $rmax),
                rt_err => relerr(PDL::FFTW3::irfftn($X, $rank, zeroes(double, @$dims)), $x, $xmax));
        }
    }
}

if ($opt{out}) {
    open my $fh, '>', $opt{out} or die "$opt{out}: $!";
    my @cols = qw(label dims rank threads lib kind first_ms med_ms spec_err rt_err status);
    print $fh join("\t", @cols), "\n";
    for my $r (@rows) { print $fh join("\t", map { $r->{$_} // ($_ eq 'status' ? 'ok' : '') } @cols), "\n" }
    close $fh;
    print "wrote $opt{out}\n";
}
