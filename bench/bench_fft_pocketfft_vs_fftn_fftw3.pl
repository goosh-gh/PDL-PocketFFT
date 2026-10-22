#!/usr/bin/env perl
# bench_fft_pocketfft_vs_fftn_fftw3.pl
#
# Compares three FFT implementations available to PDL, in speed and accuracy:
#   fftn   : PDL::FFT (PDL's own lib/PDL/FFT/fftn.c), complex only
#   pocket : PDL::PocketFFT (bundled pocketfft C), complex and real
#   fftw3  : PDL::FFTW3 (FFTW 3, FFTW_ESTIMATE plans, cached), complex and real
#
# kind: complex / real return a new array; cplx-ip transforms a copy of the
# input in place (pocket: $z->inplace->cfft, fftw3: output = input), the way
# PDL::FFT works, so no new array is allocated in the timed call.
#
# For each length (and batches of short transforms, as for EEG epochs) it
# measures
#   first_ms : the first call (includes FFTW's planning)
#   med_ms   : the median of repeated calls
#   spec_err : error of the spectrum against the exact answer. The input is a
#              sum of complex exponentials (complex) or cosines (real) at integer
#              frequencies, whose transform is known in closed form; the phases
#              are computed from (t*k mod n), so the input itself is exact to
#              rounding. Reported as max|X - X_exact| / (n * max amplitude).
#   rt_err   : max|inverse(forward(x)) - x| / max|x|
#
# PDL::FFT runs in a forked child process for every case, because its fftn.c
# overflows a fixed array of 11 factors for some lengths and can abort the
# whole process ("abort" in the status column) or return wrong values (large
# spec_err). The column nf is the number of factors fftn.c's factorize()
# produces (more than 11 overflows).
#
# All libraries run single-threaded (PDL auto-pthreading off) unless --threads N.
#
# Usage:
#   perl -I<path to PDL-PocketFFT>/blib/lib -I<path>/blib/arch \
#        bench_fft_pocketfft_vs_fftn_fftw3.pl [--quick] [--budget 0.5] [--threads 0]
#                                             [--fftn-timeout 120] [--out results.tsv]
#     --quick         skip the cases with more than 2^21 points
#     --budget        seconds spent repeating each measurement (default 0.5); a single
#                     call longer than 4 budgets is not repeated
#     --fftn-timeout  seconds after which the PDL::FFT child is stopped (default 120)
#     --out           also write the table as TSV
#
# The status column of fftn shows the first line fftn.c printed to stderr, if any.
use strict;
use warnings;
use Getopt::Long;
use Time::HiRes qw(time);
use POSIX ();
use PDL;
use PDL::FFT ();
use PDL::PocketFFT qw(cfft icfft rfft irfft);
use PDL::FFTW3 ();

my %opt = (budget => 0.5, threads => 0, 'fftn-timeout' => 120);
GetOptions(\%opt, 'quick', 'budget=f', 'threads=i', 'out=s', 'fftn-timeout=i') or die "bad options\n";
set_autopthread_targ($opt{threads});

my $PI = 3.14159265358979323846;

# cases: [label, n, batch]
my @cases = (
    ['pow2',          1024,     1], ['pow2',     16384,   1], ['pow2',  65536, 1],
    ['pow2',       1048576,     1], ['pow2',   4194304,   1], ['pow2', 8388608, 1],
    ['2-3-5',         6000,     1], ['2-3-5',    30000,   1],
    ['2-3-5',       157464,     1],    # 2^3 3^9: smallest 2-3-5 length fftn.c overflows on
    ['2-3-5',       262440,     1],    # 2^3 3^8 5
    ['EEG 500Hz 10min', 300000, 1],
    ['EEG 1kHz 30min', 1800000, 1],
    ['EEG 1kHz 1h',  3600000,   1],
    ['prime',         1009,     1], ['prime',   100003,   1], ['prime', 1000003, 1],
    ['batch',          256,  4000],    # short epochs / TFA windows
    ['batch',         1000,  2000],
    ['batch',         1024,  2000],
);
@cases = grep { $_->[1] * $_->[2] <= 1048576 * 2 } @cases if $opt{quick};

# ---------------------------------------------------------------------------
# factor count of fftn.c's factorize() (a copy of its steps; metadata only)
sub fftn_nfactor {
    my $n = shift;
    my @f;
    my $kt;
    while ($n % 16 == 0) { push @f, 4; $n = int($n / 16) }
    my ($j, $jj) = (3, 9);
    do {
        while ($n % $jj == 0) { push @f, $j; $n = int($n / $jj) }
        $j += 2;
        $jj = $j * $j;
    } while ($jj <= $n);
    if ($n <= 4) { $kt = @f; push @f, $n if $n != 1 }
    else {
        if ($n % 4 == 0) { push @f, 2; $n = int($n / 4) }
        $kt = @f;
        $j = 2;
        do {
            if ($n % $j == 0) { push @f, $j; $n = int($n / $j) }
            $j = (int(($j + 1) / 2) << 1) + 1;
        } while ($j <= $n);
    }
    return @f + $kt;
}

sub median_of {
    my @s = sort { $a <=> $b } @_;
    return @s % 2 ? $s[$#s / 2] : 0.5 * ($s[@s / 2 - 1] + $s[@s / 2]);
}

# ---------------------------------------------------------------------------
# test signals with exact spectra. Frequencies are fixed fractions of n (so the
# same signal shape at every length), distinct, and not 0 or n/2.
sub freqs_for {
    my $n = shift;
    my %seen;
    my @k = grep { $_ > 0 && 2 * $_ != $n && !$seen{$_}++ }
            map { int($n * $_) } (0.013, 0.0707, 0.21, 0.333, 0.4471);
    return @k;
}

# phase(t, k) = 2 pi (t k mod n) / n, exact for t k < 2^63
sub phase_of {
    my ($n, $k) = @_;
    return 2 * $PI * ((sequence(longlong, $n) * $k) % $n)->double / $n;
}

# complex signal (n, B) and its exact spectrum
sub complex_signal {
    my ($n, $B) = @_;
    my @k   = freqs_for($n);
    my $z   = zeroes(cdouble, $n, $B);
    my $Xt  = zeroes(cdouble, $n, $B);
    my $amax = 0;
    for my $i (0 .. $#k) {
        # amplitude per row: deterministic, varies with row and frequency
        my $r   = sequence($B);
        my $amp = czip(cos(0.7 * $i + 0.11 * $r), sin(1.3 * $i + 0.05 * $r)) * (1 + 0.1 * $i);
        my $ph  = phase_of($n, $k[$i]);
        $z += exp(czip(zeroes($n), $ph))->dummy(1, $B) * $amp->dummy(0, $n);
        $Xt->slice("($k[$i]),:") .= $n * $amp;
        my $m = $amp->abs->max->sclr;
        $amax = $m if $m > $amax;
    }
    return ($z, $Xt, $amax);
}

# real signal (n, B) and its exact half spectrum (n/2+1, B)
sub real_signal {
    my ($n, $B) = @_;
    my @k   = freqs_for($n);
    my $x   = zeroes(double, $n, $B);
    my $Xt  = zeroes(cdouble, int($n / 2) + 1, $B);
    my $amax = 0;
    for my $i (0 .. $#k) {
        my $r    = sequence($B);
        my $amp  = 1 + 0.1 * $i + 0.05 * cos(0.3 * $r);
        my $phi  = 0.4 * $i + 0.02 * $r;
        my $ph   = phase_of($n, $k[$i]);
        $x += cos($ph->dummy(1, $B) + $phi->dummy(0, $n)) * $amp->dummy(0, $n);
        $Xt->slice("($k[$i]),:") .= ($n / 2) * $amp * exp(czip(zeroes($B), $phi));
        my $m = $amp->max->sclr;
        $amax = $m if $m > $amax;
    }
    return ($x, $Xt, $amax);
}

# ---------------------------------------------------------------------------
# time_calls($setup, $run): $setup returns fresh arguments (not timed), $run is
# timed. Returns (first_ms, median_ms, reps, result of the first call).
sub time_calls {
    my ($setup, $run) = @_;
    my @args = $setup->();
    my $t0 = time;
    my $res = $run->(@args);
    my $first = (time - $t0) * 1e3;
    # a single call slower than 4 budgets is not repeated: med_ms = first_ms
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

sub relerr { my ($p, $q, $scale) = @_; return ($p - $q)->abs->max->sclr / $scale }

my @rows;
sub row {
    my %r = @_;
    push @rows, \%r;
    printf "%-16s %8d %5d %3s  %-6s %-7s %10s %10s %10s %10s  %s\n",
        @r{qw(label n batch nf lib kind)},
        map({ defined $_ ? sprintf('%.3f', $_) : '-' } @r{qw(first_ms med_ms)}),
        map({ defined $_ ? sprintf('%.1e', $_) : '-' } @r{qw(spec_err rt_err)}),
        $r{status} // 'ok';
}

printf "PDL %s, PDL::PocketFFT %s, PDL::FFTW3 %s, perl %vd, autopthread targ %d, budget %.2f s\n",
    $PDL::VERSION, $PDL::PocketFFT::VERSION, $PDL::FFTW3::VERSION, $^V, $opt{threads}, $opt{budget};
printf "%-16s %8s %5s %3s  %-6s %-7s %10s %10s %10s %10s  %s\n",
    qw(case n batch nf lib kind first_ms med_ms spec_err rt_err status);

for my $c (@cases) {
    my ($label, $n, $B) = @$c;
    my $nf = fftn_nfactor($n);
    my %base = (label => $label, n => $n, batch => $B, nf => $nf);

    # ----- complex -----
    my ($z, $Xt, $amax) = complex_signal($n, $B);
    my $zmax = $z->abs->max->sclr;

    # fftn (PDL::FFT) in a child process
    {
        my $errfile = "/tmp/bench_fftn_stderr.$$";
        pipe(my $rd, my $wr) or die "pipe: $!";
        my $pid = fork() // die "fork: $!";
        if ($pid == 0) {
            close $rd;
            open STDERR, '>', $errfile or die "$errfile: $!";
            alarm $opt{'fftn-timeout'};
            my ($first, $med, $reps, $res) = time_calls(
                sub { ($z->re->copy, $z->im->copy) },
                sub { PDL::FFT::fft($_[0], $_[1]); [$_[0], $_[1]] });
            my $X  = czip(@$res);
            my $se = relerr($X, $Xt, $n * $amax);
            my ($r2, $i2) = ($X->re->copy, $X->im->copy);
            PDL::FFT::ifft($r2, $i2);
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
        # first line fftn.c wrote to stderr, if any ("Error: FFTRADIX() - ...")
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
            row(%base, lib => 'fftn', kind => 'complex', first_ms => $first, med_ms => $med,
                spec_err => $se, rt_err => $re, status => ($se > 1e-6 ? 'WRONG' : 'ok') . $msg);
        } else {
            my $sig = $st & 127;
            my $why = $sig == 14 ? "timeout (> $opt{'fftn-timeout'} s)"
                    : $sig       ? "abort (signal $sig)"
                    :              'exit ' . ($st >> 8);
            row(%base, lib => 'fftn', kind => 'complex', status => $why . $msg);
        }
    }

    # pocketfft
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($z) }, sub { cfft($_[0]) });
        row(%base, lib => 'pocket', kind => 'complex', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xt, $n * $amax), rt_err => relerr(icfft($X), $z, $zmax));
    }

    # FFTW3
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($z) }, sub { PDL::FFTW3::fft1($_[0]) });
        row(%base, lib => 'fftw3', kind => 'complex', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xt, $n * $amax), rt_err => relerr(PDL::FFTW3::ifft1($X), $z, $zmax));
    }
    # in place, as PDL::FFT is measured: the input copy is made outside the timing
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($z->copy) }, sub { $_[0]->inplace->cfft });
        row(%base, lib => 'pocket', kind => 'cplx-ip', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xt, $n * $amax), rt_err => relerr(icfft($X), $z, $zmax));
    }
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($z->copy) },
            sub { my $w = $_[0]; PDL::FFTW3::fft1($w, $w); $w });
        row(%base, lib => 'fftw3', kind => 'cplx-ip', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xt, $n * $amax), rt_err => relerr(PDL::FFTW3::ifft1($X), $z, $zmax));
    }
    undef $z;
    undef $Xt;

    # ----- real -----
    my ($x, $Xr, $rmax) = real_signal($n, $B);
    my $xmax = $x->abs->max->sclr;
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($x) }, sub { rfft($_[0]) });
        row(%base, lib => 'pocket', kind => 'real', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xr, $n * $rmax), rt_err => relerr(irfft($X, $n), $x, $xmax));
    }
    {
        my ($first, $med, $reps, $X) = time_calls(sub { ($x) }, sub { PDL::FFTW3::rfft1($_[0]) });
        row(%base, lib => 'fftw3', kind => 'real', first_ms => $first, med_ms => $med,
            spec_err => relerr($X, $Xr, $n * $rmax),
            rt_err => relerr(PDL::FFTW3::irfft1($X, zeroes(double, $n, $B)), $x, $xmax));
    }
}

if ($opt{out}) {
    open my $fh, '>', $opt{out} or die "$opt{out}: $!";
    my @cols = qw(label n batch nf lib kind first_ms med_ms spec_err rt_err status);
    print $fh join("\t", @cols), "\n";
    for my $r (@rows) { print $fh join("\t", map { $r->{$_} // ($_ eq 'status' ? 'ok' : '') } @cols), "\n" }
    close $fh;
    print "wrote $opt{out}\n";
}
