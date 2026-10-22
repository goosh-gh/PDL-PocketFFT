# t/01-transforms.t — PDL::PocketFFT against ground truths that do not use pocketfft:
# a direct O(n^2) DFT, transforms of single complex exponentials and cosines
# (known in closed form), and PDL::FFT for lengths it handles. Includes negative
# controls (the opposite sign convention must NOT match).
use strict;
use warnings;
use Test::More;
use PDL;
use PDL::PocketFFT qw(cfft icfft rfft irfft fft_inplace ifft_inplace);
use PDL::FFT ();

my $PI = 3.14159265358979323846;

# direct DFT: X[k] = sum_t x[t] exp(sign * 2 pi i (t k mod n) / n); dims of x: (n)
sub dft_direct {
    my ($x, $sign) = @_;
    my $n  = $x->dim(0);
    my $tk = (xvals(long, $n, $n) * yvals(long, $n, $n)) % $n;   # dim0 = t, dim1 = k
    my $W  = exp(czip(zeroes($n, $n), $sign * 2 * $PI * $tk->double / $n));
    return ($W * $x)->sumover;                                    # -> (k)
}

sub maxerr { my ($p, $q) = @_; return ($p - $q)->abs->max->sclr }

CORE::srand(20261022);
sub rnd { my $n = shift; return pdl([map { CORE::rand() - 0.5 } 1 .. $n]) }

# --- direct DFT, small lengths (primes, powers of 2, mixed) ---------------
for my $n (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 25, 30, 31, 64, 97, 100, 127, 210) {
    my $re = rnd($n);
    my $im = rnd($n);
    my $z  = czip($re, $im);
    my $ref = dft_direct($z, -1);
    my $tol = 1e-13 * $n;
    cmp_ok(maxerr(cfft($z), $ref), '<', $tol, "cfft n=$n vs direct DFT");
    cmp_ok(maxerr(icfft($z), dft_direct($z, +1) / $n), '<', $tol, "icfft n=$n vs direct DFT");
    my $rref = dft_direct($re, -1)->slice('0:' . int($n / 2));
    cmp_ok(maxerr(rfft($re), $rref), '<', $tol, "rfft n=$n vs direct DFT");
    is(rfft($re)->dim(0), int($n / 2) + 1, "rfft n=$n gives n/2+1 frequencies");
    my ($r2, $i2) = ($re->copy, $im->copy);
    fft_inplace($r2, $i2);
    cmp_ok(maxerr(czip($r2, $i2), $ref), '<', $tol, "fft_inplace n=$n vs direct DFT");
    # negative control: the opposite sign must not match (n >= 3 random data)
    if ($n >= 3) {
        cmp_ok(maxerr(cfft($z), dft_direct($z, +1)), '>', 1e-3, "negative control: cfft n=$n is not the + sign DFT");
    }
}

# --- round trips ----------------------------------------------------------
for my $n (1, 2, 3, 10, 11, 1000, 1009, 4096) {
    my $z = czip(rnd($n), rnd($n));
    cmp_ok(maxerr(icfft(cfft($z)), $z), '<', 1e-14, "icfft(cfft) n=$n");
    my $x = rnd($n);
    cmp_ok(maxerr(irfft(rfft($x), $n), $x), '<', 1e-14, "irfft(rfft) n=$n");
    my ($re, $im) = (rnd($n), rnd($n));
    my ($r2, $i2) = ($re->copy, $im->copy);
    fft_inplace($r2, $i2);
    ifft_inplace($r2, $i2);
    cmp_ok(maxerr(czip($r2, $i2), czip($re, $im)), '<', 1e-14, "ifft_inplace(fft_inplace) n=$n");
}

# --- closed forms at long lengths, including those PDL::FFT's fftn.c
# --- overflows on (157464 = 2^3 3^9, 262440 = 2^3 3^8 5, prime 100003) ---
for my $n (157464, 262440, 100003, 65536) {
    my $k0 = 1234;
    my $t  = sequence($n);
    my $ph = 2 * $PI * (($t * $k0) % $n) / $n;              # exact phase, reduced mod n
    my $z  = exp(czip(zeroes($n), $ph));                     # exp(+2 pi i k0 t / n)
    my $Z  = cfft($z);
    my $want = zeroes(cdouble, $n);
    $want->set($k0, $n);
    cmp_ok(maxerr($Z, $want) / $n, '<', 1e-12, "cfft n=$n: exp(i k0) -> n at k0 only");
    my $X = rfft(cos($ph));
    my $wr = zeroes(cdouble, int($n / 2) + 1);
    $wr->set($k0, $n / 2);
    cmp_ok(maxerr($X, $wr) / $n, '<', 1e-12, "rfft n=$n: cos(k0) -> n/2 at k0 only");
}

# --- agreement with PDL::FFT for lengths fftn.c handles --------------------
for my $n (8, 60, 1000, 4096, 6000) {
    my ($re, $im) = (rnd($n), rnd($n));
    my ($pr, $pi) = ($re->copy, $im->copy);
    PDL::FFT::fft($pr, $pi);
    my ($qr, $qi) = ($re->copy, $im->copy);
    fft_inplace($qr, $qi);
    cmp_ok(maxerr(czip($pr, $pi), czip($qr, $qi)), '<', 1e-10, "fft_inplace n=$n agrees with PDL::FFT::fft");
    PDL::FFT::ifft($pr, $pi);
    ifft_inplace($qr, $qi);
    cmp_ok(maxerr(czip($pr, $pi), czip($qr, $qi)), '<', 1e-13, "ifft_inplace n=$n agrees with PDL::FFT::ifft");
}

# --- broadcasting, slices, inplace, types ----------------------------------
{
    my $x = grandom(48, 3, 2);
    my $X = rfft($x);
    is(join(',', $X->dims), '25,3,2', 'rfft broadcasts: dims (48,3,2) -> (25,3,2)');
    my $worst = 0;
    for my $j (0 .. 2) {
        for my $l (0 .. 1) {
            my $e = maxerr($X->slice(":,($j),($l)"), rfft($x->slice(":,($j),($l)")->copy));
            $worst = $e if $e > $worst;
        }
    }
    cmp_ok($worst, '<', 1e-15, 'rfft broadcast rows equal single-row transforms');

    my $z  = czip(grandom(30, 4), grandom(30, 4));
    my $zt = $z->transpose->copy;                             # (4,30)
    my $viaslice = cfft($zt->transpose);                      # non-contiguous view
    cmp_ok(maxerr($viaslice, cfft($z)), '<', 1e-15, 'cfft on a transposed (strided) view');

    my $w = $z->copy;
    $w->inplace->cfft;
    cmp_ok(maxerr($w, cfft($z)), '<', 1e-15, 'cfft inplace');
    $w->inplace->icfft;
    cmp_ok(maxerr($w, $z), '<', 1e-14, 'icfft inplace');

    my $f = float([1, 2, 3, 4]);
    is(rfft($f)->type, 'cdouble', 'float input -> cdouble output');
    is(cfft(pdl([1, 0, 0, 0]))->type, 'cdouble', 'real input to cfft -> cdouble');
    cmp_ok(maxerr(cfft(pdl([1, 0, 0, 0])), czip(ones(4), zeroes(4))), '<', 1e-16, 'cfft of delta is all ones');
    cmp_ok(maxerr(rfft(pdl([1, 2, 3, 4])), czip(pdl([10, -2, -2]), pdl([0, 2, 0]))), '<', 1e-15,
        'rfft [1,2,3,4] = [10, -2+2i, -2] (by hand)');
}

# --- irfft argument checks -------------------------------------------------
{
    my $X = rfft(rnd(9));                                     # 5 frequencies
    ok(!eval { irfft($X, 12); 1 }, 'irfft croaks when n/2+1 != number of frequencies');
    like($@, qr/irfft/, 'irfft error message names irfft');
    is(irfft($X, 9)->dim(0), 9, 'irfft with odd n given explicitly');
    is(irfft($X)->dim(0), 8, 'irfft default n = 2*(m-1)');
    ok(!eval { fft_inplace(zeroes(4), zeroes(5)); 1 }, 'fft_inplace croaks on mismatched dims');
}

done_testing();
