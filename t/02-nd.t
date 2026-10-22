# t/02-nd.t — n-dimensional transforms of PDL::PocketFFT (cfftnd, icfftnd,
# rfftnd, irfftnd) against ground truths that do not use pocketfft: a direct
# DFT applied axis by axis (matrix products), plane waves whose transform is
# known in closed form, and PDL::FFT::fftnd for small sizes. Also checks that
# the result does not depend on the column block width or on PDL's
# auto-pthreading, that inputs are left untouched, and the argument checks.
use strict;
use warnings;
use Test::More;
use PDL;
use PDL::PocketFFT qw(cfftnd icfftnd rfftnd irfftnd);
use PDL::FFT ();

my $PI = 3.14159265358979323846;

sub maxerr { my ($p, $q) = @_; return ($p - $q)->abs->max->sclr }

# DFT matrix W[t, k] = exp(sign 2 pi i (t k mod n) / n), dims (t, k)
sub dft_matrix {
    my ($n, $sign) = @_;
    my $tk = (xvals(long, $n, $n) * yvals(long, $n, $n)) % $n;
    return exp(czip(zeroes($n, $n), $sign * 2 * $PI * $tk->double / $n));
}

# direct DFT along dim $d of $x (complex), returns a new ndarray
sub dft_axis {
    my ($x, $d, $sign) = @_;
    my $n = $x->dim($d);
    my $W = dft_matrix($n, $sign);                        # (t, k)
    my $v = $x->mv($d, 0);                                # (t, ...)
    # X[k, ...] = sum_t W[t, k] v[t, ...]
    my $X = ($v->dummy(1, $n) * $W)->sumover;             # (t, k, ...) -> (k, ...)
    return $X->mv(0, $d)->copy;
}

sub dft_nd {
    my ($x, $rank, $sign) = @_;
    my $y = $x->copy;
    $y = dft_axis($y, $_, $sign) for 0 .. $rank - 1;
    return $y;
}

# --- direct DFT, small arrays ---------------------------------------------
for my $dims ([8, 6], [5, 7], [17, 5, 3], [4, 9, 2, 3], [1, 9], [9, 1], [12, 13, 4]) {
    my $r = @$dims;
    my $z = czip(grandom(@$dims), grandom(@$dims));
    my $N = 1;
    $N *= $_ for @$dims;
    my $tol = 1e-13 * $N;
    my $label = join('x', @$dims);
    my $zc = $z->copy;
    cmp_ok(maxerr(cfftnd($z, $r), dft_nd($z, $r, -1)), '<', $tol, "cfftnd $label vs direct DFT");
    cmp_ok(maxerr(icfftnd($z, $r), dft_nd($z, $r, +1) / $N), '<', $tol, "icfftnd $label vs direct DFT");
    cmp_ok(maxerr($z, $zc), '==', 0, "cfftnd/icfftnd $label leave the input untouched");
    my $x  = grandom(@$dims);
    my $Xr = dft_nd(czip($x, zeroes(@$dims)), $r, -1)->slice('0:' . int($dims->[0] / 2));
    cmp_ok(maxerr(rfftnd($x, $r), $Xr), '<', $tol, "rfftnd $label vs direct DFT");
    cmp_ok(maxerr(irfftnd(rfftnd($x, $r), $r, $dims->[0]), $x), '<', 1e-14, "irfftnd(rfftnd) $label");
    if ($r > 1) {
        # rank smaller than ndims: the last dim is broadcast over
        cmp_ok(maxerr(cfftnd($z, $r - 1), dft_nd($z, $r - 1, -1)), '<', $tol, "cfftnd $label rank " . ($r - 1) . " broadcasts");
        # (a transform along a dim of length 1 is the identity, so only for longer last dims)
        cmp_ok(maxerr(cfftnd($z, $r), dft_nd($z, $r - 1, -1)), '>', 1e-3,
            "negative control: full-rank cfftnd $label differs from rank " . ($r - 1))
            if $dims->[-1] > 1;
    }
}

# --- plane waves, closed form, including lengths fftn.c overflows on ------
for my $case ([[256, 243], [17, 30]], [[1009, 64], [500, 7]], [[60, 48, 50], [7, 11, 13]]) {
    my ($dims, $k) = @$case;
    my $r = @$dims;
    my $ph = zeroes(@$dims);
    for my $d (0 .. $r - 1) {
        my $t = (sequence(long, $dims->[$d]) * $k->[$d]) % $dims->[$d];
        my $shape = [(1) x $r];
        $shape->[$d] = $dims->[$d];
        $ph += (2 * $PI * $t->double / $dims->[$d])->reshape(@$shape);
    }
    my $N = 1;
    $N *= $_ for @$dims;
    my $Z = cfftnd(exp(czip(zeroes(@$dims), $ph)), $r);
    my $want = zeroes(cdouble, @$dims);
    $want->set(@$k, $N);
    my $label = join('x', @$dims);
    cmp_ok(maxerr($Z, $want) / $N, '<', 1e-12, "cfftnd $label: plane wave -> N at its wavevector only");
}

# --- PDL::FFT::fftnd for small sizes --------------------------------------
for my $dims ([16, 8], [6, 10, 4]) {
    my ($re, $im) = (grandom(@$dims), grandom(@$dims));
    my ($pr, $pi) = ($re->copy, $im->copy);
    PDL::FFT::fftnd($pr, $pi);
    my $label = join('x', @$dims);
    cmp_ok(maxerr(cfftnd(czip($re, $im), scalar @$dims), czip($pr, $pi)), '<', 1e-11, "cfftnd $label agrees with PDL::FFT::fftnd");
}

# --- block width and threads do not change the result ----------------------
{
    my $z = czip(grandom(100, 37, 6), grandom(100, 37, 6));       # 100 = 6*16 + 4: tail block
    local $PDL::PocketFFT::BLOCK = 16;
    my $ref = cfftnd($z, 3);
    for my $B (1, 3, 7, 64, 1000) {
        local $PDL::PocketFFT::BLOCK = $B;
        cmp_ok(maxerr(cfftnd($z, 3), $ref), '==', 0, "cfftnd result identical with block width $B");
    }
    my $big = czip(grandom(128, 128, 80), grandom(128, 128, 80));  # > 1M elements
    my $old = get_autopthread_targ();
    set_autopthread_targ(0);
    my $single = cfftnd($big, 3);
    set_autopthread_targ(2);
    set_autopthread_size(0);
    my $multi = cfftnd($big, 3);
    set_autopthread_targ($old);
    cmp_ok(maxerr($single, $multi), '==', 0, 'cfftnd identical with and without auto-pthreading');
}

# --- strided input, odd n0, types -----------------------------------------
{
    my $x = grandom(10, 7, 9);
    my $v = $x->mv(2, 0);                                         # (9, 10, 7) strided view
    cmp_ok(maxerr(rfftnd($v, 3), rfftnd($v->copy, 3)), '<', 1e-15, 'rfftnd on a strided view');
    my $odd = grandom(9, 5);
    cmp_ok(maxerr(irfftnd(rfftnd($odd, 2), 2, 9), $odd), '<', 1e-14, 'irfftnd with odd n0');
    is(irfftnd(rfftnd($odd, 2), 2)->dim(0), 8, 'irfftnd default n0 = 2*(dim 0 - 1)');
    is(cfftnd(float(1, 2, 3, 4)->reshape(2, 2), 2)->type, 'cdouble', 'float input -> cdouble');
}

# --- inplace and method forms --------------------------------------------
{
    my $z   = czip(grandom(40, 30, 3), grandom(40, 30, 3));
    my $ref = cfftnd($z, 2);
    my $w   = $z->copy;
    my $ret = $w->inplace->cfftnd(2);
    cmp_ok(maxerr($w, $ref), '==', 0, 'inplace cfftnd transforms the array itself');
    ok(${$ret} == ${$w}, 'inplace cfftnd returns the same array');
    $w->inplace->icfftnd(2);
    cmp_ok(maxerr($w, $z), '<', 1e-14, 'inplace icfftnd undoes it');
    my $v = $z->copy;
    my $sl = $v->slice(':,:,(1)');
    $sl->inplace->cfftnd(2);
    cmp_ok(maxerr($v->slice(':,:,(1)'), $ref->slice(':,:,(1)')), '<', 1e-15, 'inplace cfftnd on a slice writes through to the parent');
    cmp_ok(maxerr($v->slice(':,:,(0)'), $z->slice(':,:,(0)')), '==', 0, '... and leaves the rest untouched');
    my $f = grandom(8, 6);
    my $g = $f->copy;
    my $F = $f->inplace->cfftnd(2);                      # real input: new array, input untouched
    cmp_ok(maxerr($f, $g), '==', 0, 'inplace cfftnd on real input leaves the input untouched');
    cmp_ok(maxerr($F, cfftnd($g, 2)), '<', 1e-15, '... and returns the transform');
    ok(!$f->is_inplace, '... and clears the inplace flag');
    my $x = grandom(10, 7);
    cmp_ok(maxerr($x->rfftnd(2)->irfftnd(2, 10), $x), '<', 1e-14, 'method forms rfftnd / irfftnd');
}

# --- argument checks ------------------------------------------------------
{
    my $z = zeroes(cdouble, 4, 4);
    ok(!eval { cfftnd($z); 1 },    'cfftnd croaks without rank');
    ok(!eval { cfftnd($z, 3); 1 }, 'cfftnd croaks when rank > ndims');
    ok(!eval { cfftnd($z, 0); 1 }, 'cfftnd croaks when rank < 1');
    like($@, qr/cfftnd: rank/, 'rank error names the function');
}

done_testing();
