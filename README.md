# PDL::PocketFFT

FFTs of any length for PDL: one-dimensional along dim 0, and n-dimensional
over the leading dims, broadcasting over the other dims. The transforms are
computed by
[pocketfft](https://github.com/mreineck/pocketfft) (C version, Martin Reinecke,
Max-Planck-Society, 3-clause BSD), which is bundled, so no external library is
needed.

```perl
use PDL;
use PDL::PocketFFT qw(cfft icfft rfft irfft fft_inplace ifft_inplace);

my $Z  = cfft($z);               # complex (n, ...) -> complex (n, ...)
my $z2 = icfft($Z);              # inverse, scaled by 1/n
my $X  = rfft($x);               # real (n, ...) -> complex (n/2+1, ...)
my $x2 = irfft($X, $n);          # back to real (n, ...), scaled by 1/n

fft_inplace($re, $im);           # PDL::FFT::fft conventions, in place
ifft_inplace($re, $im);          # PDL::FFT::ifft conventions, in place

use PDL::PocketFFT qw(cfftnd icfftnd rfftnd irfftnd);
my $Z2 = cfftnd($z, 2);          # over the first 2 dims, broadcasting over the rest
my $X3 = rfftnd($x, 3);          # real (n0, n1, n2, ...) -> complex (n0/2+1, n1, n2, ...)
my $x3 = irfftnd($X3, 3, $n0);
```

The n-dimensional functions take the rank (number of leading dims to
transform) as PDL::FFTW3's `fftn`, `rfftn` and friends do; their names differ
so that both modules can be imported together. Dim 0 is transformed row by
row; each further dim d is transformed in place on the view `mv(d, 1)`, with
the contiguous dim 0 cut by `splitdim` into blocks of `$PDL::PocketFFT::BLOCK`
columns (default 16) that are gathered and transformed together. The block
index is a broadcast dim, so PDL's auto-pthreading splits the work over
threads; the results are identical with and without threads and for any block
width.

- Forward transforms are unscaled, `X[k] = sum_t x[t] exp(-2 pi i k t / n)`;
  inverse transforms are scaled by 1/n (as PDL::FFT and PDL::FFTW3).
- Any length: lengths with large prime factors use Bluestein's algorithm, so
  every length costs O(n log n).
- Double precision; other input types are converted. `cfft` and `icfft` can
  be used inplace (`$z->inplace->cfft`).
- `irfft` takes the output length n (m = n/2+1 must hold); it defaults to
  2*(m-1). The imaginary parts of the zero-frequency term (and of the n/2 term
  when n is even) are ignored.

## Build

```
perl Makefile.PL
make
make test
make install
```

Requires PDL 2.088 or later.

## Tests

`t/01-transforms.t` checks the transforms against a direct O(n^2) DFT
(lengths 1 to 210, including primes), closed-form transforms of complex
exponentials and cosines at long lengths (157464, 262440, 65536 and the prime
100003), PDL::FFT for lengths it handles, round trips, broadcasting, strided
views, inplace use and argument checks. Negative controls check that the
opposite sign convention does not match.

`t/02-nd.t` checks `cfftnd`, `icfftnd`, `rfftnd` and `irfftnd` against a
direct DFT applied axis by axis (2 to 4 dims, ranks below ndims), plane waves
with known spectra (256x243, 1009x64, 60x48x50) and PDL::FFT::fftnd; that the
result is identical for block widths 1 to 1000 and with and without
auto-pthreading; that inputs are left untouched; strided views, odd n0 and the
rank checks.

## License

This module: same terms as Perl itself. Bundled `pocketfft_lib.c` (pocketfft.c renamed) and
`pocketfft.h`: 3-clause BSD, see `LICENSE.pocketfft`.
