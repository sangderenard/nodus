# fftfree Nodus tool row

`nodus.fftfree` is a native row registered with `register_fftfree_tool` and a
path to a built `fft_cffi` shared library. Its three ports are a U64
configuration tensor, a U64 work tensor, and a U64 receipt tensor. Both
arguments are passed through ordinary typed table edges. The transform data
stays at addresses held by `TensorStateLibraryTool`; the row borrows those
addresses only during execution. A table may publish its receipt to multiple
subscribers using the existing keyed edge FIFO.
`KernelLibrarySession::register_fftfree` owns the registry factory alongside
the session's state-library row, so table instances can use the row without
outliving the library it borrows from.

The configuration is U64[40]. Word 0 is schema version 1. Words 1–7 are FFT
size, threads, lanes, inverse, kernel, radix, and radix-pattern length.
Words 8–15 contain up to eight radix-pattern entries. Words 16–31 match
`fft_init_full_v2` from fftfree in this order: padding, window size, hop,
STFT mode, transform, reduce magnitude, store polar, half spectrum, outer
parallel, inner parallel, inner threads, save crash logs, silent crash
reports, apply windows, overlap-add, analysis-window kind. Words 32 and 33
are analysis-window float parameters (IEEE F32 bits in the low word).
Word 34 is synthesis-window kind; 35 and 36 are its two F32 parameters.
Words 37 and 38 are window normalization and COLA mode. Word 39 is the
opt-in experimental `enable_backup` argument to fftfree execution. Enum
values are those of the loaded fftfree build; Nodus forwards them without
defining a competing FFT configuration vocabulary. The row caches a plan
until this tensor changes.

The work token is U64[10]: operation (1 real batched, 2 complex batched),
library card, input real address, input imaginary address (zero for real),
output real/PCM address, output imaginary address, output magnitude address,
input sample/frame count, maximum output frames, and job ID. Real operation
requires three distinct F32 output addresses; complex operation uses the
real/PCM output address. Every used address must belong to the same card.
The caller preallocates enough F32 samples. The receipt is U64[9]: status,
job ID, produced frames, effective FFT size, bins per frame, output address,
elapsed nanoseconds, card, operation. Status zero means the transform
completed. No receipt owns or prolongs a library state lease.
Successful work also enters the state library's CPU performance window by
FFT size, kernel, transform and inverse mode, with separate lease, dispatch
and total durations.

The focused native test uses a fresh fftfree DLL to check an eight-sample
impulse, five half-spectrum bins, inverse reconstruction, and a second FFT
kernel with recovery enabled. A three-frame overlapping STFT checks the
window/hop and batched-output layout. These tests establish the Nodus row's DLL and
addressed-tensor path. The row is a one-dimensional FFT/STFT service;
T4's authored two-dimensional angular-spectrum propagation and its
T3/T5 continuation still require the separate wave arena row and frame
contract. A visual control panel has not yet been attached to this row.
