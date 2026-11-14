# T4 arena state row

`nodus.wave.t4.state` uses the authored Spectral `wave_t4.dll` C entry,
which batches compatible arenas through the existing fftfree-backed
angular-spectrum code. `KernelLibrarySession::register_wave_t4` registers
the row with the same tensor state library used by the shader rows.

The control port accepts U64[8]: schema 1, grid width and height, arena
count, bands per arena, IEEE F64 bits for cell pitch and propagation step,
and signed direction. The work port accepts U64[4+2*arena count]: library
card, wavelength-state address, frame identity, continuation generation,
then real and imaginary F32 state addresses for each arena. The wavelength
state is F64[arena count * bands]. Each field state holds at least
width * height * bands samples. All addresses belong to the same card.
The row refuses a work item before changing field state if any medium
wavelength is nonpositive, nonfinite, or shorter than twice the cell pitch.
This is the existing two samples per wavelength arena admission rule; callers
must provide wavelengths in the arena medium.
The row leases fields once, runs the authored batch entry, and releases the
loans before emitting a U64[9] receipt: status, frame identity, generation,
arena count, bands, width, height, elapsed nanoseconds, card. Successful
work enters the state library's CPU performance window. The existing table
edge FIFO supports independent subscribers for the receipt; `top_k=0` keeps
the continuation edge lossless.

A native test sent two 16×16 arena fields and their wavelengths through the
state library, ran the T4 row, and read the same receipt from two subscribers.
A separate Spectral C-API check compared a two-arena batch with two independent
authored calls. These checks cover the real Nodus state and FFT path. They do
not establish T3 conversion into a wavelength-sized surface region, full
Jones continuation, separated-region energy conservation, or a T5 image.
