# Notices

This project is an independent Rust implementation inspired by:

PencilArrays.jl\
Copyright (c) 2020 Juan Ignacio Polanco <jipolanc@gmail.com> and contributors\
https://github.com/jipolanco/PencilArrays.jl\
MIT License

PencilFFTs.jl\
Copyright (c) 2019 Juan Ignacio Polanco\
https://github.com/jipolanco/PencilFFTs.jl\
MIT License

Reference commits used by the design:

- PencilArrays.jl `12229b99b827e07880517982c3365a18d1f9b8dc`
- PencilFFTs.jl `1d98a3ff790c40445987ad64b99eb3b946a11034`

## Optional native FFTW

The new Rust `pencil-fftw` adapter source is part of this MIT-licensed project.
It does not vendor FFTW source or binaries. The optional backend loads a separately
installed FFTW runtime; ordinary RustFFT/RealFFT builds do not require it.

FFTW itself is distributed under the GNU General Public License, version 2 or
(at the recipient's option) any later version. Commercial licensing is also
available from its rights holders. Enabling or distributing FFTW-backed software
requires evaluating and complying with the applicable FFTW licensing terms.
Dynamic loading is **not** a GPL exemption or a determination that a combined
distribution is MIT-only. This notice does not provide legal advice.

- FFTW: https://www.fftw.org/
- License and copyright: https://www.fftw.org/doc/License-and-Copyright.html
- The runtime's own copyright/license notices remain authoritative for the
  installed or redistributed version.
