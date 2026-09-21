using FFTW
using Printf
include(joinpath(@__DIR__, "mixed_reference.jl"))

# Separate sign-reference format: keep the 82 legacy fixtures untouched.
const DIRECTION_REFERENCE_VERSION = 1

function _direction_array(shape, seed)
    a = Array{ComplexF64}(undef, Tuple(reverse(shape)))
    for index in CartesianIndices(a)
        c = reverse(Tuple(index))
        a[index] = complex(seed + sum((axis + 0.17) * coordinate for (axis, coordinate) in enumerate(c)),
            -seed / 3 + sum((axis + 0.11) * (coordinate + 1) for (axis, coordinate) in enumerate(c)))
    end
    a
end

function _section(io, name, values)
    println(io, "section ", name, " complex ", length(values))
    for value in vec(values)
        @printf(io, "%.17g %.17g\n", real(value), imag(value))
    end
    println(io, "end")
end

function _write(path, name, shape, transforms, directions, provider, native)
    if :rfft in transforms
        boundary = findfirst(==(:rfft), transforms)
        input = real.(_direction_array(shape, 0.37))
        # Independent arbitrary reduced data, real endpoints before the complex suffix.
        reduced = copy(shape); reduced[boundary] = shape[boundary] ÷ 2 + 1
        inverse_input = _direction_array(reduced, 0.91)
        dim = length(shape) - boundary + 1
        selectdim(inverse_input, dim, 1) .= real.(selectdim(inverse_input, dim, 1))
        if iseven(shape[boundary])
            selectdim(inverse_input, dim, reduced[boundary]) .= real.(selectdim(inverse_input, dim, reduced[boundary]))
        end
        for axis in boundary-1:-1:1
            inverse_input = _axis_forward(inverse_input, length(shape)-axis+1, transforms[axis]; positive=directions[axis] == :backward)
        end
        forward = mixed_r2c_reference(input, transforms; directions=directions)
        inverse = mixed_r2c_reference(inverse_input, transforms; inverse=true, real_n=shape[boundary], directions=directions)
        backward = mixed_r2c_reference(inverse_input, transforms; backward=true, real_n=shape[boundary], directions=directions)
    else
        input = _direction_array(shape, 0.37)
        inverse_input = _direction_array(shape, 0.91)
        forward = mixed_c2c_reference(input, transforms; directions=directions)
        inverse = mixed_c2c_reference(inverse_input, transforms; inverse=true, directions=directions)
        backward = mixed_c2c_reference(inverse_input, transforms; backward=true, directions=directions)
    end
    open(path, "w") do io
        println(io, "PENCIL_FFTW_DIRECTION_REFERENCE ", DIRECTION_REFERENCE_VERSION)
        println(io, "runtime julia=", VERSION, " fftw_jl=", Base.pkgversion(FFTW),
            " native=", native, " provider=", provider)
        println(io, "case ", name)
        println(io, "shape ", join(shape, " "))
        println(io, "transforms ", join(string.(transforms), " "))
        println(io, "directions ", join(string.(directions), " "))
        _section(io, "input", input)
        _section(io, "inverse_input", inverse_input)
        _section(io, "forward_expected", forward)
        _section(io, "inverse_expected", inverse)
        _section(io, "backward_expected", backward)
    end
end

function main()
    length(ARGS) == 1 || error("usage: directions_reference.jl OUTPUT_DIRECTORY")
    output = abspath(ARGS[1]); mkpath(output)
    isempty(readdir(output)) || error("output directory must be empty: ", output)
    provider = String(FFTW.fftw_provider)
    provider == "fftw" || error("FFTW provider is $provider")
    VERSION == v"1.12.6" || error("Julia 1.12.6 is required")
    native = string(FFTW.version)
    cases = [
        ("mixed_r2c_odd", [3, 5, 2], [:fft, :rfft, :none], [:backward, :forward, :forward]),
        ("mixed_r2c_even", [3, 4, 2], [:fft, :rfft, :none], [:backward, :forward, :forward]),
        ("c2c_2d_3x4_forward-backward", [3, 4], [:fft, :fft], [:forward, :backward]),
        ("c2c_3d_3x2x4_backward-forward-forward", [3, 2, 4], [:fft, :fft, :fft], [:backward, :forward, :forward]),
        ("mixed_c2c_3d_3x2x4_backward-forward-forward", [3, 2, 4], [:fft, :dctii, :dht], [:backward, :forward, :forward]),
    ]
    for (name, shape, transforms, directions) in cases
        _write(joinpath(output, name * ".txt"), name, shape, transforms, directions, provider, native)
    end
    println("generated ", length(cases), " direction references in ", output)
end
main()
