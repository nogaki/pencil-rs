using FFTW

# Small independent oracle for the public mixed-axis plans.  Axes are Rust
# logical axes (zero-based, in ascending order); Julia's dimension is N-axis.
# `transforms` uses :none, :fft, :rfft, :dcti/:dctii/:dctiii/:dctiv,
# :dsti/:dstii/:dstiii/:dstiv, and :dht.

const _R2R_CODES = Dict(
    :dcti => FFTW.REDFT00,
    :dctii => FFTW.REDFT10,
    :dctiii => FFTW.REDFT01,
    :dctiv => FFTW.REDFT11,
    :dsti => FFTW.RODFT00,
    :dstii => FFTW.RODFT10,
    :dstiii => FFTW.RODFT01,
    :dstiv => FFTW.RODFT11,
)

function _r2r_pair(kind)
    kind in (:dcti, :dctiv, :dsti, :dstiv) && return kind
    kind == :dctii && return :dctiii
    kind == :dctiii && return :dctii
    kind == :dstii && return :dstiii
    kind == :dstiii && return :dstii
    error("unknown R2R kind: $kind")
end

function _r2r_factor(kind, n)
    kind == :dcti && return 2 * (n - 1)
    kind == :dsti && return 2 * (n + 1)
    return 2 * n
end

function _dht_axis(a, dim)
    n = size(a, dim)
    out = similar(a)
    for index in CartesianIndices(a)
        coordinates = collect(Tuple(index))
        k = coordinates[dim] - 1
        value = zero(eltype(a))
        for j in 0:n-1
            coordinates[dim] = j + 1
            value += a[CartesianIndex(Tuple(coordinates))] *
                (cos(2pi * j * k / n) + sin(2pi * j * k / n))
        end
        out[index] = value
    end
    return out
end

function _axis_forward(a, dim, kind)
    kind == :none && return a
    kind == :fft && return FFTW.plan_fft(a, [dim]; flags=FFTW.ESTIMATE, num_threads=1) * a
    kind == :dht && return _dht_axis(a, dim)
    code = _R2R_CODES[kind]
    return FFTW.plan_r2r(a, [code], [dim]; flags=FFTW.ESTIMATE, num_threads=1) * a
end

function _axis_reverse(a, dim, kind, backward)
    kind == :none && return a
    if kind == :fft
        plan = backward ? FFTW.plan_bfft(a, [dim]; flags=FFTW.ESTIMATE, num_threads=1) :
            FFTW.plan_ifft(a, [dim]; flags=FFTW.ESTIMATE, num_threads=1)
        return plan * a
    end
    if kind == :dht
        raw = _dht_axis(a, dim)
        return backward ? raw : raw ./ size(a, dim)
    end
    pair = _r2r_pair(kind)
    code = _R2R_CODES[pair]
    raw = FFTW.plan_r2r(a, [code], [dim]; flags=FFTW.ESTIMATE, num_threads=1) * a
    return backward ? raw : raw ./ _r2r_factor(kind, size(a, dim))
end

"""Reference C2C mixed transform for a Julia array in reversed Rust order."""
function mixed_c2c_reference(input, transforms; backward=false, inverse=false)
    any(==( :rfft), transforms) && error("C2C cannot contain :rfft")
    output = copy(input)
    n = length(transforms)
    for rust_axis in n:-1:1
        dim = n - rust_axis + 1
        output = backward || inverse ?
            _axis_reverse(output, dim, transforms[rust_axis], backward) :
            _axis_forward(output, dim, transforms[rust_axis])
    end
    return output
end

"""Reference R2C/C2R mixed transform with one :rfft boundary."""
function mixed_r2c_reference(input, transforms; backward=false, inverse=false, real_n=nothing)
    boundaries = findall(==( :rfft), transforms)
    length(boundaries) == 1 || error("R2C requires exactly one :rfft")
    boundary = only(boundaries)
    (backward || inverse) && real_n === nothing && error("real_n is required for reverse R2C references")
    output = copy(input)
    n = length(transforms)
    if !backward && !inverse
        for rust_axis in n:-1:1
            dim = n - rust_axis + 1
            if rust_axis == boundary
                output = FFTW.plan_rfft(output, [dim]; flags=FFTW.ESTIMATE, num_threads=1) * output
            else
                output = _axis_forward(output, dim, transforms[rust_axis])
            end
        end
        return output
    end
    for rust_axis in 1:n
        dim = n - rust_axis + 1
        kind = transforms[rust_axis]
        if rust_axis == boundary
            n_real = real_n === nothing ? size(input, dim) : real_n
            output = backward ?
                FFTW.brfft(copy(output), n_real, [dim]) :
                FFTW.plan_irfft(output, n_real, [dim]; flags=FFTW.ESTIMATE, num_threads=1) * output
        else
            output = _axis_reverse(output, dim, kind, backward)
        end
    end
    return output
end
