using FFTW
using Printf

const REFERENCE_VERSION = 6

struct ReferenceCase
    name::String
    kind::Symbol
    spatial::Vector{Int}
    extra::Vector{Int}
    selected::Vector{Int}
    axis_kinds::Vector{Union{Nothing, Symbol}}
end

ReferenceCase(name, kind, spatial, extra, selected) = ReferenceCase(
    name,
    kind,
    spatial,
    extra,
    selected,
    Union{Nothing, Symbol}[nothing for _ in spatial],
)

ReferenceCase(name, kind, spatial, extra, selected, axis_kinds::AbstractVector) = ReferenceCase(
    name,
    kind,
    spatial,
    extra,
    selected,
    Union{Nothing, Symbol}[axis_kinds...],
)

function full_cases()
    return [
        ReferenceCase("c2c_2d_3x4", :c2c, [3, 4], Int[], [0, 1]),
        ReferenceCase("c2c_3d_3x2x5", :c2c, [3, 2, 5], [2, 3], [0, 1, 2]),
        ReferenceCase("c2c_4d_2x1x3x4", :c2c, [2, 1, 3, 4], [2], [0, 1, 2, 3]),
        ReferenceCase("r2c_2d_3x1", :r2c, [3, 1], Int[], [0, 1]),
        ReferenceCase("r2c_2d_3x2", :r2c, [3, 2], Int[], [0, 1]),
        ReferenceCase("r2c_3d_3x1x4_extra2x3", :r2c, [3, 1, 4], [2, 3], [0, 1, 2]),
        ReferenceCase("r2c_3d_3x1x5_extra2x3", :r2c, [3, 1, 5], [2, 3], [0, 1, 2]),
        ReferenceCase("r2c_4d_2x1x3x3_extra2", :r2c, [2, 1, 3, 3], [2], [0, 1, 2, 3]),
    ]
end

function partial_cases()
    return [
        ReferenceCase("c2c_4d_2x3x2x3_selnone", :c2c, [2, 3, 2, 3], Int[], Int[]),
        ReferenceCase("c2c_4d_2x3x2x3_sel0-3", :c2c, [2, 3, 2, 3], Int[], [0, 3]),
        ReferenceCase("r2c_4d_2x3x2x3_sel0", :r2c, [2, 3, 2, 3], Int[], [0]),
        ReferenceCase("r2c_4d_2x3x2x3_sel0-2", :r2c, [2, 3, 2, 3], Int[], [0, 2]),
        ReferenceCase("r2c_4d_2x3x2x3_sel0-3", :r2c, [2, 3, 2, 3], Int[], [0, 3]),
        ReferenceCase("r2c_3d_2x3x4_extra2_sel0-2", :r2c, [2, 3, 4], [2], [0, 2]),
    ]
end

function dht_cases()
    return [
        ReferenceCase(
            "dht_2d_3x4_dht-dht",
            :dht,
            [3, 4],
            Int[],
            [0, 1],
            [:dht, :dht],
        ),
        ReferenceCase(
            "dht_3d_3x2x4_extra2x3_dht-none-dht",
            :dht,
            [3, 2, 4],
            [2, 3],
            [0, 2],
            [:dht, nothing, :dht],
        ),
        ReferenceCase(
            "dht_4d_2x3x2x3_dht-none-none-dht",
            :dht,
            [2, 3, 2, 3],
            Int[],
            [0, 3],
            [:dht, nothing, nothing, :dht],
        ),
        ReferenceCase(
            "dht_4d_2x3x2x3_none-none-none-none",
            :dht,
            [2, 3, 2, 3],
            Int[],
            Int[],
            [nothing, nothing, nothing, nothing],
        ),
    ]
end

function r2r_cases()
    return [
        ReferenceCase("r2r_2d_3x4_dcti-dctii", :r2r, [3, 4], Int[], [0, 1], [:dcti, :dctii]),
        ReferenceCase("r2r_2d_3x4_dctiii-dctiv", :r2r, [3, 4], Int[], [0, 1], [:dctiii, :dctiv]),
        ReferenceCase("r2r_2d_3x4_dsti-dstii", :r2r, [3, 4], Int[], [0, 1], [:dsti, :dstii]),
        ReferenceCase("r2r_2d_3x4_dstiii-dstiv", :r2r, [3, 4], Int[], [0, 1], [:dstiii, :dstiv]),
        ReferenceCase("r2r_3d_3x2x4_extra2x3_dctii-none-dstii", :r2r, [3, 2, 4], [2, 3], [0, 2], [:dctii, nothing, :dstii]),
        ReferenceCase("r2r_4d_2x1x3x3_extra2_none-none-none-none", :r2r, [2, 1, 3, 3], [2], Int[], [nothing, nothing, nothing, nothing]),
    ]
end

function reference_cases()
    return vcat(full_cases(), partial_cases(), r2r_cases(), dht_cases())
end

function precision_name(::Type{Float32})
    return "f32"
end

function precision_name(::Type{Float64})
    return "f64"
end

precision_name(::Type{Complex{T}}) where {T} = precision_name(T)

element_kind(::Type{<:Complex}) = "complex"
element_kind(::Type) = "real"

function axis_kind_name(::Nothing)
    return "none"
end

axis_kind_name(kind::Symbol) = String(kind)

function logical_shape(item::ReferenceCase)
    shape = vcat(item.extra, item.spatial)
    all(>(0), shape) || error("reference shape has a zero extent")
    return shape
end

function logical_array(::Type{T}, item::ReferenceCase) where {T}
    return Array{T}(undef, Tuple(reverse(logical_shape(item))))
end

function logical_coordinates(index)
    return [Int(coordinate) - 1 for coordinate in reverse(Tuple(index))]
end

function complex_value(::Type{T}, coordinates::Vector{Int}, seed::Float64) where {T}
    real_part = 0.173 + 0.019 * seed
    imaginary_part = -0.271 - 0.013 * seed
    for (axis, coordinate) in enumerate(coordinates)
        real_part += (0.071 + 0.009 * axis) * (coordinate + 1)
        imaginary_part += (0.053 + 0.011 * axis) * (coordinate + 2)
        real_part += 0.017 * sin((axis + 2) * (coordinate + 1) + seed * 0.07)
        imaginary_part += 0.023 * cos((axis + 3) * (coordinate + 2) - seed * 0.05)
    end
    for left in 1:length(coordinates)-1
        for right in left+1:length(coordinates)
            real_part += 0.0067 * left * (right + 1) * (coordinates[left] + 1) * (coordinates[right] + 2)
            imaginary_part -= 0.0049 * (left + 1) * right * (coordinates[left] + 2) * (coordinates[right] + 1)
        end
    end
    return Complex{T}(T(real_part), T(imaginary_part))
end

function real_value(coordinates::Vector{Int}, seed::Float64)
    value = 0.417 + 0.023 * seed
    for (axis, coordinate) in enumerate(coordinates)
        value += (0.087 + 0.013 * axis) * (coordinate + 1)
        value += 0.031 * sin((axis + 1) * (coordinate + 2) + seed * 0.09)
    end
    for left in 1:length(coordinates)-1
        for right in left+1:length(coordinates)
            value += 0.0083 * (left + 1) * (right + 2) * (coordinates[left] + 1) * (coordinates[right] + 2)
        end
    end
    return value
end

function fill_complex!(array, seed::Float64)
    for index in CartesianIndices(array)
        array[index] = complex_value(eltype(array).parameters[1], logical_coordinates(index), seed)
    end
    return array
end

function fill_real!(array, seed::Float64)
    for index in CartesianIndices(array)
        array[index] = eltype(array)(real_value(logical_coordinates(index), seed))
    end
    return array
end

function selected_julia_dims(item::ReferenceCase)
    return sort([length(item.spatial) - axis for axis in item.selected])
end

function reduction_rust_axis(item::ReferenceCase)
    item.kind == :r2c || error("only R2C cases have a reduction axis")
    isempty(item.selected) && error("R2C selection must be nonempty")
    return maximum(item.selected)
end

function c2c_values(::Type{T}, item::ReferenceCase) where {T}
    input = fill_complex!(logical_array(Complex{T}, item), 1.25)
    inverse_input = fill_complex!(logical_array(Complex{T}, item), 23.75)
    expected_input_shape = Tuple(reverse(logical_shape(item)))
    @assert size(input) == expected_input_shape
    @assert size(inverse_input) == expected_input_shape
    dims = selected_julia_dims(item)
    input_before = copy(input)
    inverse_input_before = copy(inverse_input)
    if isempty(dims)
        forward = copy(input)
        inverse = copy(inverse_input)
        backward = copy(inverse_input)
    else
        forward_plan = FFTW.plan_fft(input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
        inverse_plan = FFTW.plan_ifft(inverse_input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
        forward = forward_plan * input
        inverse = inverse_plan * inverse_input
        backward = FFTW.bfft(copy(inverse_input), dims)
    end
    @assert size(forward) == expected_input_shape
    @assert size(inverse) == expected_input_shape
    @assert size(backward) == expected_input_shape
    @assert input == input_before
    @assert inverse_input == inverse_input_before
    return input, inverse_input, forward, inverse, backward
end

function r2c_values(::Type{T}, item::ReferenceCase) where {T}
    input = fill_real!(logical_array(T, item), 31.5)
    inverse_real_input = fill_real!(logical_array(T, item), 67.25)
    expected_input_shape = Tuple(reverse(logical_shape(item)))
    @assert size(input) == expected_input_shape
    @assert size(inverse_real_input) == expected_input_shape
    dims = selected_julia_dims(item)
    isempty(dims) && error("R2C selection must be nonempty")
    real_axis = reduction_rust_axis(item)
    real_n = item.spatial[real_axis + 1]
    input_before = copy(input)
    forward_plan = FFTW.plan_rfft(input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
    inverse_input = forward_plan * inverse_real_input
    reduced_spatial = copy(item.spatial)
    reduced_spatial[real_axis + 1] = real_n ÷ 2 + 1
    expected_output_shape = Tuple(reverse(vcat(item.extra, reduced_spatial)))
    @assert size(inverse_input) == expected_output_shape
    inverse_input_recorded = copy(inverse_input)
    inverse_plan_input = copy(inverse_input_recorded)
    @assert inverse_plan_input == inverse_input_recorded
    inverse_plan = FFTW.plan_irfft(
        inverse_plan_input,
        real_n,
        dims;
        flags = FFTW.ESTIMATE,
        num_threads = 1,
    )
    inverse_work = copy(inverse_input_recorded)
    @assert inverse_work == inverse_input_recorded
    inverse = inverse_plan * inverse_work
    backward = FFTW.brfft(copy(inverse_input_recorded), real_n, dims)
    forward = forward_plan * input
    @assert size(inverse) == expected_input_shape
    @assert size(backward) == expected_input_shape
    @assert size(forward) == expected_output_shape
    @assert inverse_input == inverse_input_recorded
    @assert input == input_before
    return input, inverse_input_recorded, forward, inverse, backward
end

function r2r_julia_kind(kind::Symbol)
    return Dict(
        :dcti => FFTW.REDFT00,
        :dctii => FFTW.REDFT10,
        :dctiii => FFTW.REDFT01,
        :dctiv => FFTW.REDFT11,
        :dsti => FFTW.RODFT00,
        :dstii => FFTW.RODFT10,
        :dstiii => FFTW.RODFT01,
        :dstiv => FFTW.RODFT11,
    )[kind]
end

function r2r_pair_kind(kind::Symbol)
    kind in (:dcti, :dctiv, :dsti, :dstiv) && return kind
    kind == :dctii && return :dctiii
    kind == :dctiii && return :dctii
    kind == :dstii && return :dstiii
    kind == :dstiii && return :dstii
    error("unknown R2R kind: ", kind)
end

function r2r_logical_factor(item::ReferenceCase)
    factor = 1
    for (axis, kind) in enumerate(item.axis_kinds)
        kind === nothing && continue
        n = item.spatial[axis]
        factor *= if kind == :dcti
            2 * (n - 1)
        elseif kind == :dsti
            2 * (n + 1)
        else
            2 * n
        end
    end
    return factor
end

function dht_axis(array, dimension)
    T = eltype(array)
    n = size(array, dimension)
    output = similar(array)
    for index in CartesianIndices(array)
        coordinates = collect(Tuple(index))
        k = coordinates[dimension] - 1
        value = zero(T)
        for j in 0:n-1
            coordinates[dimension] = j + 1
            value += array[CartesianIndex(Tuple(coordinates))] * T(cos(2pi * j * k / n) + sin(2pi * j * k / n))
        end
        output[index] = value
    end
    return output
end

function dht_values(::Type{T}, item::ReferenceCase) where {T}
    input = if T <: Complex
        fill_complex!(logical_array(T, item), 307.5)
    else
        fill_real!(logical_array(T, item), 307.5)
    end
    inverse_input = if T <: Complex
        fill_complex!(logical_array(T, item), 409.75)
    else
        fill_real!(logical_array(T, item), 409.75)
    end
    selected_dims = selected_julia_dims(item)
    forward = copy(input)
    backward = copy(inverse_input)
    for dimension in selected_dims
        forward = dht_axis(forward, dimension)
        backward = dht_axis(backward, dimension)
    end
    factor = prod(item.spatial[axis + 1] for axis in item.selected; init = 1)
    inverse = backward ./ T(factor)
    @assert size(forward) == Tuple(reverse(logical_shape(item)))
    @assert size(inverse) == Tuple(reverse(logical_shape(item)))
    @assert size(backward) == Tuple(reverse(logical_shape(item)))
    return input, inverse_input, forward, inverse, backward
end

function r2r_values(::Type{T}, item::ReferenceCase) where {T}
    input = if T <: Complex
        fill_complex!(logical_array(T, item), 101.5)
    else
        fill_real!(logical_array(T, item), 101.5)
    end
    inverse_input = if T <: Complex
        fill_complex!(logical_array(T, item), 203.75)
    else
        fill_real!(logical_array(T, item), 203.75)
    end
    expected_shape = Tuple(reverse(logical_shape(item)))
    @assert size(input) == expected_shape
    @assert size(inverse_input) == expected_shape
    selected = findall(kind -> kind !== nothing, item.axis_kinds)
    selected_dims_and_kinds = sort([
        (length(item.spatial) - axis + 1, item.axis_kinds[axis]) for axis in selected
    ]; by = first)
    dims = [pair[1] for pair in selected_dims_and_kinds]
    kinds = [r2r_julia_kind(pair[2]) for pair in selected_dims_and_kinds]
    paired_kinds = [r2r_julia_kind(r2r_pair_kind(pair[2])) for pair in selected_dims_and_kinds]
    input_before = copy(input)
    inverse_input_before = copy(inverse_input)
    if isempty(dims)
        forward = copy(input)
        backward = copy(inverse_input)
    else
        forward_plan = FFTW.plan_r2r(input, kinds, dims; flags = FFTW.ESTIMATE, num_threads = 1)
        backward_plan = FFTW.plan_r2r(inverse_input, paired_kinds, dims; flags = FFTW.ESTIMATE, num_threads = 1)
        forward = forward_plan * input
        backward = backward_plan * inverse_input
    end
    inverse = backward ./ r2r_logical_factor(item)
    @assert size(forward) == expected_shape
    @assert size(inverse) == expected_shape
    @assert size(backward) == expected_shape
    @assert input == input_before
    @assert inverse_input == inverse_input_before
    return input, inverse_input, forward, inverse, backward
end

function print_header(io, item::ReferenceCase, ::Type{T}, provider, native_version) where {T}
    println(io, "PENCIL_FFTW_REFERENCE ", REFERENCE_VERSION)
    println(
        io,
        "runtime julia=", VERSION,
        " fftw_jl=", Base.pkgversion(FFTW),
        " native=", native_version,
        " provider=", provider,
    )
    println(io, "case ", item.name, "_", precision_name(T))
    println(io, "kind ", item.kind)
    fixture_element_kind = item.kind == :c2c ? "complex" : item.kind == :r2c ? "real" : element_kind(T)
    println(io, "element_kind ", fixture_element_kind)
    println(io, "precision ", precision_name(T))
    println(io, "original_shape ", join(item.spatial, " "))
    println(io, "extra_shape", isempty(item.extra) ? "" : " " * join(item.extra, " "))
    println(io, "axis_kinds", isempty(item.axis_kinds) ? "" : " " * join(axis_kind_name.(item.axis_kinds), " "))
    println(io, "selected_axes", isempty(item.selected) ? "" : " " * join(item.selected, " "))
end

function print_real_section(io, name::String, values)
    println(io, "section ", name, " real ", length(values))
    for value in vec(values)
        @printf(io, "%.17g\n", Float64(value))
    end
    println(io, "end")
end

function print_complex_section(io, name::String, values)
    println(io, "section ", name, " complex ", length(values))
    for value in vec(values)
        @printf(io, "%.17g %.17g\n", Float64(real(value)), Float64(imag(value)))
    end
    println(io, "end")
end

function write_case(output_directory::String, item::ReferenceCase, ::Type{T}, provider, native_version) where {T}
    suffix = item.kind in (:r2r, :dht) ? "_" * element_kind(T) : ""
    filename = joinpath(output_directory, item.name * suffix * "_" * precision_name(T) * ".txt")
    open(filename, "w") do io
        print_header(io, item, T, provider, native_version)
        if item.kind == :c2c
            input, inverse_input, forward, inverse, backward = c2c_values(T, item)
            print_complex_section(io, "input", input)
            print_complex_section(io, "inverse_input", inverse_input)
            print_complex_section(io, "forward_expected", forward)
            print_complex_section(io, "inverse_expected", inverse)
            print_complex_section(io, "backward_expected", backward)
        elseif item.kind == :r2c
            input, inverse_input, forward, inverse, backward = r2c_values(T, item)
            print_real_section(io, "input", input)
            print_complex_section(io, "inverse_input", inverse_input)
            print_complex_section(io, "forward_expected", forward)
            print_real_section(io, "inverse_expected", inverse)
            print_real_section(io, "backward_expected", backward)
        elseif item.kind == :dht
            input, inverse_input, forward, inverse, backward = dht_values(T, item)
            if T <: Complex
                print_complex_section(io, "input", input)
                print_complex_section(io, "inverse_input", inverse_input)
                print_complex_section(io, "forward_expected", forward)
                print_complex_section(io, "inverse_expected", inverse)
                print_complex_section(io, "backward_expected", backward)
            else
                print_real_section(io, "input", input)
                print_real_section(io, "inverse_input", inverse_input)
                print_real_section(io, "forward_expected", forward)
                print_real_section(io, "inverse_expected", inverse)
                print_real_section(io, "backward_expected", backward)
            end
        else
            input, inverse_input, forward, inverse, backward = r2r_values(T, item)
            if T <: Complex
                print_complex_section(io, "input", input)
                print_complex_section(io, "inverse_input", inverse_input)
                print_complex_section(io, "forward_expected", forward)
                print_complex_section(io, "inverse_expected", inverse)
                print_complex_section(io, "backward_expected", backward)
            else
                print_real_section(io, "input", input)
                print_real_section(io, "inverse_input", inverse_input)
                print_real_section(io, "forward_expected", forward)
                print_real_section(io, "inverse_expected", inverse)
                print_real_section(io, "backward_expected", backward)
            end
        end
    end
end

function main()
    length(ARGS) == 1 || error("usage: generate.jl OUTPUT_DIRECTORY")
    output_directory = abspath(ARGS[1])
    mkpath(output_directory)
    isempty(readdir(output_directory)) || error("output directory must be empty: ", output_directory)
    VERSION == v"1.12.6" || error("Julia 1.12.6 is required, got ", VERSION)

    provider = String(FFTW.fftw_provider)
    provider == "fftw" || error("FFTW provider is $provider, expected fftw")
    FFTW.set_num_threads(1)
    native_version = string(FFTW.version)
    @info "generating Julia/FFTW reference fixtures" julia_version = VERSION fftw_jl_version = Base.pkgversion(FFTW) fftw_native_version = native_version provider = provider plan_flags = "ESTIMATE" threads = 1

    for item in vcat(full_cases(), partial_cases())
        write_case(output_directory, item, Float32, provider, native_version)
        write_case(output_directory, item, Float64, provider, native_version)
    end
    for item in r2r_cases()
        for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            write_case(output_directory, item, T, provider, native_version)
        end
    end
    for item in dht_cases()
        for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            write_case(output_directory, item, T, provider, native_version)
        end
    end
    files = filter(name -> endswith(name, ".txt"), readdir(output_directory))
    expected = 68
    length(files) == expected || error("generated ", length(files), " fixtures, expected ", expected)
    println("generated ", length(files), " fixtures in ", output_directory)
end

main()
