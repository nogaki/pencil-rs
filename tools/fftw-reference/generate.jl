using FFTW
using Printf

const REFERENCE_VERSION = 2

struct ReferenceCase
    name::String
    kind::Symbol
    spatial::Vector{Int}
    extra::Vector{Int}
end

function reference_cases()
    return [
        ReferenceCase("c2c_2d_3x4", :c2c, [3, 4], Int[]),
        ReferenceCase("c2c_3d_3x2x5", :c2c, [3, 2, 5], [2, 3]),
        ReferenceCase("c2c_4d_2x1x3x4", :c2c, [2, 1, 3, 4], [2]),
        ReferenceCase("r2c_2d_3x1", :r2c, [3, 1], Int[]),
        ReferenceCase("r2c_2d_3x2", :r2c, [3, 2], Int[]),
        ReferenceCase("r2c_3d_3x1x4_extra2x3", :r2c, [3, 1, 4], [2, 3]),
        ReferenceCase("r2c_3d_3x1x5_extra2x3", :r2c, [3, 1, 5], [2, 3]),
        ReferenceCase("r2c_4d_2x1x3x3_extra2", :r2c, [2, 1, 3, 3], [2]),
    ]
end

function precision_name(::Type{Float32})
    return "f32"
end

function precision_name(::Type{Float64})
    return "f64"
end

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

function c2c_values(::Type{T}, item::ReferenceCase) where {T}
    input = fill_complex!(logical_array(Complex{T}, item), 1.25)
    inverse_input = fill_complex!(logical_array(Complex{T}, item), 23.75)
    expected_input_shape = Tuple(reverse(logical_shape(item)))
    @assert size(input) == expected_input_shape
    @assert size(inverse_input) == expected_input_shape
    dims = 1:length(item.spatial)
    input_before = copy(input)
    inverse_input_before = copy(inverse_input)
    forward_plan = FFTW.plan_fft(input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
    inverse_plan = FFTW.plan_ifft(inverse_input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
    forward = forward_plan * input
    inverse = inverse_plan * inverse_input
    backward = FFTW.bfft(copy(inverse_input), dims)
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
    dims = 1:length(item.spatial)
    input_before = copy(input)
    forward_plan = FFTW.plan_rfft(input, dims; flags = FFTW.ESTIMATE, num_threads = 1)
    inverse_input = forward_plan * inverse_real_input
    reduced_spatial = copy(item.spatial)
    reduced_spatial[end] = reduced_spatial[end] ÷ 2 + 1
    expected_output_shape = Tuple(reverse(vcat(item.extra, reduced_spatial)))
    @assert size(inverse_input) == expected_output_shape
    inverse_input_recorded = copy(inverse_input)
    inverse_plan_input = copy(inverse_input_recorded)
    @assert inverse_plan_input == inverse_input_recorded
    inverse_plan = FFTW.plan_irfft(
        inverse_plan_input,
        item.spatial[end],
        dims;
        flags = FFTW.ESTIMATE,
        num_threads = 1,
    )
    inverse_work = copy(inverse_input_recorded)
    @assert inverse_work == inverse_input_recorded
    inverse = inverse_plan * inverse_work
    forward = forward_plan * input
    @assert size(inverse) == expected_input_shape
    @assert size(forward) == expected_output_shape
    @assert inverse_input == inverse_input_recorded
    @assert input == input_before
    return input, inverse_input_recorded, forward, inverse
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
    println(io, "precision ", precision_name(T))
    println(io, "original_shape ", join(item.spatial, " "))
    println(io, "extra_shape", isempty(item.extra) ? "" : " " * join(item.extra, " "))
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
    filename = joinpath(output_directory, item.name * "_" * precision_name(T) * ".txt")
    open(filename, "w") do io
        print_header(io, item, T, provider, native_version)
        if item.kind == :c2c
            input, inverse_input, forward, inverse, backward = c2c_values(T, item)
            print_complex_section(io, "input", input)
            print_complex_section(io, "inverse_input", inverse_input)
            print_complex_section(io, "forward_expected", forward)
            print_complex_section(io, "inverse_expected", inverse)
            print_complex_section(io, "backward_expected", backward)
        else
            input, inverse_input, forward, inverse = r2c_values(T, item)
            print_real_section(io, "input", input)
            print_complex_section(io, "inverse_input", inverse_input)
            print_complex_section(io, "forward_expected", forward)
            print_real_section(io, "inverse_expected", inverse)
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

    for item in reference_cases()
        write_case(output_directory, item, Float32, provider, native_version)
        write_case(output_directory, item, Float64, provider, native_version)
    end
    files = filter(name -> endswith(name, ".txt"), readdir(output_directory))
    length(files) == 16 || error("generated ", length(files), " fixtures, expected 16")
    println("generated ", length(files), " fixtures in ", output_directory)
end

main()
