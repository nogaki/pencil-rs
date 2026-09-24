using Test
include("verify_sources.jl")

@testset "reference source integrity" begin
    mktempdir() do work
        source = joinpath(work, "Example")
        mkpath(joinpath(source, "src"))
        file = joinpath(source, "src", "Example.jl")
        original = "module Example\nend\n"
        write(file, original)
        chmod(file, 0o644)
        # Independent Git plumbing; all objects stay in a temporary bare repo.
        repo = joinpath(work, "git")
        run(`git init --bare --quiet $repo`)
        blob = readchomp(pipeline(`git hash-object --stdin`, stdin=IOBuffer(original)))
        subtree = readchomp(pipeline(`git -C $repo mktree --missing`, stdin=IOBuffer("100644 blob $blob\tExample.jl\n")))
        expected = readchomp(pipeline(`git -C $repo mktree --missing`, stdin=IOBuffer("040000 tree $subtree\tsrc\n")))
        @test verify_tree("Example", source, expected) == expected
        write(file, original * "# changed\n")
        @test_throws r"expected=.*observed=" verify_tree("Example", source, expected)
        rm(file)
        @test_throws r"expected=.*observed=" verify_tree("Example", source, expected)
        write(file, original)
        extra = joinpath(source, "extra.jl")
        write(extra, "extra")
        @test_throws r"expected=.*observed=" verify_tree("Example", source, expected)
        rm(extra)
        chmod(file, 0o755)
        @test_throws r"expected=.*observed=" verify_tree("Example", source, expected)
        chmod(file, 0o644)
        outside = joinpath(work, "outside.jl")
        write(outside, original)
        rm(file)
        symlink(outside, file)
        @test_throws r"unsupported entry" verify_tree("Example", source, expected)
        rm(outside) # dangling links also fail closed
        @test_throws r"unsupported entry" verify_tree("Example", source, expected)
        rm(file)
        write(file, original)
        symlink(source, joinpath(work, "linked"))
        @test_throws r"symlinked source" verify_tree("Example", joinpath(work, "linked"), expected)
        mkdir(joinpath(source, ".git"))
        @test_throws r"unsupported entry" verify_tree("Example", source, expected)
        rm(joinpath(source, ".git"))
        run(`mkfifo $extra`)
        @test_throws r"unsupported entry" verify_tree("Example", source, expected)
        rm(extra)
        @test_throws r"missing or symlinked" verify_tree("Example", joinpath(work, "missing"), expected)
        @test verify_tree("Example", source, expected) == expected

        # Hash-only manifest, installed exactly where Pkg selects it. Never load Example.
        write(file, "error(\"package must not be loaded\")\n")
        pinned = bytes2hex(Pkg.GitTools.tree_hash(source))
        uuid = UUID("7876af07-990d-54b4-ab0e-23690620f79a")
        project = joinpath(work, "Project.toml")
        write(project, "[deps]\nExample = \"$uuid\"\n")
        entry = Dict("uuid" => string(uuid), "version" => "1.0.0", "git-tree-sha1" => pinned)
        function manifest!()
            open(joinpath(work, "Manifest.toml"), "w") do io
                TOML.print(io, Dict("manifest_format" => "2.0", "julia_version" => string(VERSION),
                    "deps" => Dict("Example" => [entry])))
            end
        end
        manifest!()
        depots = [joinpath(work, "depot1"), joinpath(work, "depot2")]
        slug = Base.version_slug(uuid, Base.SHA1(pinned))
        roots = [joinpath(depot, "packages", "Example", slug) for depot in depots]
        mkpath(dirname(roots[1])); cp(source, roots[1])
        old_project, old_depots = Base.ACTIVE_PROJECT[], copy(DEPOT_PATH)
        check(; preflight=false) = verify_sources(; preflight, expected_count=1)
        try
            Base.ACTIVE_PROJECT[] = project
            empty!(DEPOT_PATH); append!(DEPOT_PATH, depots)
            @test isnothing(check(; preflight=true))
            @test isnothing(check())
            @test_throws r"expected 9.*found 1" verify_sources()
            for key in ("path", "repo-url", "repo-rev", "repo-subdir", "entryfile")
                entry[key] = source; manifest!()
                @test_throws r"unsupported manifest source override" check(; preflight=true)
                delete!(entry, key)
            end
            delete!(entry, "git-tree-sha1"); manifest!()
            @test_throws r"missing git-tree-sha1" check(; preflight=true)
            entry["git-tree-sha1"] = pinned; manifest!()
            rm(roots[1]; recursive=true)
            @test isnothing(check(; preflight=true))
            @test_throws r"missing selected package source" check()
            write(roots[1], "not a directory")
            @test_throws r"malformed source" check(; preflight=true)
            rm(roots[1]); symlink(source, roots[1])
            @test_throws r"symlinked source" check(; preflight=true)
            rm(roots[1]); symlink(joinpath(work, "absent"), roots[1])
            @test_throws r"symlinked source" check(; preflight=true)
            rm(roots[1]); cp(source, roots[1])
            rm(joinpath(roots[1], "src", "Example.jl"))
            @test_throws r"missing entrypoint" check(; preflight=true)
            @test_throws r"missing entrypoint" check()
            rm(roots[1]; recursive=true); cp(source, roots[1])
            mkpath(dirname(roots[2])); cp(source, roots[2])
            write(joinpath(roots[1], "extra"), "changed")
            @test_throws r"expected=.*observed=" check(; preflight=true)
            reverse!(DEPOT_PATH)
            @test Pkg.Operations.find_installed("Example", uuid, Base.SHA1(pinned)) == roots[2]
            @test isnothing(check())
            reverse!(DEPOT_PATH)
            @test_throws r"expected=.*observed=" check()
            rm(joinpath(roots[1], "extra"))

            # Pkg.instantiate runs this selector even with build/precompile disabled.
            marker = joinpath(work, "HOOK_EXECUTED")
            mkpath(joinpath(roots[1], ".pkg"))
            write(joinpath(roots[1], "Artifacts.toml"), "")
            write(joinpath(roots[1], ".pkg", "select_artifacts.jl"),
                "write($(repr(marker)), \"executed\"); error(\"malicious selector executed\")\n")
            # A private empty registry prevents default-registry network setup.
            registry = joinpath(depots[1], "registries", "Fixture")
            mkpath(registry)
            write(joinpath(registry, "Registry.toml"),
                "name = \"Fixture\"\nuuid = \"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\"\nrepo = \"https://invalid.invalid/fixture\"\n[packages]\n")
            helper = joinpath(@__DIR__, "verify_sources.jl")
            instantiate = "Pkg.instantiate(; update_registry=false, allow_build=false, allow_autoprecomp=false)"
            function child(code)
                command = `$(Base.julia_cmd()) --compiled-modules=no --startup-file=no --history-file=no --project=$work -L $helper -e $code`
                # Explicit allowlist; never inherit credentials or the user's depot.
                command = setenv(command, Dict("PATH" => "/usr/bin:/bin", "HOME" => work,
                    "JULIA_DEPOT_PATH" => join(depots, ':'), "JULIA_LOAD_PATH" => "@:@stdlib",
                    "JULIA_PKG_OFFLINE" => "true", "JULIA_PKG_PRECOMPILE_AUTO" => "0"))
                output = IOBuffer()
                process = run(pipeline(ignorestatus(command); stdout=output, stderr=output))
                return process.exitcode, String(take!(output))
            end
            status, message = child("verify_sources(; preflight=true, expected_count=1); $instantiate")
            @test status == 1
            @test occursin("expected=$pinned observed=", message)
            @test !ispath(marker)
            println("malicious selector blocked before instantiate; marker absent")
            status, message = child(instantiate)
            @test status == 1
            @test occursin("malicious selector executed", message)
            @test read(marker, String) == "executed"
            println("unguarded instantiate control executed malicious selector")
        finally
            Base.ACTIVE_PROJECT[] = old_project
            empty!(DEPOT_PATH); append!(DEPOT_PATH, old_depots)
        end
    end
end
