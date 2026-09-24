# Only stdlibs: run before importing any manifest package.
using Pkg, TOML, UUIDs

function verify_tree(name, source, expected)
    isdir(source) && !islink(source) || error("reference source integrity: $name missing or symlinked source: $source")
    # Git ignores .git and special files are not valid package inputs. Reject
    # symlinks conservatively rather than trusting content outside this tree.
    for (root, dirs, files) in walkdir(source; follow_symlinks=false)
        for entry in vcat(dirs, files)
            path = joinpath(root, entry)
            entry != ".git" && !islink(path) && (isdir(path) || isfile(path)) ||
                error("reference source integrity: $name unsupported entry: $path")
        end
    end
    observed = bytes2hex(Pkg.GitTools.tree_hash(source))
    observed == expected || error("reference source integrity: $name path=$source expected=$expected observed=$observed; refusing to load packages (cache not repaired)")
    println("reference source verified: $name path=$source git-tree-sha1=$observed")
    return observed
end

function verify_sources(; preflight=false, expected_count=9)
    manifest = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    count = 0
    verified = 0
    for (name, entries) in sort!(collect(manifest["deps"]); by=first)
        for entry in entries
            any(key -> haskey(entry, key), ("path", "repo-url", "repo-rev", "repo-subdir", "entryfile")) &&
                error("reference source integrity: $name unsupported manifest source override")
            uuid = UUID(entry["uuid"])
            if !haskey(entry, "git-tree-sha1")
                Pkg.Types.is_stdlib(uuid) && continue
                error("reference source integrity: $name missing git-tree-sha1")
            end
            count += 1
            hash = Base.SHA1(entry["git-tree-sha1"])
            source = Pkg.Operations.find_installed(name, uuid, hash)
            # Pkg ignores dangling links when selecting roots. Do not mistake
            # those (or symlinked parent directories) for downloadable absence.
            for slug in (Base.version_slug(uuid, hash), Base.version_slug(uuid, hash, 4)), depot in DEPOT_PATH
                candidate = abspath(depot, "packages", name, slug)
                for path in (abspath(depot), dirname(dirname(candidate)), dirname(candidate), candidate)
                    islink(path) && error("reference source integrity: $name symlinked source: $path")
                    ispath(path) && !isdir(path) && error("reference source integrity: $name malformed source: $path")
                end
            end
            if !ispath(source)
                preflight && continue # Only pre-instantiate may download absent trees.
                error("reference source integrity: $name missing selected package source: $source")
            end
            isfile(joinpath(source, "src", name * ".jl")) ||
                error("reference source integrity: $name missing entrypoint: $source")
            verify_tree(name, source, entry["git-tree-sha1"])
            verified += 1
        end
    end
    count == expected_count || error("reference source integrity: expected $expected_count pinned package trees, found $count")
    println("reference source integrity: verified $verified of $count pinned package trees (preflight=$preflight)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_sources()
end
