
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/hashlookup/poppy/rust.yml?style=for-the-badge)

# Building

## Regular building

```bash
cargo build --release --bins
```

## Building with MUSL (static binary)

```bash
# You can skip this step if you already have musl installed
rustup target add x86_64-unknown-linux-musl
# Build poppy with musl target
cargo build --release --target=x86_64-unknown-linux-musl --bins
```

# CLI Usage

```
Usage: poppy <COMMAND>

Commands:
  create  Create a new bloom filter
  insert  Insert data into an existing bloom filter
  check   Checks entries against an existing bloom filter
  show    Show information about an existing bloom filter
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

## Difference with Go and C CLI

This implementation allows to **insert** and **check** entries in the bloom
filter in a multi-threaded fashion. You can specify a number of **jobs** to run
in parallel either with **insert** or **check** commands. It is worth noting that
this parallelization works only if several files are specified in the command line,
as parallelization occurs at a file level.

### Multi-threaded insertion VS single-threaded insertion

Trying to make a bulk insert with **poppy** in a single thread can be achieved by simply
piping entries to insert.

```
Command being timed: "bash -c cat src/data/all-hashes/*.txt | /tmp/poppy insert /tmp/hashlookup.bloom"
User time (seconds): 226.76
System time (seconds): 18.11
Percent of CPU this job got: 105%
Elapsed (wall clock) time (h:mm:ss or m:ss): 3:52.75
...
Page size (bytes): 4096
Exit status: 0
```

If one wants to achieve parallelization and expect some improvements in term of speed it is possible by providing
both a number of jobs (here 8) and a list of files containing entries to insert.
```
Command being timed: "bash -c /tmp/poppy insert -j 8 /tmp/hashlookup.bloom src/data/all-hashes/*.txt"
User time (seconds): 467.73
System time (seconds): 8.96
Percent of CPU this job got: 743%
Elapsed (wall clock) time (h:mm:ss or m:ss): 1:04.12
...
Exit status: 0
```


