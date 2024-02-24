# poppy - Rust implementation of the DCSO Bloom filter tool

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/hashlookup/poppy/rust.yml?style=for-the-badge)

Bloom is a straightforward tool offering an efficient implementation of the Bloom filter for the Rust language. It includes a command-line tool that allows users
to effortlessly create [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter) with their desired capacity and false positive probability.
Values can be added to the filters via standard input, facilitating the use of this tool in a pipeline workflow. poppy is compatible with the [DCSO bloom software](https://github.com/DCSO/bloom).

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

# Funding

The NGSOTI project is dedicated to training the next generation of Security Operation Center (SOC) operators, focusing on the human aspect of cybersecurity.
It underscores the significance of providing SOC operators with the necessary skills and open-source tools to address challenges such as detection engineering, 
incident response, and threat intelligence analysis. Involving key partners such as CIRCL, Restena, Tenzir, and the University of Luxembourg, the project aims
to establish a real operational infrastructure for practical training. This initiative integrates academic curricula with industry insights, 
offering hands-on experience in cyber ranges.

NGSOTI is co-funded under Digital Europe Programme (DEP) via the ECCC (European cybersecurity competence network and competence centre).

