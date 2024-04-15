# poppy

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/hashlookup/poppy/rust.yml?style=for-the-badge)](https://github.com/hashlookup/poppy/actions)

**Poppy** is a Rust crate offering an efficient implementation of [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter). It also includes a **command-line
utility** (also called poppy) allowing users to effortlessly create filters with their desired capacity and false positive probability.
Values can be added to the filters via standard input, facilitating the use of this tool in a pipeline workflow. 

Poppy **ensures cross-compatibility with** the bloom filter format used by [DCSO bloom software](https://github.com/DCSO/bloom) but **also provides its own** Bloom filter implementation and format.

## FAQ

### Which format to choose ?

It depends what you want to achieve. If you want to be cross compatible with DCSO tools and library, you must absolutely choose DCSO format. In any other scenario
we advice to use Poppy format (the default), as it is more robust, faster and provides room for customization. A comparison between the two formats and implementations can be found
[in this blog post](https://www.misp-project.org/2024/03/25/Poppy-a-new-bloom-filter-format-and-project.html/)

### How to build the project ?

#### Regular building

```bash
cargo build --release --bins
```

#### Building with MUSL (static binary)

```bash
# You can skip this step if you already have musl installed
rustup target add x86_64-unknown-linux-musl
# Build poppy with musl target
cargo build --release --target=x86_64-unknown-linux-musl --bins
```

### How to use Poppy in other languages ?

#### In Python

Poppy comes with Python bindings, using the great [PyO3 crate](https://github.com/PyO3/pyo3).

Please take a look at [Poppy Bindings](./poppy-py) for further details.

## CLI Usage

```
Usage: poppy [OPTIONS] <COMMAND>

Commands:
  create  Create a new bloom filter
  insert  Insert data into an existing bloom filter
  check   Checks entries against an existing bloom filter
  bench   Benchmark the bloom filter. If the bloom filter behaves in an unexpected way, the benchmark fails. Input data is read from stdin
  show    Show information about an existing bloom filter
  help    Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose      Verbose output
  -j, --jobs <JOBS>  The number of jobs to use when parallelization is possible. For write operations the original filter is copied into the memory of each job so you can expect the memory of the whole process to be N times the size of the filter [default: 2]
  -h, --help         Print help
```

### Easy filter creation

One can easily create filter directly from a bunch of data. In this case the filter capacity will
be set to the number of entries in the dataset.

```
# this creates a new filter saved in filter.pop with all entries (one per line)
# found in .txt files under the dataset directory using available CPUs (-j 0)
poppy -j 0 create -p 0.001 /path/to/output/filter.pop /path/to/dataset/*.txt
```

## Funding

The NGSOTI project is dedicated to training the next generation of Security Operation Center (SOC) operators, focusing on the human aspect of cybersecurity.
It underscores the significance of providing SOC operators with the necessary skills and open-source tools to address challenges such as detection engineering, 
incident response, and threat intelligence analysis. Involving key partners such as CIRCL, Restena, Tenzir, and the University of Luxembourg, the project aims
to establish a real operational infrastructure for practical training. This initiative integrates academic curricula with industry insights, 
offering hands-on experience in cyber ranges.

NGSOTI is co-funded under Digital Europe Programme (DEP) via the ECCC (European cybersecurity competence network and competence centre).

