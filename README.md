![Logo](./assets/logo.svg)

[![Crates.io Version](https://img.shields.io/crates/v/poppy-filters?style=for-the-badge)](https://crates.io/crates/poppy-filters)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/hashlookup/poppy/rust.yml?style=for-the-badge)](https://github.com/hashlookup/poppy/actions)
[![docs.rs](https://img.shields.io/docsrs/poppy-filters?style=for-the-badge&logo=docs.rs&color=blue)](https://docs.rs/poppy-filters)


**Poppy** is a Rust crate offering an efficient implementation of [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter). It also includes a **command-line
utility** (also called poppy) allowing users to effortlessly create filters with their desired capacity and false positive probability.
Values can be added to the filters via standard input, facilitating the use of this tool in a pipeline workflow. 

Poppy **ensures cross-compatibility with** the bloom filter format used by [DCSO bloom software](https://github.com/DCSO/bloom) but **also provides its own** Bloom filter implementation and format.

## FAQ

### Which format to choose ?

It depends what you want to achieve. If you want to be cross compatible with DCSO tools and library, you must absolutely choose DCSO format. In any other scenario
we advice to use Poppy format (the default), as it is more robust, faster and provides room for customization. A comparison between the two formats and implementations can be found
[in this blog post](https://www.misp-project.org/2024/03/25/Poppy-a-new-bloom-filter-format-and-project.html/). By default, **library and CLI** chooses **poppy** format. If one wants
to select **DCSO format** when creating a filter **from CLI**, one has to use `poppy create --version 1`.
 
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

Please take a look at [Poppy Bindings](./python) for further details.

## Command Line Interface

### Installation

In order to install `poppy` **command line utility**, one has to run the following command: `cargo install poppy-filters`

An alternative installation is by cloning this repository and compile from source using `cargo`.

### Usage

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

Every command has its own arguments and help information. For example to get `create` command help run: `poppy create help`.

### Examples

#### Creating an empty Bloom filter

```bash
# creating a filter with a desired capacity `-c` and false positive probability `-p`
poppy create -c 1000 -p 0.001 /path/to/output/filter.pop

# showing information about the filter we just created
poppy show /path/to/output/filter.pop
```

#### Inserting data into the filter

One can insert data in the filter in two ways, **either** by reading from **stdin** or by **files**.
Reading data from stdin cannot be parallelized, so if one wants to insert a lot of data in the 
filter and speed up insertion, one has to insert from files (and setting the number of CPUs to use
with `-j` option).

```bash
# insertion from stdin
cat data-1.txt data-2.txt | poppy insert filter.pop
# we verify number of element in the filter
poppy show filter.pop

# insertion from files
poppy insert filter.pop data-1.txt data-2.txt
# we verify number of element in the filter
poppy show filter.pop

# insertion from several files in parallel
poppy -j 0 insert filter.pop data-1.txt data-2.txt
```

#### Creating and Inserting in one command

One can easily create filter directly from a bunch of data. In this case the filter capacity will
be set to the number of entries in the dataset.

```bash
# this creates a new filter saved in filter.pop with all entries (one per line)
# found in .txt files under the dataset directory using available CPUs (-j 0)
poppy -j 0 create -p 0.001 /path/to/output/filter.pop /path/to/dataset/*.txt
```

#### Checking if some data is in the filter

Check operation comes in the same variant as insertion, either from **stdin** or from **files**
(when one need to take advantage of parallelization). By default, when an entry **is inside** the 
filter **it is going to be printed out to stdout**.

```bash
# check from stdin
cat data-1.txt data-2.txt | poppy check filter.pop

# check from files
poppy check filter.pop data-1.txt data-2.txt

# check from several files in parallel
poppy -j 0 check filter.pop data-1.txt data-2.txt
```

#### Benchmarking filter

Benchmarking a filter is an important step as it allow you to make sure that what you get is what
you expected, in terms of false positive probability. The benchmark needs to take data already 
inserted in the filter, it will then randomly mutate entries and check them against the filter.

```bash
# run a benchmark against data known to be in the filter
cat data-1.txt data-2.txt | poppy bench filter.pop
```

## Funding

The NGSOTI project is dedicated to training the next generation of Security Operation Center (SOC) operators, focusing on the human aspect of cybersecurity.
It underscores the significance of providing SOC operators with the necessary skills and open-source tools to address challenges such as detection engineering, 
incident response, and threat intelligence analysis. Involving key partners such as CIRCL, Restena, Tenzir, and the University of Luxembourg, the project aims
to establish a real operational infrastructure for practical training. This initiative integrates academic curricula with industry insights, 
offering hands-on experience in cyber ranges.

NGSOTI is co-funded under Digital Europe Programme (DEP) via the ECCC (European cybersecurity competence network and competence centre).

