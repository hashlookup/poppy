
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/hashlookup/poppy/rust.yml?style=for-the-badge)

# Building

## Regular building

```bash
cargo build --release --bins
```

## Building with MUSL (static binary)

```bash
cargo build --release --target=x86_64-unknown-linux-musl --bins
```

# Usage

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
