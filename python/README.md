[![PyPI - Version](https://img.shields.io/pypi/v/poppy-py?style=for-the-badge)](https://pypi.org/project/poppy-py/)

This is a package providing python bindings for [poppy](https://github.com/hashlookup/poppy), a bloom filter library written in Rust.

# Installation from git

The following example assumes you are using `virtualenvwrapper`, it is trivial to adapt to make it work with your preferred python virtual environment.

## With virtual environment

```bash
mkvirtualenv poppy
git clone https://github.com/hashlookup/poppy
cd poppy/poppy-py
# install maturin (tool shipped with PyO3)
pip install maturin
# installs current bindings into the current virtual env
maturin develop --release
python -c "import poppy; help(poppy)"
```

## With maturin already installed

One can install `maturin` outside a virtual environment.
For example by doing:

```bash
pipx install maturin
```

Please look at the [PyO3 documentation](https://pyo3.rs/main/getting-started.html?#building) to find the different ways to install maturin.

Once this is done, the dependency can be simply installed with 

```bash
git clone https://github.com/hashlookup/poppy
cd poppy/poppy-py
# installs current bindings into the current virtual env
maturin develop --release
```
