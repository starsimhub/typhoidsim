# Typhoidsim docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the Typhoidsim docs at https://starsimhub.github.io/typhoidsim.

To build the docs, follow these steps:

1. Install Quarto: https://quarto.org/docs/get-started/

2.  Make sure the Python dependencies are installed:
    ```
    pip install -r requirements.txt
    ```

3.  Build the documents with `./render` (requires Typhoidsim to be installed as well).

4.  The built documents will be in `./_site`.

To preview the docs with live reloading, run `./preview`. To remove all temporary and build files, run `./clean_all`.
