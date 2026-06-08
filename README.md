# Typhoidsim

Typhoidsim implements an agent-based disease model of typhoid fever (caused by S. Typhi). This package is part of the [Starsim](https://github.com/starsimhub) family. Currently, Typhoidsim needs **v2.3.0** of the Starsim framework to run.


## Installation

Make sure you're inside the Typhoidsim repository:

```bash
cd typhoidsim
```

And then run (don't forget the dot at the end!):

```bash
pip install -e .
```

There are a few versions of Typhoidsim:

1. Legacy [typhoidsim v0.11.7](https://github.com/starsimhub/typhoidsim/tree/legacy/main-v0.11.7-starsim-1.0.3), which exclusively works with [starsim 1.0.3](https://pypi.org/project/starsim/1.0.3/).
2. Legacy [typhoidsim v0.26.9](https://github.com/starsimhub/typhoidsim/tree/legacy/main-v0.26.9-starsim-2.2.0), which exclusively works with [starsim 2.2.0](https://pypi.org/project/starsim/2.2.0/).
3. Legacy [typhoidsim v0.30.6](https://github.com/starsimhub/typhoidsim/tree/legacy/main-v0.30.6-starsim-2.3.0), which exclusively works with [starsim 2.3.0](https://pypi.org/project/starsim/).
4. Typhoidsim from the [main](https://github.com/starsimhub/typhoidsim) branch, which works with [starsim 2.3.0](https://pypi.org/project/starsim/).

To install the legacy version that runs with starsim 1.0.3, first clone the specific branch:

```bash
git clone --single-branch --branch legacy/main-v0.11.7-starsim-1.0.3 origin
```

To install the legacy version that runs with starsim 2.2.0, first clone the specific branch:

```bash
git clone --single-branch --branch legacy/main-v0.26.9-starsim-2.2.0 origin
```

More detailed instructions can be found [here](https://github.com/starsimhub/typhoidsim/issues/87).


## Usage and documentation

Documentation is available at https://starsimhub.github.io/typhoidsim. [Tutorials](https://github.com/starsimhub/typhoidsim/tree/main/docs/tutorials) (Jupyter notebooks) are also available.


## Contributing

If you wish to contribute, please see the code of conduct and contributing documents.


## Disclaimer

The code in this repository was developed by [IDM](https://idmod.org) and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
