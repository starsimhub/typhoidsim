Typhoidsim
=======

Typhodisim implements an agent-based disease model of Typhoid fever (caused by S. Typhi).
This package is part of the `Starsim <https://github.com/starsimhub>`_ family. Currently, typhoidsim needs **v1.0.3** of the Starsim framework to run.


Installation
------------

`typhoidsim` can be installed

There are a few version of typhoidsim:
  - 1. legacy `typhoidsim v0.11.7 <https://github.com/starsimhub/typhoidsim/tree/legacy/main-v0.11.7-starsim-1.0.3>`_ , which exclusively works with `starsim 1.0.3 <https://pypi.org/project/starsim/1.0.3/>`_.
  - 2. legacy `typhoidsim v0.26.9 <https://github.com/starsimhub/typhoidsim/tree/legacy/main-v0.26.9-starsim-2.2.0>`_ , which exclusively works with `starsim 2.2.0 <https://pypi.org/project/starsim/2.2.0/>`_.
  - 3. typhoidsim from the `main <https://github.com/starsimhub/typhoidsim>`_ branch, which works with the bleeding edge `starsim rc2.3 <https://github.com/starsimhub/starsim/tree/rc2.3_calib_betafix>`_.

To install the legacy version, that runs with starsim 1.0.3, first clone the specific branch
  - ``git clone --single-branch --branch legacy/main-v0.11.7-starsim-1.0.3 origin``


To install the legacy version, that runs with starsim 2.2.0, first clone the specific branch
  - ``git clone --single-branch --branch legacy/main-v0.26.9-starsim-2.2.0 origin``

Make sure you're inside the typhoidsim repository:
   - ``cd typhoidsim``

And then run
   - ``pip install -e .`` (don't forget the dot at the end!).


!!! Install the correct version of starsim for the main branch, current as of 2025-01-07
   - ``pip install git+https://github.com/starsimhub/starsim.git@rc2.3_calib_betafix``



More detailed instructions can be found `here <https://github.com/starsimhub/typhoidsim/issues/87>`_.

Usage and documentation
-----------------------

Documentation is available at <PLACEHOLDER>.
`Tutorials <https://github.com/starsimhub/typhoidsim/tree/main/docs/tutorials>`_ (ipython notebooks) are available.


Contributing
------------

If you wish to contribute, please see the code of conduct and contributing documents.


Disclaimer
----------

The code in this repository was developed by IDM. It wil be made public under
the `MIT <https://github.com/starsimhub/typhoidsim/blob/main/LICENSE>`_ License. We make no representations that the code works as intended or
that we will provide support, address issues that are found, or accept pull requests.
You are welcome to create your own fork and modify the code to suit your own
modeling needs as permitted under the MIT License.
