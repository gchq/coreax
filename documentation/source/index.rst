Coreax (|version|)
==================

Coreax is a library for **coreset algorithms**, written in
:doc:`JAX <jax:quickstart>` for fast execution and GPU support.


Setup
-----

Before installing Coreax, make sure JAX is installed. Be sure to install the preferred
version of JAX for your system.

1. :doc:`Install JAX <jax:installation>` noting that there are (currently) different
   setup paths for CPU and GPU use.
2. Install Coreax:

.. code:: shell

   $ python3 -m pip install coreax

3. Optionally, install additional dependencies required to run the examples:

.. code:: shell

   $ python3 -m pip install coreax[test]

Should the installation fail, try again using stable pinned package versions. Note that
these versions may be rather outdated, although we endeavour to avoid versions with
known vulnerabilities. To install Coreax:

.. code::

    $ python3 -m pip install --no-dependencies -r requirements.txt

To run the examples, use :code:`requirements-test.txt` instead.

Contents
--------

.. toctree::
   :maxdepth: 2

   coresets
   quickstart
   faq

.. toctree::
   :hidden:
   :caption: Examples

   examples/herding_stein_weighted
   examples/pounce
   examples/pounce_map_reduce
   examples/david_map_reduce_weighted

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    coreax/approximations
    coreax/coresets
    coreax/coresubset
    coreax/data
    coreax/kernel
    coreax/metrics
    coreax/networks
    coreax/reduction
    coreax/refine
    coreax/score_matching
    coreax/solvers
    coreax/utils
    coreax/weights


Bibliography
------------

.. bibliography::


Release Cycle
-------------

We anticipate two release types: feature releases and security releases. Security
releases will be issued as needed in accordance with the
`security policy <https://github.com/gchq/coreax/security/policy>`_. Feature releases
will be issued as appropriate, dependent on the feature pipeline and development
priorities.


Coming Soon
-----------

Some features coming soon include:

* Coordinate bootstrapping for high-dimensional data.
* Other coreset-style algorithms, including kernel thinning and recombination, as means
  to reducing a large dataset whilst maintaining properties of the underlying
  distribution.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
