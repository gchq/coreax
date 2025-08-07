Coreax (|version|)
==================

Coreax is a library for **coreset algorithms**, written in
:doc:`JAX <jax:index>` for fast execution and GPU support.


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
   benchmark

.. toctree::
   :hidden:
   :caption: Examples

   examples/herding_stein_weighted
   examples/pounce
   examples/pounce_map_reduce
   examples/david_map_reduce_weighted
   examples/analytical_greedy_kernel_points
   examples/analytical_kernel_herding
   examples/analytical_rpcholesky
   examples/analytical_stein_thinning

.. toctree::
    :maxdepth: 2
    :caption: API Reference

    coreax/approximations
    coreax/coresets
    coreax/data
    coreax/kernel
    coreax/least_squares
    coreax/metrics
    coreax/networks
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


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
