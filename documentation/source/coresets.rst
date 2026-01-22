About Coresets
==============

For :math:`n` points in :math:`d` dimensions, a coreset algorithm takes an
:math:`n \times d` data set and reduces it to :math:`m \ll n` points whilst attempting
to preserve the statistical properties of the full data set. The algorithm maintains the
dimension of the original data set. Thus the :math:`m` points, referred to as the
**coreset**, are also :math:`d`-dimensional.

The :math:`m` points need not be in the original data set. We refer to the special case
where all selected points are in the original data set as a **coresubset**.

Some algorithms return the :math:`m` points with weights, so that importance can be
attributed to each point in the coreset. The weights, :math:`w_i` for
:math:`i=1,\dots,m`, are often chosen from the simplex. In this case, they are
non-negative and sum to 1: :math:`w_i > 0` and :math:`\sum_{i} w_i = 1`.


Quick Example
-------------

Consider :math:`n=10,000` points drawn from six 2D multivariate Gaussian distributions.
We wish to reduce this to only 100 points, whilst maintaining underlying statistical
properties. We achieve this by generating a coreset, setting :math:`m=100`. We plot the
underlying data (blue) as-well as the coreset points (red), which are plotted
sequentially based on the order the algorithm selects them in. The coreset points are
weighted (size of point) to optimally reconstruct the underlying distribution. Run
:mod:`examples.herding_stein_weighted` to replicate.

We compare the coreset to the full original dataset by calculating the maximum mean
discrepancy `MMD <https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions>`__.
This key property is an integral probability metric, measuring the distance between the
empirical distributions of the full dataset and the coreset. A good coreset algorithm
produces a coreset that has significantly smaller MMD than randomly sampling the same
number of points from the original data, as is the case in the example below.

.. list-table::
    :header-rows: 1
    :align: left

    * - Kernel Herding
      - Random Sample
    * - .. image:: ../../examples/data/coreset_seq/coreset_seq.gif
      - .. image:: ../../examples/data/random_seq/random_seq.gif


Example applications
--------------------


Choosing pixels from an image
_____________________________


In the example below, we reduce the original 180x215 pixel image (38,700 pixels in
total) to a coreset approximately 20% of this size. Run
:mod:`examples.david_map_reduce_weighted` to replicate.

.. figure:: ../../examples/data/david_coreset.png

    (Left) original image.
    (Centre) 8,000 coreset points chosen using Stein kernel herding, with point size a
    function of weight.
    (Right) 8,000 points chosen randomly.


Video event detection
_____________________

Here we identify representative frames such that most of the useful information in a
video is preserved. Run :mod:`examples.pounce` to replicate.

.. list-table::
    :header-rows: 1
    :align: left

    * - Original
      - Coreset
    * - .. image:: ../../examples/pounce/pounce.gif
      - .. image:: ../../examples/pounce/pounce_coreset.gif
