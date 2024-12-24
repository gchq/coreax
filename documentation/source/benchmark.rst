Benchmarking Coreset Algorithms
===============================

In this benchmark, we assess the performance of four different coreset algorithms:
:class:`~coreax.solvers.KernelHerding`, :class:`~coreax.solvers.SteinThinning`,
:class:`~coreax.solvers.RandomSample`, and :class:`~coreax.solvers.RPCholesky`.
Each of these algorithms is evaluated across four different tests, providing a
comparison of their performance and applicability to various datasets.

Test 1: Benchmarking Coreset Algorithms on the MNIST Dataset
------------------------------------------------------------

The first test evaluates the performance of the coreset algorithms on the
**MNIST dataset** using a simple neural network classifier. The process follows
these steps:

1. **Dataset**: The MNIST dataset consists of 60,000 training images and 10,000
   test images. Each image is a 28x28 pixel grey-scale image of a handwritten digit.

2. **Model**: A Multi-Layer Perceptron (MLP) neural network is used for
   classification. The model consists of a single hidden layer with 64 nodes.
   Images are flattened into vectors for input.

3. **Dimensionality Reduction**: To speed up computation and reduce dimensionality, a
   density preserving :class:`~umap.umap_.UMAP` is applied to project the 28x28 images into 16 components
   before applying any coreset algorithm.

4. **Coreset Generation**: Coresets of various sizes are generated using the
   different coreset algorithms. For :class:`~coreax.solvers.KernelHerding` and
   :class:`~coreax.solvers.SteinThinning`, :class:`~coreax.solvers.MapReduce` is
   employed to handle large-scale data.

5. **Training**: The model is trained using the selected coresets, and accuracy is
   measured on the test set of 10,000 images.

6. **Evaluation**: Due to randomness in the coreset algorithms and training process,
   the experiment is repeated 5 times with different random seeds. The benchmark is run
   on an **Amazon g4dn.12xlarge instance** with 4 NVIDIA T4 Tensor Core GPUs, 48 vCPUs,
   and 192 GiB memory.

**Results**:
The accuracy of the MLP classifier when trained using the full MNIST dataset
(60,000 training images) was 97.31%, serving as a baseline for evaluating the performance
of the coreset algorithms.

- Plots showing the accuracy (with error bars) of the MLP's predictions on the test set,
  along with the time taken generate coreset for each coreset size and algorithm.

  .. image:: ../../examples/benchmarking_images/mnist_benchmark_accuracy.png
     :alt: Benchmark Results for MNIST Coreset Algorithms

  **Figure 1**: Accuracy of coreset algorithms on the MNIST dataset. Bar heights
  represent the average accuracy. Error bars represent the min-max range for accuracy
  for each coreset size across 5 runs.

  .. image:: ../../examples/benchmarking_images/mnist_benchmark_time_taken.png
     :alt: Time Taken Benchmark Results for MNIST Coreset Algorithms

  **Figure 2**: Time taken to generate coreset for each coreset algorithm. Bar heights
  represent the average time taken. Error bars represent the min-max range for each
  coreset size across 5 runs.

Test 2: Benchmarking Coreset Algorithms on a Synthetic Dataset
--------------------------------------------------------------

In this second test, we evaluate the performance of the coreset algorithms on a
**synthetic dataset**. The dataset consists of 1,000 points in two-dimensional space,
generated using :func:`sklearn.datasets.make_blobs`. The process follows these steps:

1. **Dataset**: A synthetic dataset of 1,000 points is generated to test the
   quality of coreset algorithms.

2. **Coreset Generation**: Coresets of different sizes (10, 50, 100, and 200 points)
   are generated using each coreset algorithm.

3. **Evaluation Metrics**: Two metrics evaluate the quality of the generated coresets:
   :class:`~coreax.metrics.MMD` and :class:`~coreax.metrics.KSD`.

4. **Optimisation**: We optimise the weights for coresets to minimise the MMD score
   and recompute both MMD and KSD metrics. These entire process is repeated 5 times with
   5 random seeds and the metrics are averaged.

**Results**:
The tables below show the performance metrics (Unweighted MMD, Unweighted KSD,
Weighted MMD, Weighted KSD, and Time) for each coreset algorithm and each coreset size.
For each metric and coreset size, the best performance score is highlighted in bold.

.. list-table:: Coreset Size 10 (Original Sample Size 1,000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - **0.071504**
     - 0.087505
     - 0.037931
     - 0.082903
     - 5.884511
   * - RandomSample
     - 0.275138
     - 0.106468
     - 0.080327
     - **0.082597**
     - **2.705248**
   * - RPCholesky
     - 0.182342
     - 0.079254
     - **0.032423**
     - 0.085621
     - 3.177700
   * - SteinThinning
     - 0.186064
     - **0.078773**
     - 0.087347
     - 0.085194
     - 4.450125

.. list-table:: Coreset Size 50 (Original Sample Size 1,000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - **0.016602**
     - 0.080800
     - 0.003821
     - **0.079875**
     - 5.309067
   * - RandomSample
     - 0.083658
     - 0.084844
     - 0.005009
     - 0.079948
     - **2.636160**
   * - RPCholesky
     - 0.133182
     - **0.061976**
     - **0.001859**
     - 0.079935
     - 3.201798
   * - SteinThinning
     - 0.079028
     - 0.074763
     - 0.009652
     - 0.080119
     - 3.735810

.. list-table:: Coreset Size 100 (Original Sample Size 1,000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - **0.007747**
     - 0.080280
     - **0.001582**
     - 0.080024
     - 5.425807
   * - RandomSample
     - 0.032532
     - 0.077081
     - 0.001638
     - 0.080073
     - **3.009871**
   * - RPCholesky
     - 0.069909
     - **0.072023**
     - 0.000977
     - 0.079995
     - 3.497632
   * - SteinThinning
     - 0.118452
     - 0.081853
     - 0.002652
     - **0.079836**
     - 3.766622

.. list-table:: Coreset Size 200 (Original Sample Size 1,000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - **0.003937**
     - 0.079932
     - 0.001064
     - 0.080012
     - 5.786333
   * - RandomSample
     - 0.048701
     - 0.077522
     - 0.000913
     - 0.080059
     - **2.964436**
   * - RPCholesky
     - 0.052085
     - **0.075708**
     - **0.000772**
     - 0.080050
     - 3.722556
   * - SteinThinning
     - 0.129073
     - 0.084883
     - 0.002329
     - **0.079847**
     - 4.004353


**Visualisation**: The results in this table can be visualised as follows:

  .. image:: ../../examples/benchmarking_images/blobs_benchmark_results.png
     :alt: Benchmark Results for Synthetic Dataset

  **Figure 3**: Line graphs depicting the average performance metrics across 5 runs of
  each coreset algorithm on a synthetic dataset.

Test 3: Benchmarking Coreset Algorithms on Pixel Data from an Image
-------------------------------------------------------------------

This test evaluates the performance of coreset algorithms on pixel data extracted
from an input image. The process follows these steps:

1. **Image Preprocessing**: An image is loaded and converted to grey-scale. Pixel
   locations and values are extracted for use in the coreset algorithms.

2. **Coreset Generation**: Coresets (of size 20% of the original image) are generated
   using each coreset algorithm.

3. **Visualisation**: The original image is plotted alongside coresets generated by
   each algorithm. This visual comparison helps assess how well each algorithm
   represents the image.

**Results**: The following plot visualises the pixels chosen by each coreset algorithm.

  .. image:: ../../examples/benchmarking_images/david_benchmark_results.png
     :alt: Coreset Visualisation on Image

  **Figure 4**: The original image and pixels selected by each coreset algorithm
  plotted side-by-side for visual comparison.

Test 4: Benchmarking Coreset Algorithms on Frame Data from a GIF
----------------------------------------------------------------

The fourth and final test evaluates the performance of coreset algorithms on data
extracted from an input **GIF**. This test involves the following steps:

1. **Input GIF**: A GIF is loaded, and its frames are preprocessed.

2. **Dimensionality Reduction**: On each frame data, a density preserving
   :class:`~umap.umap_.UMAP` is applied to reduce dimensionality of each frame to 25.

3. **Coreset Generation**: Coresets are generated using each coreset algorithm, and
   selected frames are saved as new GIFs.


**Result**:
- GIF files showing the selected frames for each coreset algorithm.

  .. image:: ../../examples/pounce/pounce.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 1**: Original gif file.

  .. image:: ../../examples/benchmarking_images/RandomSample_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 2**: Frames selected by Random Sample.

  .. image:: ../../examples/benchmarking_images/SteinThinning_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 3**: Frames selected by Stein Thinning.

  .. image:: ../../examples/benchmarking_images/RPCholesky_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 4**: Frames selected by RP Cholesky.

  .. image:: ../../examples/benchmarking_images/KernelHerding_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 5**: Frames selected by Kernel Herding.

  .. image:: ../../examples/benchmarking_images/pounce_frames.png
     :alt: Coreset Visualisation on GIF Frames

  **Figure 5**:Frames chosen by each each coreset algorithm with action frames (the
  frames in which pounce action takes place) highlighted in red.

Conclusion
----------

In this benchmark, we evaluated four coreset algorithms across various datasets and
tasks, including image classification, synthetic datasets, and pixel/frame data
processing. Based on the results, **Kernel Herding** emerges as the preferred choice
for most tasks due to its consistent performance. For larger datasets,
combining Kernel Herding with distributed frameworks like **Map Reduce** is
recommended to ensure scalability and efficiency.

For specialised tasks, such as frame selection from GIFs (Test 4), **Stein Thinning**
demonstrated superior performance and may be the optimal choice.

Ultimately, this conclusion reflects one interpretation of the results, and readers are
encouraged to analyse the benchmarks and derive their own insights based on the specific
requirements of their tasks.
