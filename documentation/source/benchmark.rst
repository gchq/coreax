Benchmarking Coreset Algorithms
===============================

In this benchmark, we assess the performance of four different coreset algorithms:
:class:`coreax.solvers.KernelHerding`, :class:`coreax.solvers.SteinThinning`,
:class:`coreax.solvers.RandomSample`, and :class:`coreax.solvers.RPCholesky`.
Each of these algorithms is evaluated across four different tests, providing a
comparison of their performance and applicability to various datasets.

Test 1: Benchmarking Coreset Algorithms on the MNIST Dataset
------------------------------------------------------------

The first test evaluates the performance of the coreset algorithms on the
**MNIST dataset** using a simple neural network classifier. The process follows
these steps:

1. **Dataset**: The MNIST dataset consists of 60_000 training images and 10_000
   test images. Each image is a 28x28 pixel grey-scale image of a handwritten digit.

2. **Model**: A Multi-Layer Perceptron (MLP) neural network is used for
   classification. The model consists of a single hidden layer with 64 nodes.
   Images are flattened into vectors for input.

3. **Dimensionality Reduction**: To speed up computation and reduce dimensionality, a
   density preserving **UMAP** is applied to project the 28x28 images into 16 components
   before applying any coreset algorithm.

4. **Coreset Generation**: Coresets of various sizes are generated using the
   different coreset algorithms. For :class:`coreax.solvers.KernelHerding` and
   :class:`coreax.solvers.SteinThinning`, :class:`coreax.solvers.MapReduce` is employed
   to handle large-scale data.

5. **Training**: The model is trained using the selected coresets, and accuracy is
   measured on the test set of 10_000 images.

6. **Evaluation**: Due to randomness in the coreset algorithms and training process,
   the experiment is repeated 5 times with different random seeds. The benchmark is run
   on an **Amazon g4dn.12xlarge instance** with 4 NVIDIA T4 Tensor Core GPUs, 48 vCPUs,
   and 192 GiB memory.

**Results**:
- Plots showing accuracy (with error bars) and time taken for each coreset size and
algorithm.

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
**synthetic dataset**. The dataset consists of 1_000 two-dimensional points,
generated using :func:`sklearn.datasets.make_blobs`. The process follows these steps:

1. **Dataset**: A synthetic dataset of 1_000 points is generated to test the
   quality of coreset algorithms.

2. **Coreset Generation**: Coresets of different sizes (10, 50, 100, and 200 points)
   are generated using each coreset algorithm.

3. **Evaluation Metrics**: Two metrics evaluate the quality of the generated coresets:
   - **Maximum Mean Discrepancy (MMD)**
   - **Kernel Stein Discrepancy (KSD)**

4. **Optimisation**: We optimise the weights for coresets to minimise the MMD score
   and recompute both MMD and KSD metrics.

.. list-table:: Coreset Size 10 (Original Sample Size 1_000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - 0.071501
     - 0.087505
     - 0.037938
     - 0.082897
     - 4.807713
   * - RandomSample
     - 0.275138
     - 0.106468
     - 0.080327
     - 0.082597
     - 1.285308
   * - RPCholesky
     - 0.182340
     - 0.079254
     - 0.032427
     - 0.085621
     - 1.492112
   * - SteinThinning
     - 0.186064
     - 0.078773
     - 0.087343
     - 0.085194
     - 1.867980

.. list-table:: Coreset Size 50 (Original Sample Size 1_000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - 0.011488
     - 0.080717
     - 0.003478
     - 0.079936
     - 2.864534
   * - RandomSample
     - 0.083657
     - 0.084844
     - 0.004985
     - 0.079951
     - 1.237755
   * - RPCholesky
     - 0.140705
     - 0.063904
     - 0.000732
     - 0.079960
     - 1.473465
   * - SteinThinning
     - 0.079031
     - 0.074763
     - 0.009612
     - 0.080122
     - 1.732926

.. list-table:: Coreset Size 100 (Original Sample Size 1_000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - 0.008475
     - 0.080237
     - 0.001907
     - 0.079933
     - 2.627561
   * - RandomSample
     - 0.032530
     - 0.077081
     - 0.001709
     - 0.080082
     - 1.257214
   * - RPCholesky
     - 0.075108
     - 0.073218
     - 0.000880
     - 0.080010
     - 1.500886
   * - SteinThinning
     - 0.118450
     - 0.081853
     - 0.002618
     - 0.079857
     - 2.208143

.. list-table:: Coreset Size 200 (Original Sample Size 1_000)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - KernelHerding
     - 0.004305
     - 0.080290
     - 0.000598
     - 0.080049
     - 2.638131
   * - RandomSample
     - 0.048703
     - 0.077522
     - 0.000488
     - 0.080062
     - 1.265551
   * - RPCholesky
     - 0.048309
     - 0.078012
     - 0.000846
     - 0.079993
     - 1.535722
   * - SteinThinning
     - 0.129071
     - 0.084883
     - 0.002251
     - 0.079875
     - 1.755289

**Visualisation**: The results in this table can be visualised as follows:

  .. image:: ../../examples/benchmarking_images/blobs_benchmark_results.png
     :alt: Benchmark Results for Synthetic Dataset

  **Figure 3**: Line graphs showing the performance metrics of different coreset
  algorithms on a synthetic dataset.

Test 3: Benchmarking Coreset Algorithms on Pixel Data from an Image
-------------------------------------------------------------------

This test evaluates the performance of coreset algorithms on pixel data extracted
from an input image. The process follows these steps:

1. **Input Image**: An image is loaded and downsampled to reduce resolution (a
   downsampling factor of 1 corresponds to no downsampling).

2. **Image Preprocessing**: The image is converted to grey-scale. Pixel locations
   and values are extracted for use in the coreset algorithms.

3. **Coreset Generation**: Coresets (of size 20% of the original image) are generated
   using each coreset algorithm.

4. **Visualisation**: The original image is plotted alongside coresets generated by
   each algorithm. This visual comparison helps assess how well each algorithm
   represents the image.

**Results**: The following plot visualises the pixels chosen by each coreset algorithm
with no downsampling.

  .. image:: ../../examples/benchmarking_images/david_benchmark_results.png
     :alt: Coreset Visualisation on Image

  **Figure 4**: The original image and pixels selected by each coreset algorithm
  plotted side-by-side for visual comparison.

Test 4: Benchmarking Coreset Algorithms on Frame Data from a GIF
----------------------------------------------------------------

The fourth and final test evaluates the performance of coreset algorithms on data
extracted from an input **GIF**. This test involves the following steps:

1. **Input GIF**: A GIF is loaded, and its frames are preprocessed.

2. **Dimensionality Reduction**: On each frame data, a density preserving **UMAP** is
   applied to reduce dimensionality of each frame to 25.

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
combining Kernel Herding with distributed frameworks like **MapReduce** is
recommended to ensure scalability and efficiency.

For specialised tasks, such as frame selection from GIFs (Test 4), **Stein Thinning**
demonstrated superior performance and may be the optimal choice.

Ultimately, this conclusion reflects one interpretation of the results, and readers are
encouraged to analyse the benchmarks and derive their own insights based on the specific
requirements of their tasks.
