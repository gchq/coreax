Benchmarking Coreset Algorithms
===============================

In this benchmark, we assess the performance of different coreset algorithms:
:class:`~coreax.solvers.KernelHerding`, :class:`~coreax.solvers.SteinThinning`,
:class:`~coreax.solvers.RandomSample`, :class:`~coreax.solvers.RPCholesky` and
:class:`~coreax.solvers.KernelThinning`, :class:`~coreax.solvers.CompressPlusPlus`,
:class:`~coreax.solvers.IterativeKernelHerding`. Each of these algorithms is evaluated
across four different tests, providing a comparison of their performance and
applicability to various datasets.

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
   density preserving :class:`~umap.umap_.UMAP` is applied to project the 28x28 images
   into 16 components before applying any coreset algorithm.

4. **Coreset Generation**: Coresets of various sizes are generated using the
   different coreset algorithms. For :class:`~coreax.solvers.KernelHerding`,
   :class:`~coreax.solvers.SteinThinning`, :class:`~coreax.solvers.KernelThinning`, and
   :class:`~coreax.solvers.IterativeKernelHerding`,
   :class:`~coreax.solvers.MapReduce` is employed to handle large-scale data.

5. **Training**: The model is trained using the selected coresets, and accuracy is
   measured on the test set of 10,000 images.

6. **Evaluation**: Due to randomness in the coreset algorithms and training process,
   the experiment is repeated 4 times with different random seeds. The benchmark is run
   on an **Amazon Web Services EC2 g4dn.12xlarge instance** with 4 NVIDIA T4 Tensor Core
   GPUs, 48 vCPUs, and 192 GiB memory.

Impact of UMAP and MapReduce on Coreset Performance
---------------------------------------------------

Since **UMAP** was used to reduce dimensionality and **MapReduce** was required to
handle large-scale data, the coreset algorithms were not applied directly to the raw
MNIST images. While these preprocessing steps improved efficiency, they may have
**impacted the full potential of the coreset methods**.

**Results**:
The accuracy of the MLP classifier when trained using the full MNIST dataset
(60,000 training images) was 97.31%, serving as a baseline for evaluating the
performance of the coreset algorithms.



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
**synthetic dataset**. The dataset consists of 1,024 points in two-dimensional space,
generated using :func:`sklearn.datasets.make_blobs`. The process follows these steps:

1. **Dataset**: A synthetic dataset of 1,024 points is generated to test the
   quality of coreset algorithms.

2. **Coreset Generation**: Coresets of different sizes (10, 50, 100, and 200 points)
   are generated using each coreset algorithm.

3. **Evaluation Metrics**: Two metrics evaluate the quality of the generated coresets:
   :class:`~coreax.metrics.MMD` and :class:`~coreax.metrics.KSD`.

4. **Optimisation**: We optimise the weights for coresets to minimise the MMD score
   and recompute both MMD and KSD metrics. These entire process is repeated 5 times with
   a different random seed each time and the metrics are averaged.

**Results**:
The tables below show the performance metrics (Unweighted MMD, Unweighted KSD,
Weighted MMD, Weighted KSD, and Time) for each coreset algorithm and each coreset size.
For each metric and coreset size, the best performance score is highlighted in bold.

.. list-table:: Coreset Size 25 (Original Sample Size 1,024)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - Kernel Herding
     - 0.024273
     - 0.072547
     - 0.008471
     - 0.072267
     - 3.859628
   * - RandomSample
     - 0.125471
     - 0.087859
     - 0.037686
     - 0.074856
     - **2.659764**
   * - RP Cholesky
     - 0.140715
     - **0.059376**
     - **0.003011**
     - **0.071982**
     - 3.312633
   * - Stein Thinning
     - 0.165692
     - 0.073476
     - 0.033367
     - 0.073952
     - 3.714297
   * - Kernel Thinning
     - 0.014093
     - 0.071987
     - 0.005737
     - 0.072614
     - 23.659113
   * - Compress++
     - 0.010929
     - 0.072254
     - 0.005783
     - 0.072447
     - 15.278997
   * - Probabilistic Iterative Herding
     - 0.017470
     - 0.074181
     - 0.007226
     - 0.072694
     - 4.330906
   * - IIterative Herding
     - **0.006842**
     - 0.072133
     - 0.004978
     - 0.072212
     - 3.399839

.. list-table:: Coreset Size 50 (Original Sample Size 1,024)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - Kernel Herding
     - 0.014011
     - 0.072273
     - 0.003191
     - 0.072094
     - 3.417109
   * - RandomSample
     - 0.100558
     - 0.080291
     - 0.005518
     - 0.072549
     - **2.575190**
   * - RP Cholesky
     - 0.136605
     - **0.055552**
     - **0.001971**
     - 0.072116
     - 3.227958
   * - Stein Thinning
     - 0.152293
     - 0.073183
     - 0.017996
     - **0.071682**
     - 4.056369
   * - Kernel Thinning
     - 0.006482
     - 0.071823
     - 0.002541
     - 0.072183
     - 12.507483
   * - Compress++
     - 0.006065
     - 0.071981
     - 0.002633
     - 0.072257
     - 9.339439
   * - Probabilistic Iterative Herding
     - 0.010031
     - 0.072707
     - 0.002906
     - 0.072432
     - 4.279948
   * - IIterative Herding
     - **0.003546**
     - 0.072107
     - 0.002555
     - 0.072203
     - 3.291645

.. list-table:: Coreset Size 100 (Original Sample Size 1,024)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - Kernel Herding
     - 0.007909
     - 0.071763
     - 0.001859
     - 0.072205
     - 3.583433
   * - RandomSample
     - 0.067373
     - 0.077506
     - 0.001673
     - 0.072329
     - **2.631034**
   * - RP Cholesky
     - 0.091372
     - **0.059889**
     - **0.001174**
     - 0.072281
     - 3.426726
   * - Stein Thinning
     - 0.102536
     - 0.074250
     - 0.007770
     - **0.071809**
     - 3.673147
   * - Kernel Thinning
     - 0.002811
     - 0.072218
     - 0.001414
     - 0.072286
     - 7.878599
   * - Compress++
     - 0.003343
     - 0.072287
     - 0.001486
     - 0.072283
     - 6.930467
   * - Probabilistic Iterative Herding
     - 0.006254
     - 0.072408
     - 0.001649
     - 0.072289
     - 4.381068
   * - IIterative Herding
     - **0.002130**
     - 0.072142
     - 0.001373
     - 0.072248
     - 3.502385

.. list-table:: Coreset Size 200 (Original Sample Size 1,024)
   :header-rows: 1
   :widths: 20 15 15 15 15 15

   * - Method
     - Unweighted_MMD
     - Unweighted_KSD
     - Weighted_MMD
     - Weighted_KSD
     - Time
   * - Kernel Herding
     - 0.004259
     - 0.072017
     - 0.001173
     - 0.072242
     - 3.810858
   * - RandomSample
     - 0.031644
     - 0.074061
     - 0.001005
     - 0.072271
     - **2.787691**
   * - RP Cholesky
     - 0.052786
     - **0.065218**
     - **0.000784**
     - 0.072269
     - 3.545290
   * - Stein Thinning
     - 0.098395
     - 0.078290
     - 0.004569
     - **0.071896**
     - 3.910901
   * - Kernel Thinning
     - **0.001175**
     - 0.072160
     - 0.000933
     - 0.072273
     - 5.720256
   * - Compress++
     - 0.001336
     - 0.072193
     - 0.000788
     - 0.072228
     - 6.081252
   * - Probabilistic Iterative Herding
     - 0.005056
     - 0.072054
     - 0.000852
     - 0.072287
     - 5.043387
   * - IIterative Herding
     - 0.001346
     - 0.072169
     - 0.001020
     - 0.072241
     - 3.699600



**Visualisation**: The results in this table can be visualised as follows:

  .. image:: ../../examples/benchmarking_images/blobs_1.png
     :alt: Benchmark Results for Synthetic Dataset

  .. image:: ../../examples/benchmarking_images/blobs_2.png
     :alt: Benchmark Results for Synthetic Dataset

  .. image:: ../../examples/benchmarking_images/blobs_3.png
     :alt: Benchmark Results for Synthetic Dataset

  .. image:: ../../examples/benchmarking_images/blobs_4.png
     :alt: Benchmark Results for Synthetic Dataset

  .. image:: ../../examples/benchmarking_images/blobs_5.png
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

Test 4: Selecting Key Frames from a Video Data
----------------------------------------------

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

  **Gif 3**: Frames selected by Stein thinning.

  .. image:: ../../examples/benchmarking_images/RPCholesky_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 4**: Frames selected by RP Cholesky.

  .. image:: ../../examples/benchmarking_images/KernelHerding_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 5**: Frames selected by kernel herding.

  .. image:: ../../examples/benchmarking_images/KernelThinning_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 6**: Frames selected by kernel thinning.

  .. image:: ../../examples/benchmarking_images/CompressPlusPlus_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 7**: Frames selected by compress++.

  .. image:: ../../examples/benchmarking_images/ProbabilisticIterativeHerding_coreset.gif
     :alt: Coreset Visualisation on GIF Frames

  **Gif 8**: Frames selected by probabilistic iterative kernel herding.

Conclusion
----------
This benchmark evaluated four coreset algorithms across various tasks, including image
classification and frame selection. *Iterative kernel herding* and *kernel thinning*
emerged as the top performers, offering strong and consistent results. For large-scale
datasets, *compress++* and *map reduce* provide efficient scalability.
