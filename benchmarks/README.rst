Benchmarks
==========

Welcome to the Benchmarking section of this repository! Here, we provide configurations to help you evaluate and compare the performance of different methods.


Running a Benchmark
-------------------

To execute a benchmark, you can use the following command:

.. code-block:: bash

    stable-ssl -m --config-path <dataset> --config-name <model>

Parameters
~~~~~~~~~~

- **`<dataset>`**: Specify the dataset you are benchmarking (e.g., ``cifar10``, ``imagenette``, etc.).
- **`<model>`**: Specify the model or method you are using for the benchmark (e.g., ``simclr``, ``byol``, etc.).

Ensure that you have the appropriate configuration files set up for the dataset and model you are testing.


Contributing to Benchmarks
--------------------------

We welcome contributions! Whether you're proposing new methods, adding datasets, or improving existing configurations, your input is highly valued and will help provide accurate and reliable benchmarking.
