To launch a benchmark run locally, you can use the following command:

.. code-block:: bash

    python run.py -m mode=one_gpu_local experiment=simclr_cifar10 # choose any config in config/experiment/


To launch a benchmark run on slurm, you can use:

.. code-block:: bash

    python run.py -m mode=one_gpu_slurm experiment=simclr_cifar10 # choose any config in config/experiment/
