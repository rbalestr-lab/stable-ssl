stable_ssl.data
=================
.. module:: stable_ssl.data
.. currentmodule:: stable_ssl.data

The data module provides comprehensive tools for dataset handling, transforms, sampling, and data loading in self-supervised learning contexts.

Core Components
---------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   DataModule
   Collator

Samplers
---------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   RepeatedRandomSampler
   SupervisedBatchSampler
   RandomBatchSampler

Dataset Classes
---------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   GMM
   Subset
   FromTorchDataset
   MinariStepsDataset
   HFDataset

Noise Models
------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   Categorical
   ExponentialMixtureNoiseModel
   ExponentialNormalNoiseModel

Utility Functions
----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myfunc_template.rst

   fold_views
   random_split
   download
   bulk_download

Modules
-------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   transforms
   dataset
