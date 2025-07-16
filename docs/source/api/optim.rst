stable_ssl.optim
================
.. module:: stable_ssl.optim
.. currentmodule:: stable_ssl.optim

The optim module provides custom optimizers and learning rate schedulers for self-supervised learning.

Optimizers
----------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   LARS

Learning Rate Schedulers
------------------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   CosineDecayer
   LinearWarmup
   LinearWarmupCosineAnnealing
   LinearWarmupCyclicAnnealing
   LinearWarmupThreeStepsAnnealing
