stable_ssl.callbacks
====================
.. module:: stable_ssl.callbacks
.. currentmodule:: stable_ssl.callbacks

The callbacks module provides various monitoring and evaluation tools for self-supervised learning training.

Online Monitoring
----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   OnlineProbe
   OnlineKNN
   OnlineWriter
   RankMe
   LiDAR

Training Utilities
-----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   EarlyStopping
   TrainerInfo
   LoggingCallback
   ModuleSummary

Model Persistence
-----------------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   SklearnCheckpoint

Evaluation
----------

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   ImageRetrieval
