SleepECG API Reference
======================

.. automodule:: sleepecg

This page gives an overview of all public SleepECG functions and classes.

Datasets
--------
See :ref:`Datasets <datasets>` for information about the available datasets and instructions for retrieving NSRR data.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   io.download_nsrr
   io.download_physionet
   io.read_gudb
   io.read_ltdb
   io.read_mesa
   io.read_mitdb
   io.read_shhs
   io.read_slpdb
   io.set_nsrr_token


Feature extraction
------------------
Detailed information on the implemented features is available :ref:`here <feature_extraction>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   extract_features
   preprocess_rri


Heartbeat detection
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compare_heartbeats
   detect_heartbeats
   rri_similarity


Configuration
-------------
Possible configuration settings are explained :ref:`here <configuration>`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_config
   set_config
