:tocdepth: 1

SleepECG API
============

.. automodule:: sleepecg

Datasets
--------
See :ref:`Datasets <datasets>` for information about available datasets and instructions for retrieving NSRR data.

.. autosummary::
   :toctree: generated
   :nosignatures:

   download_nsrr
   download_physionet
   export_ecg_record
   read_gudb
   read_ltdb
   read_mesa
   read_mitdb
   read_shhs
   read_slpdb
   set_nsrr_token
   ECGRecord
   SleepRecord
   SleepStage


Feature extraction
------------------
Detailed information on implemented features is available :ref:`here <feature_extraction>`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   extract_features
   preprocess_rri


Heartbeat detection
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   compare_heartbeats
   detect_heartbeats
   rri_similarity


Sleep stage classification
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate
   list_classifiers
   load_classifier
   plot_hypnogram
   prepare_data_keras
   print_class_balance
   save_classifier
   stage
   SleepClassifier


Configuration
-------------
Configuration settings are explained :ref:`here <configuration>`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_config
   set_config
