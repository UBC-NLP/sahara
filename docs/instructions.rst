=======================
Submission Guidelines
=======================

Follow these instructions to successfully submit your generations to the SAHARA benchmark and appear on the leaderboard.

----

Understanding the Scoring
-------------------------

* **SAHARA Score:** We calculate scores for each task based on their specific metrics. These scores are then averaged to produce the final SAHARA score.
* **Main Leaderboard:** To be ranked on the main leaderboard, your submission must include generation files for **all task-clusters**.
* **Task Leaderboards:** Submissions for individual tasks are ranked separately on their corresponding leaderboards using the task-specific metric.
* **Submission Status:** Submissions are private by default but can be made public to appear on the leaderboards. Please note that all submissions are permanent and cannot be deleted.

----

File Formatting Requirements
----------------------------

Correctly formatting your generation files is crucial for a successful submission.

* **File Naming:** Name each generation file using the format ``{task_identifier}_generation.json``.

    .. note::
        Example: The generation file for the **sentiment** task must be named **``sentiment_generation.json``**. You can find all task identifiers on the Tasks page.

* **File Content:** Each row in the ``.jsonl`` file must be a JSON object representing the generation for a single data point. The order must match the original test set. The JSON object must have the following structure:

    .. code-block:: json

        {"lang_code": "bam", "generation": "B", "example_id": "0"}

    * ``lang_code``: The language code from the original test set for that data point.
    * ``generation``: Your model's generated output for the task.
    * ``example_id``: The index of the data point, starting from 0.

* **Compression:** Compress all your ``*.json`` generation files into a single ``.zip`` archive.

    .. important::
        Zip the results files (``*.jsonl``) directly. Do not place them in a folder and zip the folder.

    .. image:: https://africa.dlnlp.ai/sahara/img/compress_your_results.png
       :width: 50%
       :align: center

=================================
Step-by-Step Submission Process
=================================

Get Dataset Access
------------------

To obtain access to the Sahara test set dataset, log in to your Hugging Face account and request access to the `SAHARA Benchmark dataset <https://huggingface.co/datasets/UBC-NLP/sahara_benchmark>`_ by filling out the form as shown below.

.. image:: https://africa.dlnlp.ai/sahara/img/dataset_access.png
   :width: 70%
   :align: center

Create a Profile
----------------

Register for an account on the `official SAHARA website <https://africa.dlnlp.ai/sahara/login>`_.

.. image:: https://africa.dlnlp.ai/sahara/img/login.png
   :width: 70%
   :align: center

Generate Files
--------------

Evaluate your model on the SAHARA test set and generate your files according to the formatting rules above. You can use our `official evaluation script <https://github.com/UBC-NLP/sahara>`_ or your own.

Submit Your Results
-------------------

* Compress your generation files into a ``.zip`` file and upload it through the submission portal on your profile (See file formatting requirement above).
* Fill out the submission form as shown below.

.. image:: https://africa.dlnlp.ai/sahara/img/submit_new_results.png
   :width: 70%
   :align: center

* Once the submission is processed, you will receive an email notification with your scores and a link to view your results.

.. image:: https://africa.dlnlp.ai/sahara/img/submission_info.png
   :width: 70%
   :align: center

View Your Scores
----------------

Once processed, you can explore your private results on `your profile page <https://africa.dlnlp.ai/sahara/profile>`_.

.. image:: https://africa.dlnlp.ai/sahara/img/private_results.png
   :width: 70%
   :align: center

How to Make Your Submission Public
----------------------------------

All submissions are private by default. To make a submission public and have it appear on the `main leaderboard </leaderboards>`_, you must meet the following requirements:

1.  The submission must include results for all **18 tasks** required to obtain a Sahara Score.
2.  The model must be publicly available on **Hugging Face**.
3.  After meeting the first two conditions, please `email us <mailto:a.elmadany@ubc.ca>`_ to request that your submission be made public.

Once your submission is approved to be public, it will appear on the main leaderboard as shown below.

.. image:: https://africa.dlnlp.ai/sahara/img/public_results.png
   :width: 70%
   :align: center
