Sahara Benchmark Documentation
=======================
.. raw:: html

   <p align="center">
    <br>
    <img src="https://africa.dlnlp.ai/sahara/img/sahara_web_main.jpg"/>
    <br>
   <p>


Sahara is a comprehensive benchmark for African NLP, part of our ACL 2025 paper, "[Where Are We? Evaluating LLM Performance on African Languages]()". Africa's rich linguistic heritage remains underrepresented in NLP, largely due to historical policies that favor foreign languages and create significant data inequities. In the paper, we integrate theoretical insights on Africa's language landscape with an empirical evaluation using Sahara. Sahara is curated from large-scale, publicly accessible datasets capturing the continent's linguistic diversity. By systematically assessing the performance of leading large language models (LLMs) on Sahara, we demonstrate how policy-induced data variations directly impact model effectiveness across African languages. Our findings reveal that while a few languages perform reasonably well, many Indigenous languages remain marginalized due to sparse data. Sahara supports an impressive 517 languages and varieties, across 16 tasks, making it the most extensive and representative benchmark for African NLP.

:github: https://github.com/UBC-NLP/sahara
:official websire: https://africa.dlnlp.ai/sahara/
:leaderbaords: https://huggingface.co/spaces/UBC-NLP/sahara
:paper: https://aclanthology.org/2025.acl-long.1572/

 
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tasks
   instructions



.. _citation:

========
Citation
========

If you use the Sahara benchmark for your scientific publication, or if you find the resources in this website useful, please cite our paper.

.. code-block:: bibtex

    @inproceedings{adebara-etal-2025-evaluating,
        title = "Where Are We? Evaluating {LLM} Performance on {A}frican Languages",
        author = "Adebara, Ife  and      Toyin, Hawau Olamide  and      Ghebremichael, Nahom Tesfu  and      Elmadany, AbdelRahim A.  and      Abdul-Mageed, Muhammad",
        editor = "Che, Wanxiang  and      Nabende, Joyce  and      Shutova, Ekaterina  and      Pilehvar, Mohammad Taher",
        booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2025",
        address = "Vienna, Austria",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2025.acl-long.1572/",
        pages = "32704--32731",
        ISBN = "979-8-89176-251-0",
    }
