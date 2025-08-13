.. Sahara Benchmark Tasks

=========
Sahara Tasks
=========

The Sahara benchmark evaluates model performance across 16 tasks, categorized into four primary clusters, to reflect Africa's diverse linguistic landscape.

Multiple-Choice, Comprehensive and Reasoning (MCCR) Tasks
=========================================================

* **Context-based Question Answering** (squad_qa): Evaluated using *Macro F1*, supporting 1 language.
* **General Knowledge** (mmlu): Evaluated using *Accuracy*, supporting 16 languages.
* **Mathematical Word Problems** (mgsm): Evaluated using *Exact Match*, supporting 16 languages.
* **Reading Comprehension** (belebele): Evaluated using *Accuracy*, supporting 25 languages.

Text Classification Tasks
=========================

* **Cross-Lingual Natural Language Inference** (xlni): Evaluated using *Accuracy*, supporting 16 languages.
* **Language Identification** (lid): Evaluated using *Macro F1*, supporting 517 languages.
* **News Classification** (news): Evaluated using *Macro F1*, supporting 4 languages.
* **Sentiment Analysis** (sentiment): Evaluated using *Macro F1*, supporting 3 languages.
* **Topic Classification** (topic): Evaluated using *Macro F1*, supporting 2 languages.

Text Generation Tasks
=====================

* **Machine Translation - African to African** (mt_xx2xx): Evaluated using *spBleu-1K*, supporting 29 languages.
* **Machine Translation - English to African** (mt_eng2xx): Evaluated using *spBleu-1K*, supporting 29 languages.
* **Machine Translation - French to African** (mt_fra2xx): Evaluated using *spBleu-1K*, supporting 29 languages.
* **Paraphrase** (paraphrase): Evaluated using *spBleu-1K*, supporting 4 languages.
* **Summarization** (summary): Evaluated using *RougeL*, supporting 10 languages.
* **Title Generation** (title): Evaluated using *spBleu-1K*, supporting 10 languages.

Tokens Level Tasks
==================

* **NER** (ner): Evaluated using *Macro F1*, supporting 27 languages.
* **Phrase Chunking** (phrase): Evaluated using *Macro F1*, supporting 8 languages.
* **POS Tagging** (pos): Evaluated using *Macro F1*, supporting 1 language.
