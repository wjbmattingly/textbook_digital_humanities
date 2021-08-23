There are many different ways we can extract keywords from a text using both heuristic, or rules-based, and ML-based approaches. There are three different methods in this app: TF-IDF, Graph-Based, and KeyBERT. Paste in three different texts below or use the ones provided. Play with the different approaches.

TF-IDF stands for Term-Frequency, Inverse-Document Frequency. It is the standard algorithm used by most search engines today. It is considered a rules-based approach because the algorithm was created by man and does not require training to hone its parameters. On this app, you can control several parameters of the TF-IDF algorithm:

* Max Features => This is the quantity of words the algorithm is allowed to accept.
* N-Gram Range (Low) => The minimum size of words (i.e. machine_learning would be considered 2-grams)
* N-Gram Range (High) => The maximum size of a gram, or words.

**THE N-GRAM (HIGH) MUST! BE HIGHER THAN THE LOW**

Graph-based  keyword extraction is still a heuristic algorithm, but the algorithm is a bit more complex than TF-IDF which only considers a few aspects of a word relative to the corpus. Graph-based methods apply graphing, or network, algorithms to text-based questions. Some graphing approaches work at the word-level, while others work at the sentence-level. The idea is to understand a word in context to rank its importance.

KeyBERT is a newer approach. It leverages a BERT model, an advanced language model that is used for everything from text summarization to machine translation. BERT models can be used to extract keywords and phrases from texts, an essential step in the process of text summarization.
