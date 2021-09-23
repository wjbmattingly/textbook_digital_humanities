import streamlit as st
import glob

def read_md(file):
    with open (file, "r", encoding="utf-8") as f:
        text = f.read()
    return (text)

st.set_page_config(page_title="The Digital Humanities App", layout="wide")
st.get_option("theme.primaryColor")


st.image("header.png")



type = st.sidebar.selectbox("Select Page",
        ("Home",
        # "Generate Text",
        "Natural Language Processing (NLP)",
        "Key Concepts"))

if type == "Home":
    st.title("Home")
    st.write(read_md("markdown_pages/home.md"))

    st.header("Directions")
    st.write(read_md("markdown_pages/directions.md"))

#
#
# if type == "Generate Text":
#     st.write("Choose the method for how you want to separate your files. You can separate things by a known occurence of a word in your individual file, such as a chapter. Another option is to upload multiple files, where each file is a unique item in your corpus.")
#
#     text_options = st.selectbox("Select Generator", ("Separate File by Chapter", "Upload Multiple Files"))
#
#     if text_options == "Separate File by Chapter":
#         text_options_form = st.form(key="Text Options Form")
#         file = text_options_form.file_uploader("Upload File")
#         chapter_format = text_options_form.text_input("How are chapters formatted, e.g. CHAPTER or Chapter 01? This is case-sensitive.")
#         text_options_form_button = text_options_form.form_submit_button("Submit")
#         if text_options_form_button:
#             file = file.read().decode().split(chapter_format)
#             all_text = "\n\n".join(file)
#             words = len(all_text.split())
#             # words = locale.format("%d", words, grouping=True)
#             st.write(f'Your text is {words} words long.')
#             output = st.text_area("Output", all_text)
#
#
#
#
#
#
#     if text_options == "Upload Multiple Files":
#         st.write("Here select all your text files that you wish to analyze in this app. The output will be long text that has all files separated by double line breaks. You can then copy and paste this file in each of the apps.")
#         files = st.file_uploader("Upload Files", accept_multiple_files=True)
#         final = []
#         for file in files:
#             file = file.read().decode()
#             temp = file.replace("-\n", "").replace("\n", " ")
#             while "  " in temp:
#                 temp = temp.replace("  ", " ")
#             final.append(temp)
#         all_text = "\n\n".join(final)
#         words = len(all_text.split())
#
#         st.write(f'Your text is {words} words long.')
#         output = st.text_area("Output", all_text)
#
# model = ""

if type == "Natural Language Processing (NLP)":
    nlp_options = st.sidebar.selectbox("NLP Options",
    (
    # "Word Embeddings - Create Model",
    # "Word Embeddings - Use Model",
    "TF-IDF",
    "Sentence Embeddings",
    "Named Entity Recognition (NER)",
    "Graph-Based Extraction",
    "KeyBERT"))
    # if nlp_options == "Word Embeddings - Create Model":
    #     st.header("Create Word Embedding Model")
    #     create_expander = st.expander("Directions for Creating a Model")
    #     create_expander.write(read_md("markdown_pages/create_embedding_model.md"))
    #
    #
    #     from gensim.models import Word2Vec, FastText
    #     from gensim.models.phrases import Phraser, Phrases
    #     import spacy
    #     from gensim.parsing.preprocessing import preprocess_documents
    #     from gensim.utils import tokenize
    #
    #
    #     nlp =  spacy.blank("en")
    #     nlp.max_length = 100000000
    #     nlp.add_pipe("sentencizer")
    #     # sp = spacy.load('en_core_web_sm')
    #     all_stopwords = nlp.Defaults.stop_words
    #     st.write(all_stopwords)
    #     word2vec_form = st.form("Word2Vec Form")
    #     text = str(word2vec_form.text_area("Insert your text here.", height=500))
    #     text = text.replace("-\n", "\n").replace("\n", " ")
    #     text = ''.join([i for i in text if not i.isdigit()])
    #     word2vec_button = word2vec_form.form_submit_button()
    #
    #     vocab = ""
    #     if word2vec_button:
    #
    #         import string
    #         sentences  = []
    #         text = str(text).replace("“", "").replace("“", "")
    #         doc = nlp(text)
    #         prepared_texts = []
    #         for sent in doc.sents:
    #             text_tokens = [token.text for token in sent if not token.text in all_stopwords]
    #             sentence = " ".join(text_tokens)
    #             sentences.append(sentence)
    #         tokenized_sentences  = [list(tokenize(doc, lower=True)) for doc in sentences]
    #
    #         # phrases = Phrases(tokenized_sentences)
    #
    #         bigram = Phrases(tokenized_sentences)
    #         # sentences = list(bigram[tokenized_sentences])
    #
    #         trigram = Phrases(bigram[tokenized_sentences])
    #         sentences = list(trigram[tokenized_sentences])
    #
    #         model = FastText(sentences, vector_size=30, window=20, min_count=10,sg=0)
    #         st.session_state['word2vec'] = model
    #         st.write("The Word2Vec Model has finished training. You can now use it. Under NLP Options, select 'Word Embeddings - Use Model'. You can see your model's vocabulary down below.")
    #         vocab_expander = st.expander("Vocabulary")
    #         vocab = list(model.wv.index_to_key)
    #         vocab_expander.write(vocab)

    # elif nlp_options == "Word Embeddings - Use Model":
    #     st.header("Use Word Embedding Model")
    #     create_expander = st.expander("Directions for Using a Model")
    #     create_expander.write(read_md("markdown_pages/use_embedding_model.md"))
    #     #https://github.com/AmmarRashed/word_embeddings_hp/blob/master/gensim_vecs.ipynb
    #     if "word2vec" in st.session_state:
    #         # model = fasttext.load_model("temp_model.bin")
    #         model = st.session_state["word2vec"]
    #         #
    #         vocab_expander = st.expander("Vocabulary")
    #         vocab = list(model.wv.index_to_key)
    #
    #         vocab_expander.write(vocab)
    #         model_form = st.form("Model Form")
    #         search = model_form.selectbox("Search for a word", vocab)
    #         model_submit = model_form.form_submit_button("Run Search")
    #
    #         if model_submit:
    #             search = str(search)
    #             st.write(search)
    #             st.write(model)
    #
    #
    #             word_vecs = model.wv[search]
    #             col1, col2 = st.columns(2)
    #             col1.header("Word Vector Shape")
    #             col1.write(str(word_vecs))
    #
    #             col2.header("Most Similar Words")
    #             results = model.wv.most_similar(search)
    #             for x in results:
    #                 col2.write (x)
    #     else:
    #         st.warning("You must create a model first.")

### SENTENCE EMBEDDINGS###
    if nlp_options == "Sentence Embeddings":
        import pandas as pd
        import json
        from sentence_transformers import util

        @st.cache(allow_output_mutation=True)
        def cache_df():
            return (pd.read_json("data/vol7.json"))
        @st.cache(allow_output_mutation=True)
        def cache_paras():
            return (load_data("data/paraphrases-10.json"))

        def search(num, df, res_container):
            descs = df.descriptions
            names = df.names
            x=1
            res_container.header (f"Searching for Similarity to:")
            res_container.write (f"Victim: {names[num]}")
            res_container.write(f"Description: {descs[num]}")
            for paraphrase in paraphrases:
                score, i, j = paraphrase
                if i == num:
                    sent1 = descs[i]
                    sent2 = descs[j]
                    res_container.header(f"Result {x}")
                    res_container.write (f"Victim: {names[j]}")
                    res_container.write(f"{sent2}")
                    res_container.write(f"")
                    res_container.write (f" Degree of Similarity: {score}")
                    res_container.write(f"")
                    res_container.write(f"")
                    x=x+1

        def load_data(file):
            with open (file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (data)

        def write_data(file, data):
            with open (file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        st.title("TRC Volume 7 - Sentence Embedding Search Engine")
        st.write("This page may take a few seconds to load...")
        st.write("Here, you will be able to engage in a machine learning method known as sentence embeddings. Like word embeddings, sentence embeddings are numerical representations of text. Unlike word embeddings, the embedding occurs not at the word-level, rather at the sentence level. This means that each sentence's semantic and syntactic value is given in a vector. With this vector we can calculate not word similarity, rather sentence similarity. This means that we can run searches on entire sentences (or paragraphs), rather than key words. Try it out. Find a description you want to match and this search engine will use that description's vetor and compare it to all other known descriptions in the database (around 22,000). It will then return the top-10 matches based on similarity. In the sidebar type the number that corresponds to your desired search. The results will appear in the Results expander below.")
        res_container = st.expander("Results")
        df = cache_df()
        paraphrases = cache_paras()
        descs = df.descriptions
        names = df.names
        search_form = st.sidebar.form("Search Form")
        search_num = search_form.number_input("Search", step=1)

        search_button = search_form.form_submit_button("Search Button")

        st.table(df)
        if search_button:
            st.sidebar.write(f"You can see the results for a search on {names[search_num]} under Results.")
            res = search(search_num, df, res_container)
    elif nlp_options == "Named Entity Recognition (NER)":
        import spacy_streamlit
        spacy_model = "en_core_web_sm"
        st.write("Paste your own text into the text area below.")
        ner_text = st.text_area("Text", "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that.", height=200)
        doc = spacy_streamlit.process_text(spacy_model, ner_text)
        spacy_streamlit.visualize_ner(
            doc,
            labels=[
                  "CARDINAL",
                  "DATE",
                  "EVENT",
                  "FAC",
                  "GPE",
                  "LANGUAGE",
                  "LAW",
                  "LOC",
                  "MONEY",
                  "NORP",
                  "ORDINAL",
                  "ORG",
                  "PERCENT",
                  "PERSON",
                  "PRODUCT",
                  "QUANTITY",
                  "TIME",
                  "WORK_OF_ART"
                ],
            show_table=False
        )


    elif nlp_options == "TF-IDF" or nlp_options == "Graph-Based Extraction" or nlp_options == "KeyBERT":
        st.header("Keyword Extraction")
        create_expander = st.expander("Directions for Using Keyword Extraction")
        create_expander.markdown(read_md("markdown_pages/keyword.md"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        container1 = col1.container()
        container2 = col2.container()
        container3 = col3.container()

        text1 = col1.text_area("Text 1", read_md("markdown_pages/text1.md"), height=300)
        text2 = col2.text_area("Text 2", read_md("markdown_pages/text2.md"), height=300)
        text3 = col3.text_area("Text 3", read_md("markdown_pages/text3.md"), height=300)

        all_text = [str(text1), str(text2), str(text3)]

        if nlp_options == "TF-IDF":
            import string
            #https://towardsdatascience.com/keyword-extraction-python-tf-idf-textrank-topicrank-yake-bert-7405d51cd839
            from sklearn.feature_extraction.text import TfidfVectorizer
            from numpy import array, log
            from re import sub


            max_features = container1.slider("Max Features", 1, 100)
            n_gram_low = container2.slider("N-Gram Range (Low)", 1, 10)
            n_gram_high = container3.slider("N-Gram Range (High)", 1, 10)

            vectorizer = TfidfVectorizer(
                                    lowercase=True,
                                    max_features=max_features,
                                    max_df=0.4,
                                    ngram_range = (n_gram_low,n_gram_high),
                                    stop_words = "english"

                                )
            vectors = vectorizer.fit_transform(all_text)
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            all_keywords = []
            for description in denselist:
                x=0
                keywords = []
                for word in description:
                    if word > 0:
                        keywords.append(feature_names[x])
                    x=x+1
                all_keywords.append(keywords)


            col1.header(f"Key Terms")
            words = "\n * ".join(all_keywords[0])
            words = "* "+words
            col1.markdown(words)

            col2.header(f"Key Terms")
            words = "\n * ".join(all_keywords[1])
            words = "* "+words
            col2.markdown(words)

            col3.header(f"Key Terms")
            words = "\n * ".join(all_keywords[2])
            words = "* "+words
            col3.markdown(words)

        elif nlp_options == "Graph-Based Extraction":
            from summa import keywords
            all_keywords = []
            for i in range(len(all_text)):
                all_keywords.append(keywords.keywords(all_text[i], words=5).split())
            col1.header(f"Key Terms")
            words = "\n * ".join(all_keywords[0])
            words = "* "+words
            col1.markdown(words)

            col2.header(f"Key Terms")
            words = "\n * ".join(all_keywords[1])
            words = "* "+words
            col2.markdown(words)

            col3.header(f"Key Terms")
            words = "\n * ".join(all_keywords[2])
            words = "* "+words
            col3.markdown(words)

        elif nlp_options == "KeyBERT":
            from keybert import KeyBERT
            top_n = container1.slider("Top-N Words", 1,50)
            n_gram_low = container2.slider("N-Gram Range (Low)", 1, 3)
            n_gram_high = container3.slider("N-Gram Range (High)", 1, 3)

            stop_words = container1.selectbox("Stopwords", ("english", "german", "spanish"))
            nr_candidates = container2.slider("Number of Candidates", 1, 50)
            diversity = container3.slider("Diversity", 0.0, 1.0, 0.7)


            kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
            all_keywords = []
            for i in range(len(all_text)):
                words = kw_extractor.extract_keywords(all_text[i],
                                keyphrase_ngram_range=(n_gram_low, n_gram_high),
                                stop_words=stop_words,
                                top_n=top_n,
                                nr_candidates=nr_candidates)
                final = []
                for word in words:
                    new = str(word)
                    final.append(new)
                all_keywords.append(final)


            col1.header(f"Key Terms")
            words = "\n * ".join(all_keywords[0])
            words = "* "+words
            col1.markdown(words)

            col2.header(f"Key Terms")
            words = "\n * ".join(all_keywords[1])
            words = "* "+words
            col2.markdown(words)

            col3.header(f"Key Terms")
            words = "\n * ".join(all_keywords[2])
            words = "* "+words
            col3.markdown(words)
elif type == "Key Concepts":
    concept_options = st.sidebar.selectbox("Choose a Concept", ("Word Embeddings", "Sentence Embeddings"))
    word_embedding_file = "markdown_pages/word_embeddings.md"
    if concept_options == "Word Embeddings":
        file = word_embedding_file
    with open (file, "r", encoding="utf-8") as f:
        text = f.read()
    st.markdown(text)


elif type == "Images":
    st.write("Page forthcoming...")
