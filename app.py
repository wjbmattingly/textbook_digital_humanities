import streamlit as st
import fasttext
import glob
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

st.title("Digital Humanities Apps")


type = st.sidebar.selectbox("types",
        ("Generate Text", "Natural Language Processing (NLP)", "Images"))

if type == "Generate Text":
    st.write("Choose the method for how you want to separate your files. You can separate things by a known occurence of a word in your individual file, such as a chapter. Another option is to upload multiple files, where each file is a unique item in your corpus.")

    text_options = st.selectbox("Select Generator", ("Separate File by Chapter", "Upload Multiple Files"))

    if text_options == "Separate File by Chapter":
        text_options_form = st.form(key="Text Options Form")
        file = text_options_form.file_uploader("Upload File")
        chapter_format = text_options_form.text_input("How are chapters formatted, e.g. CHAPTER or Chapter 01? This is case-sensitive.")
        text_options_form_button = text_options_form.form_submit_button("Submit")
        if text_options_form_button:
            file = file.read().decode().split(chapter_format)
            all_text = "\n\n".join(file)
            words = len(all_text.split())
            words = locale.format("%d", words, grouping=True)
            st.write(f'Your text is {words} words long.')
            output = st.text_area("Output", all_text)






    if text_options == "Upload Multiple Files":
        st.write("Here select all your text files that you wish to analyze in this app. The output will be long text that has all files separated by double line breaks. You can then copy and paste this file in each of the apps.")
        files = st.file_uploader("Upload Files", accept_multiple_files=True)
        final = []
        for file in files:
            file = file.read().decode()
            temp = file.replace("-\n", "").replace("\n", " ")
            while "  " in temp:
                temp = temp.replace("  ", " ")
            final.append(temp)
        all_text = "\n\n".join(final)
        words = len(all_text.split())
        words = locale.format("%d", words, grouping=True)
        st.write(f'Your text is {words} words long.')
        output = st.text_area("Output", all_text)

model = ""

if type == "Natural Language Processing (NLP)":
    nlp_options = st.sidebar.selectbox("NLP Options",
    ("Word Embeddings - Create Model",
    "Word Embeddings - Use Model",
    "Sentence Embeddings",
    "Named Entity Recognition (NER)",
    "TF-IDF",
    "Graph-Based Extraction",
    "KeyBERT"))
    if nlp_options == "Word Embeddings - Create Model":
        from gensim.models import Word2Vec, FastText
        from gensim.models.phrases import Phraser, Phrases
        import spacy
        from gensim.parsing.preprocessing import preprocess_documents
        from gensim.utils import tokenize
        nlp =  spacy.blank("en")
        nlp.max_length = 100000000
        nlp.add_pipe("sentencizer")
        st.write("Word2Vec Analysis")
        word2vec_form = st.form("Word2Vec Form")
        text = str(word2vec_form.text_input("Insert your text here."))
        text = text.replace("-\n", "\n").replace("\n", " ")
        text = ''.join([i for i in text if not i.isdigit()])
        word2vec_button = word2vec_form.form_submit_button()

        vocab = ""
        if word2vec_button:
            import string
            sentences  = []
            text = str(text).replace("“", "").replace("“", "")
            doc = nlp(text)
            prepared_texts = []
            for sent in doc.sents:
                sentences.append(sent.text)
            tokenized_sentences  = [list(tokenize(doc, lower=True)) for doc in sentences]

            # phrases = Phrases(tokenized_sentences)

            bigram = Phrases(tokenized_sentences)
            # sentences = list(bigram[tokenized_sentences])

            trigram = Phrases(bigram[tokenized_sentences])
            sentences = list(trigram[tokenized_sentences])

            model = FastText(sentences, size=100, window=20, min_count=10,sg=0)
            st.session_state['word2vec'] = model
            st.write("The Word2Vec Model has finished training. You can now use it. Under NLP Options, select 'Word Embeddings - Use Model'. You can see your model's vocabulary down below.")
            vocab_expander = st.expander("Vocabulary")
            vocab = list(model.wv.vocab)
            vocab_expander.write(vocab)

    elif nlp_options == "Word Embeddings - Use Model":
        #https://github.com/AmmarRashed/word_embeddings_hp/blob/master/gensim_vecs.ipynb
        if "word2vec" in st.session_state:
            # model = fasttext.load_model("temp_model.bin")
            model = st.session_state["word2vec"]
            #
            vocab_expander = st.expander("Vocabulary")
            vocab = list(model.wv.vocab)

            vocab_expander.write(vocab)
            model_form = st.form("Model Form")
            search = model_form.selectbox("Search for a word", vocab)
            model_submit = model_form.form_submit_button("Run Search")

            if model_submit:
                search = str(search)
                st.write(search)
                st.write(model)


                word_vecs = model[search]
                col1, col2 = st.columns(2)
                col1.header("Word Vector Shape")
                col1.write(str(word_vecs))

                col2.header("Most Similar Words")
                results = model.most_similar(search)
                for x in results:
                    col2.write (x)
        else:
            st.warning("You must create a model first.")
    if nlp_options == "TF-IDF" or nlp_options == "Graph-Based Extraction" or nlp_options == "KeyBERT":
        text1 = st.text_input("First Text", "Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball (approximately 9.4 inches (24 cm) in diameter) through the defender's hoop (a basket 18 inches (46 cm) in diameter mounted 10 feet (3.048 m) high to a backboard at each end of the court) while preventing the opposing team from shooting through their own hoop. A field goal is worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops and the player fouled or designated to shoot a technical foul is given one, two or three one-point free throws. The team with the most points at the end of the game wins, but if regulation play expires with the score tied, an additional period of play (overtime) is mandated. Players advance the ball by bouncing it while walking or running (dribbling) or by passing it to a teammate, both of which require considerable skill. On offense, players may use a variety of shots – the layup, the jump shot, or a dunk; on defense, they may steal the ball from a dribbler, intercept passes, or block shots; either offense or defense may collect a rebound, that is, a missed shot that bounces from rim or backboard. It is a violation to lift or drag one's pivot foot without dribbling the ball, to carry it, or to hold the ball with both hands then resume dribbling.The five players on each side fall into five playing positions. The tallest player is usually the center, the second-tallest and strongest is the power forward, a slightly shorter but more agile player is the small forward, and the shortest players or the best ball handlers are the shooting guard and the point guard, who implements the coach's game plan by managing the execution of offensive and defensive plays (player positioning). Informally, players may play three-on-three, two-on-two, and one-on-one. ]Invented in 1891 by Canadian-American gym teacher James Naismith in Springfield, Massachusetts, United States, basketball has evolved to become one of the world's most popular and widely viewed sports.[1] The National Basketball Association (NBA) is the most significant professional basketball league in the world in terms of popularity, salaries, talent, and level of competition.[2][3] Outside North America, the top clubs from national leagues qualify to continental championships such as the EuroLeague and the Basketball Champions League Americas. The FIBA Basketball World Cup and Men's Olympic Basketball Tournament are the major international events of the sport and attract top national teams from around the world. Each continent hosts regional competitions for national teams, like EuroBasket and FIBA AmeriCup. The FIBA Women's Basketball World Cup and Women's Olympic Basketball Tournament feature top national teams from continental championships. The main North American league is the WNBA (NCAA Women's Division I Basketball Championship is also popular), whereas the strongest European clubs participate in the EuroLeague Women.")

        text2 = st.text_input("Second Text", 'Machine learning (ML) is the study of computer algorithms that can improve automatically through experience and by the use of data.[1] It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.[2] Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.[3] A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers; but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.[5][6] Some implementations of machine learning use data and neural networks in a way that mimics the working of a biological brain.[7][8] In its application across business problems, machine learning is also referred to as predictive analytics. Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. It involves computers learning from data provided so that they carry out certain tasks. For simple tasks assigned to computers, it is possible to program algorithms telling the machine how to execute all steps required to solve the problem at hand; on the computers part, no learning is needed. For more advanced tasks, it can be challenging for a human to manually create the needed algorithms. In practice, it can turn out to be more effective to help the machine develop its own algorithm, rather than having human programmers specify every needed step.[9]')

        text3 = st.text_input("Third Text", "Coronavirus disease 2019 (COVID-19) is a contagious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first known case was identified in Wuhan, China, in December 2019.[7] The disease has since spread worldwide, leading to an ongoing pandemic.[8] Symptoms of COVID-19 are variable, but often include fever,[9] cough, headache,[10] fatigue, breathing difficulties, and loss of smell and taste.[11][12][13] Symptoms may begin one to fourteen days after exposure to the virus. At least a third of people who are infected do not develop noticeable symptoms.[14] Of those people who develop symptoms noticeable enough to be classed as patients, most (81%) develop mild to moderate symptoms (up to mild pneumonia), while 14% develop severe symptoms (dyspnea, hypoxia, or more than 50% lung involvement on imaging), and 5% suffer critical symptoms (respiratory failure, shock, or multiorgan dysfunction).[15] Older people are at a higher risk of developing severe symptoms. Some people continue to experience a range of effects (long COVID) for months after recovery, and damage to organs has been observed.[16] Multi-year studies are underway to further investigate the long-term effects of the disease.[16] COVID-19 transmits when people breathe in air contaminated by droplets and small airborne particles. The risk of breathing these in is highest when people are in close proximity, but they can be inhaled over longer distances, particularly indoors. Transmission can also occur if splashed or sprayed with contaminated fluids, in the eyes, nose or mouth, and, rarely, via contaminated surfaces. People remain contagious for up to 20 days, and can spread the virus even if they do not develop any symptoms.[17][18] Several testing methods have been developed to diagnose the disease. The standard diagnostic method is by detection of the virus' nucleic acid by real-time reverse transcription polymerase chain reaction (rRT-PCR), transcription-mediated amplification (TMA), or by reverse transcription loop-mediated isothermal amplification (RT-LAMP) from a nasopharyngeal swab. Preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. The use of face masks or coverings has been recommended in public settings to minimize the risk of transmissions. While work is underway to develop drugs that inhibit the virus (and several vaccines for it have been approved and distributed in various countries, which have since initiated mass vaccination campaigns), the primary treatment is symptomatic. Management involves the treatment of symptoms, supportive care, isolation, and experimental measures.")
        all_text = [str(text1), str(text2), str(text3)]
        if nlp_options == "TF-IDF":
            import string
            #https://towardsdatascience.com/keyword-extraction-python-tf-idf-textrank-topicrank-yake-bert-7405d51cd839
            from sklearn.feature_extraction.text import TfidfVectorizer
            from numpy import array, log
            from re import sub


            max_features = st.slider("Max Features", 1, 40)

            vectorizer = TfidfVectorizer(
                                    lowercase=True,
                                    max_features=max_features,
                                    max_df=0.4,
                                    ngram_range = (1,2),
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

            for i in range(len(all_text)):
                st.header(f"These are the Key Terms for Text {i+1}")
                words = "\n * ".join(all_keywords[i])
                words = "* "+words
                st.markdown(words)

        elif nlp_options == "Graph-Based Extraction":
            from summa import keywords
            for i in range(len(all_text)):
                st.header(f"These are the Key Terms for Text {i+1}")
                words = keywords.keywords(all_text[i], words=5).split()
                words = "\n * ".join(words)
                words = "* "+words
                st.markdown(words)

        elif nlp_options == "KeyBERT":
            from keybert import KeyBERT
            kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
            st.write("Downloaded Model.")
            for i in range(len(all_text)):
                st.header(f"These are the Key Terms for Text {i+1}")
                words = kw_extractor.extract_keywords(all_text[i], keyphrase_ngram_range=(1, 2), stop_words='english')
                final = []
                for word in words:
                    new = str(word)
                    final.append(new)
                words = final
                words = "\n * ".join(words)
                words = "* "+words
                st.markdown(words)
