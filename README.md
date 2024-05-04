# NLP Course Notes

## Intuition for the course:

1. We are trying to apply machine learning algorithms on text data in order to analyze it and discover trends and useful insights.
2. But ML models cannot be directly applied on text (at least not easily). For that we need numbers or numerical data.
3. So here we convert our textual data into numerical format in hopes of applying machine learning algorithms to it.
4. This is mainly done by converting it into vectors by using several text pre-processing methods.

# 0. Some Basic Definitions:

1. **Tokens**: These are similar to words in English language.
2. **Corpus**: A large collection of text based on similar subject that can be used for linguistic analysis.
3. **Vocabulary**: In NLP, it can be defined as a large collection of all tokens.
4. **Bag of Words**: In NLP, it is a collection of tokens or a sequence of tokens (words) in a randomized order. Vector models extensively rely on this bag of words approach.
5. **Tokenization**: The splitting of raw text into a list of words that can be used for linguistic analysis is called tokenization.
6. **Vectorization:** Machine learning with natural language is faced with one major hurdle – its algorithms usually deal with numbers, and natural language is, well, text. So we need to transform that text into numbers, otherwise known as text vectorization. It’s a fundamental step in the process of machine learning for analyzing data.
    
   → Once you’ve transformed words into numbers, in a way that’s machine learning
    
   algorithms can understand, the TF-IDF score can be fed to algorithms.    
    
      → a word vector represents a document as a list of numbers, with one for each possible word of the corpus. Vectorizing a document is taking the text and creating one of these vectors, and the numbers of the vectors somehow represent the content of the text.
    
7. **Stopwords**: The words that we ignore from the raw text are called stopwords. They have the potential to overshadow other words that are more important to the specific linguistic analysis.
8. **Stemming & Lemmatization**: These are the text preprocessing techniques used to reduce the  dimensionality of the word vectors and reduce the size of the vocabulary. These techniques aim to dissect related words and convert it to their root form in order to ease up computation and avoid higher-dimensional word vectors.
    
    
    | Stemming  | Lemmatization |
    | --- | --- |
    | It is very crude, just cuts the end of the word to find the root | It is a sophisticated technique to find the root.  |
    | The result does not guarantee  to be a meaningful word | The result is always a true root word. |
9. **Similarity Score:** This is just a value that determines the similarity or dissimilarity of 2 vectors. Mostly in NLP, it is calculated using the inner product (dot) of the 2 vectors (X * Y * cos). The other way to calculate is using Euclidean distances but it is not accurate all the time.
    
    $$
    X.Y = |X| |Y| cos(theta)
    $$
    
    
    $$
    cos(theta) = X.Y / |X| |Y|
    $$
    
    $$
    cos(theta) = (X · Y) / √(Σ x_i^2) √(Σ y_i^2)
    $$
    
10. **Cosine Distance:** It is another metric that can be used to determine the similarity between vectors. 

$$
cosine distance = 1 - cosine Similarity
$$

# 1. Vector Models:

## A. TFIDF Vectorizer:

1. TF-IDF (Term Frequency-Inverse Document Frequency) is a vectorization technique commonly used in natural language processing (NLP) and information retrieval.
2. It considers for the things that the simple count vectorizer is not able to do. The simple count vectorizer does not account for non-informative words that appear too many times in a document, but TFIDF does. 
3. It is used to convert a collection of text documents into numerical vectors, where each document is represented as a vector in a high-dimensional space.
4. TF-IDF takes into account the importance of words in a document relative to their frequency across all documents in the corpus. The score is calculated based on two components:
    1. **Term Frequency (TF):** Measures how often a word appears in a specific document.
    2. **Inverse Document Frequency (IDF):** Measures the rarity of a word across all documents in the corpus. Words that are common across all documents receive a lower IDF score, while words that are unique or specific to certain documents receive a higher IDF score. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
5. The product of TF and IDF yields the TF-IDF score, which is used as the numerical representation of a word in a document. This vectorization technique helps capture the importance of words in each document while also considering their uniqueness across the entire corpus.
6. Words that are common in every document, such as this, what, and if, rank low even though they may appear many times, since they don’t mean much to that document in particular.
7. Multiplying these two numbers results in the TF-IDF score of a word in a document. The higher the score, the more relevant that word is in that particular document.
8. To put it in more formal mathematical terms, the TF-IDF score for the word t in the document d from the document set D is calculated as follows:
    
   
9. Once you’ve transformed words into numbers, in a way that machine learning algorithms can understand, the TF-IDF score can be fed to algorithms such as Naïve Bayes and Support Vector Machines, greatly improving the results of more basic methods like word counts.
10. TF-IDF enables us to give a way to associate each word in a document with a number that represents how relevant each word is in that document. Then, documents with similar, relevant words will have similar vectors, which is what we are looking for in a machine learning algorithm.
11. Code Links: 
    1. [TFIDF Movie Recommender Mini Project.](https://colab.research.google.com/drive/1FtSFsbjvmSnUxKpVHgt5i7gGCrAXfQd1)
    2. [TFIDF from Scratch.](https://deeplearningcourses.com/notebooks/rW3EZnH2QHQDMlYbEb5nrg/NsTJI0NMS5-D8xbcu2zGWA)

## B. Neural Word Embeddings:

1. This is just another advanced way to convert words into numerical vectors in order to apply deep learning algorithms on it.
2. Here, single words are converted into vectors instead of whole document converting to a vector. This results in extracting even more information from the words than the previous approach.
3. Therefore here, a document is converted into a sequence of vectors where their order is very important.
4. The bag of words approach doesn’t work here as the deep learning algorithms such as CNN’s and ANN’s specifically require the input data in sequence in order to produce a more detailed output. 
5. In this, the process of converting words into vectors give them a meaning just by being in the vector space. This can be used in deep learning as the weights for the initial layer of this type of model.

# 2. Probabilistic Models:

## A. Markov Models:

1. The Markov models are a type of probabilistic models where numerical probability is used in order to determine the output i.e. classify the text or model sequences. It forms the backbone of the Markov Decision Process which is the base framework used in Reinforcement Learning.
2. **Markov Property:**
    1. It is the most essential thing that we need to build an Markov model.
    2. The Markov property is a very restrictive assumption on the dependency structure of the joint distribution. Basically it means that any event or timestamp or a unit in a sequence is dependent  only on the unit or event or timestamp previous to it. Xt is dependent only on Xt-1 and not on anyone else.  
    3. The Markov property appears to be very restrictive when applied to natural language as in the English languages the last words of sentences are often dependent upon multiple words in the sentence and not just the last word as the Markov property assumes.
    
    
3. The Markov model can be described using 2 distributions:
    1. **State transition matrix:** This matrix describes the probability distribution of all states transitioning into the next state based on the state before them i.e. using the Markov property.
    2. **Initial distribution matrix:** Since the initial state cannot come under the Markov property, this N size vector describes the probability distribution of all states being the first state.  
    
    
4. When A and π are known along with the sequence we want to model, we can find out the probability of that sequence appearing which is given by:
    
    
    
5. Using maximum likelihood probabilities like this can cause problems as many of these probabilities can converge to 0 just because of the sheer size of the vocabulary and sometimes because they are not found in the entire training data set. To counter this, we can introduce **Log Probabilities.**
6. Markov Property doesn’t work well with English language text as in the English language determining the next word doesn’t depend on just the last word but actually the last 4-5 words in order to understand the context of the sentence. So, to accommodate for this, we can extend our Markov model to a second order where the current state depends on not just the previous but the previous 2 states. We can go on further with this and increase the order of the model but this is computationally very costly as with every added state we are seeing exponential growth in the size of the state distribution matrix (A).
7. Code Links : 
    1. [Text Classification using Markov Model from scratch](https://github.com/Raghav2305/NLP/tree/master/Markov%20Models/Text%20Classification%20-%20Markov%20Model)
    2. [Poetry Generation using Markov Models from scratch](https://github.com/Raghav2305/NLP/tree/master/Markov%20Models/Poetry%20Generator%20-%20Markov%20Models)
