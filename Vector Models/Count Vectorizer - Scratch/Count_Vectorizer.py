import numpy as np

class Count_Vectorizer:

    def __init__(self) -> None:
        self.vocabulary = {}

    def fit(self, documents):
        for document in documents:
            unique_words = set(document.split())
            for word in unique_words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = 0
                self.vocabulary[word] += 1
        return self

    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocabulary)))
        for i, document in enumerate(documents):
            for j, word in enumerate(self.vocabulary.keys()):
                
                X[i][j] = document.split().count(word)
        return X


vectorizer = Count_Vectorizer()

documents = ["This is the first document", "This is the second document", "This is the third document"]

X = vectorizer.fit(documents).transform(documents)
print(X)






