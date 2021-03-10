from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

a = "The quick brown fox jumps over the lazy dog"
b = "The big brown fox jumps over the lazy dog"
c = "A quick brown fox jups odor the very lazy dog"
d = "The quick brown fox jumps over the lazy dog"

docs = [a, b, c, d]
vect = TfidfVectorizer()
tfidf_matrix = vect.fit_transform(docs)
print(f"tfidf matrix : \n{tfidf_matrix}")
cosine = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
print(f"Cosine       : {cosine}")
# cosine = cosine_similarity(tfidf_matrix[0, :], tfidf_matrix[1, :])
# print(f"Cosine       : {cosine}")
# print(f"cosine similarity tfidf_matrix : {cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)}")
# print(f"Cosine similarity : {cosine_similarity(tfidf_matrix[0: 1], tfidf_matrix)[0, 1]}")


a_list = word_tokenize(a.lower())
b_list = word_tokenize(b.lower())
c_list = word_tokenize(c.lower())
d_list = word_tokenize(d.lower())

sw = stopwords.words('english')

a_set = {w for w in a_list if not w in sw}
b_set = {w for w in b_list if not w in sw}
c_set = {w for w in c_list if not w in sw}
d_set = {w for w in c_list if not w in sw}

ab_vector = a_set.union(b_set)  # union between a_set and b_set

la = []
lb = []

for w in ab_vector:
    if w in a_set: la.append(1)
    else: la.append(0)
    if w in b_set: lb.append(1)
    else: lb.append(0)

la_squared = [a**2 for a in la]
lb_squared = [b**2 for b in lb]

c = 0
for i in range(len(ab_vector)):
    c += la[i] * lb[i]

cosine = c / float((sum(la_squared) ** 0.5) * (sum(lb_squared) ** 0.5))

# cosine = c / float((sum(la) * sum(lb)) ** 0.5)
print(f"\n"
      f"Cosine similarity '{a}' and '{b}' is"
      f"\n"
      f"Cosine : {cosine}")