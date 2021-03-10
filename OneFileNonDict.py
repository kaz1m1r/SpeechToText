import spacy  # https://spacy.io/usage/linguistic-features#vectors-similarity
import speech_recognition as sr


def transcribeFile(audio_file):
    r = sr.Recognizer()

    # Open audio file
    with sr.AudioFile(audio_file) as source:
        # listen for the data (load autio to memory)
        audio_data = r.record(source)

        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data, language="en-US")  # https://stackoverflow.com/questions/14257598/what
        # -are-language-codes-in-chromes-implementation-of-the-html5-speech-recogniti

        # return content as string
        return text


def textFromTranscription(text_file):
    # open file
    with open(text_file) as file:
        # return content of file as string
        return file.read()


# audio_file = "testAudio.wav"
audio_file = "multipleSentences.wav"
audio_file_transcription = "multipleSentencesTranscription.txt"

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

t = transcribeFile(audio_file)
r = textFromTranscription(audio_file_transcription)

# tokenization (all words from transcription are saved in one big list
t_list = word_tokenize(t)
r_list = word_tokenize(r)

# saving stopwords of the english language
sw = stopwords.words('english')

l1 = []
l2 = []

# Saving words that are not common english stop words. This is done because these words have no influence on the
# actual content of the conversation. Rather, they can not really be used to identify the content of the conversation
t_set = {w for w in t_list if not w in sw}
r_set = {w for w in r_list if not w in sw}

a = []
for word in t_set:
    if word not in r_set:
        a.append(word)
print(f"words not in common   : {a}")
rvector = t_set.union(r_set)
print(f"rvector           : {rvector}")

for i in a:
    if i not in rvector:
        print(i)

for w in rvector:
    if w in t_set:
        l1.append(1)
    else:
        l1.append(0)
    if w in r_set:
        l2.append(1)
    else:
        l2.append(0)
c = 0

for i in range(len(rvector)):
    c += l1[i] * l2[i]

l1_squared = [a ** 2 for a in l1]
l2_squared = [b ** 2 for b in l2]
# cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
cosine = c / float((sum(l1_squared) ** 0.5) * (sum(l2_squared) ** 0.5))
print(f"Cosine similarity : {cosine}")