import speech_recognition as sr
from nltk.corpus import stopwords


class Transcriber:

    def __init__(self, language_pack_for_ai: str):
        self.language = language_pack_for_ai

    def transcribeAudioFileNonDict(self, audio_file):
        """
        Transcribe an audio file to string data type without dictionarization
        :param audio_file:
        :return: str
        """

        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:  # open audio file
            audio_data = r.record(source)  # listen for data (load audio to memory)
            text = r.recognize_google(audio_data, language=self.language)  # recognize (convert from speech to text
            return text  # return converted audio as string

    def readPerfectTranscript(self, transcript):
        """
        You must have a good/perfect transcript available as a .txt file as a reference to the transcription
        that was produced by the AI
        :param transcript:
        :return: str
        """

        with open(transcript) as file:  # open .txt file
            return file.read()  # read the content of .txt file and return it as one string

    def StopwordRemover(self, transcription: str, language_stopwords="english"):
        """
        Remove the stopwords from the specified language from the string argument
        :param transcription:
        :param language_stopwords:
        :return: str
        """
        l = []
        for word in transcription.split(" "):
            if word not in stopwords.words(language_stopwords):
                l.append(word)

        return " ".join(l)
