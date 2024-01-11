#import modules
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale
#from sklearn.linear_model import LinearRegression

#Using Kaggle Data (Lyric Scraper got blocked)
#Added random years because kaggle data didnt have years
#Add A Song
df = pd.read_csv('dataset_TheodoreGaidis.csv', error_bad_lines=False)
#df['Year'] = np.random.randint(1980,2020, size=len(df))

'''
Song recomendation. Finds 
'''

#Cleaning Module
#Writing my own cleaning functions bc why not
#Remove punctuation/ spaces
#make everything lowercase
#Remove Stopwords
#Word Count
#stopwords to remove
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
             "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
             "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
             "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
             "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
             "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
             "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", 
             "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", 
             "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
             "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

#go song by song, remove puncutation, make all lowercase, count words after cleaning
def cleanLyrics(df, stopwords):
    songs = df['lyrics'].tolist()
    cleaned = []
    count = []
    for song in songs:
        song = song.translate(str.maketrans('','', string.punctuation))
        song = song.lower()
        lyrics = song.split(" ")
        #print(len(lyrics))
        for word in lyrics:
            if word in stopwords:
                lyrics.remove(word)
        cleaned.append(lyrics)
        count.append(int(len(lyrics)))
    #print(cleaned, count)
    #cleanedWords = pd.DataFrame(cleaned, columns = ['CleanedLyrics'])
    wordCount = pd.DataFrame(count, columns = ['WordCount'])
    #df['CleanedLyrics'] = cleanedWords['CleanedLyrics']
    df['WordCount'] = wordCount['WordCount']
    return df

cleanDF = cleanLyrics(df, stopwords)

#Unique Words Counter
#for each song, create an empty dict and add to it if word is new, or increase count if word is repeated
#Count num of words where count is 0
def uniqueCount(df):
    uniqueCount = []
    songs = df['lyrics'].tolist()
    for song in songs:
        wordCount = {}
        lyrics = song.split(" ")
        for lyric in lyrics:
            counter = 0
            if lyric in wordCount:
                wordCount[lyric] += 1
            else:
                wordCount[lyric] = 1
        for word in wordCount:
            if wordCount[word] == 1:
                counter += 1
        uniqueCount.append(counter)
        
    countDF = pd.DataFrame(uniqueCount, columns = ['UniqueCount'])
    df['UniqueCount'] = countDF['UniqueCount']
    
    return df

allData = uniqueCount(cleanDF)

def cosine_similarity(df):
    vectors = [set(word_tokenize(lyrics.lower())) for lyrics in df['lyrics'].tolist()]
    cos_sim = []
    # Compute the cosine similarity using the last song in the list
    reference_set = vectors[-1]
    print(df.tail(1))
    
    for song_set in vectors:
        rvector = reference_set.union(song_set)
        a = np.array([1 if word in reference_set else 0 for word in rvector], dtype=np.float64)
        b = np.array([1 if word in song_set else 0 for word in rvector], dtype=np.float64)
        
        # Calculate cosine similarity
        cosine = np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
        cos_sim.append(cosine)
    
    df['COSINE_SIM'] = cos_sim
    return df

allData = cosine_similarity(allData)

print(allData.sort_values('COSINE_SIM', ascending=False).head(10))
