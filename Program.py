import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_csv('IMDB Dataset.csv')
print(data['sentiment'].value_counts())
data.dropna(inplace=True)  # checking if we have any duplicate Data and removing it

# Define stopwords and additional unnecessary words
stop_words = set(stopwords.words('english'))
unnecessary_words = {'the', 'that', 'this', 'those', 'these', 'it', 'its', 'there', 'here'}
all_stop_words = stop_words.union(unnecessary_words)  # Combine both sets
print(data)
def clean_data(review):


    # Convert to lowercase
    review = review.lower()

    # Remove HTML tags if any
    review = re.sub(r'<[^>]+>', '', review)

    # Remove URLs
    review = re.sub(r'http\S+|www.\S+', '', review)

    # Remove punctuation
    review = review.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    review = re.sub(r'\d+', '', review)

    # Tokenize and filter stopwords
    words = review.split()
    filtered_words = [word for word in words if word not in all_stop_words]

    # Rejoin filtered words
    cleaned_review = ' '.join(filtered_words)
    return cleaned_review

data['review'] = data['review'].apply(clean_data)
from wordcloud import WordCloud
# draw wordcloud
reviewsNeg = ' '.join(word for word in data['review'][data['sentiment'] == 'negative'].astype(str))
wordcloud = WordCloud(height = 600, width = 1000, max_font_size = 100)
plt.figure(figsize=(15,12))
plt.imshow(wordcloud.generate(reviewsNeg), interpolation='bilinear')
plt.axis('off')
plt.show()
reviewsPos = ' '.join(word for word in data['review'][data['sentiment']== 'positive'].astype(str))
wordcloud = WordCloud(height = 600, width = 1000, max_font_size = 100)
plt.figure(figsize=(15,12))
plt.imshow(wordcloud.generate(reviewsNeg), interpolation='bilinear')
plt.axis('off')
plt.show(title="Positive")
