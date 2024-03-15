import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the training dataset
train_data = pd.read_csv(r"C:\Users\Parth\Desktop\kaggle competitions\twitter_training.csv")

# Load the validation dataset
validation_data = pd.read_csv(r"C:\Users\Parth\Desktop\kaggle competitions\twitter_validation.csv")

# Define preprocessing function
def preprocess_text(text):
    text = text.fillna('')
    text = text.str.lower()
    text = text.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    text = text.str.replace(r'\W', ' ', regex=True)
    text = text.str.replace(r'\s+[a-zA-Z]\s+', ' ', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True)
    return text

# Define tokenization and lemmatization function
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Preprocess the 'TWEETS' column in the training dataset
train_data['TWEETS'] = preprocess_text(train_data['TWEETS'])
train_data['TWEETS'] = train_data['TWEETS'].apply(tokenize_and_lemmatize)

# Preprocess the 'TWEETS' column in the validation dataset
validation_data['TWEETS'] = preprocess_text(validation_data['TWEETS'])
validation_data['TWEETS'] = validation_data['TWEETS'].apply(tokenize_and_lemmatize)

# Vectorize text data in the training dataset
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['TWEETS'])
y_train = train_data['TARGET']

# Initialize the new model (Random Forest Classifier)
model = RandomForestClassifier()

# Train the new model
model.fit(X_train, y_train)

# Vectorize text data in the validation dataset
X_validation = vectorizer.transform(validation_data['TWEETS'])

# Make predictions on the validation set using the new model
y_pred_validation = model.predict(X_validation)

# Calculate accuracy on the validation set
accuracy_validation = accuracy_score(validation_data['TARGET'], y_pred_validation)
print("Accuracy on Validation Set (Random Forest Classifier):", accuracy_validation)
