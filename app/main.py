import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Helper function to handle unknown categories
def handle_unknown(input_value, trained_categories):
    if input_value in trained_categories:
        return input_value
    else:
        return 'Unknown'  # The 'Unknown' category must be present in the training data

# Load your data from the CSV file
df = pd.read_csv('app/data/movie_training_data.csv')

# Drop the 'Movie' column as we don't need it for training
df = df.drop(columns=['Movie'])

# Code to create unknown_data with the right number of columns
unknown_data = pd.DataFrame([['Unknown', 'Unknown', 'Unknown']], columns=['Writer', 'Actor', 'Genre'])
df = pd.concat([df, unknown_data], ignore_index=True)

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_features = encoder.fit_transform(df[['Writer', 'Actor', 'Genre']])
feature_names = encoder.get_feature_names_out(['Writer', 'Actor', 'Genre'])
encoded_df = pd.DataFrame(encoded_features, columns=feature_names)

# Prepare the features and target
X = encoded_df
y = df['Opinion'].apply(lambda x: 1 if x == 'like' else 0)  # Convert 'like'/'dislike' to binary

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# New movie details
new_movie = {
    'Writer': '[Writer Name]',  # Replace with the new movie's writer
    'Actor': '[Actor Name]',    # Replace with the new movie's main actor
    'Genre': '[Genre]'     # Replace with the new movie's genre
}

new_movie_df = pd.DataFrame([new_movie])
new_movie_encoded = encoder.transform(new_movie_df)
new_movie_encoded_df = pd.DataFrame(new_movie_encoded, columns=feature_names)

# Predict the opinion for the new movie
new_movie_pred = clf.predict(new_movie_encoded_df)

# Convert prediction from 0/1 back to 'dislike'/'like'
new_movie_opinion = 'like' if new_movie_pred[0] == 1 else 'dislike'

print(f"Predicted opinion for the new movie: {new_movie_opinion}")