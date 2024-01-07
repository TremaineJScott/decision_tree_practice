# Movie Decision Tree Classifier

## Overview

The Movie Decision Tree Classifier is a machine learning project designed to showcase the application of decision trees in the realm of predictive analytics. This project demonstrates how a decision tree model can be trained to predict individual movie preferences based on specific attributes: the writer, the main actor, and the movie's genre. It serves as a practical example for those interested in understanding and developing AI models for recommendation systems.

## Objective

The primary goal of this project is to provide a clear example of how a decision tree can be used to make predictions based on categorical data. The decision tree classifier is a popular machine learning algorithm due to its interpretability and simplicity. It models decision rules inferred from the data features, making it an excellent tool for understanding the decision-making process.

## Dataset

The dataset used for this project consists of a collection of movies with associated writers, actors, genres, and user opinions (like/dislike). This information forms the basis for training our model.

## Project Structure

- `app/`:
  - `data/`: This directory contains the training dataset in CSV format.
  - `main.py`: The Python script that includes data preprocessing, model training, and a prediction routine.

## Model Training and Evaluation

The decision tree classifier is trained on the provided dataset using scikit-learn's `DecisionTreeClassifier`. The model undergoes evaluation using a portion of the dataset reserved for testing, providing an accuracy score that reflects the model's performance.

## Intended Use

This project is intended as an educational tool to demonstrate the construction and application of a decision tree model. It is especially suited for those learning about AI and machine learning, as well as practitioners looking to explore the use of decision trees in their projects.

## Contributions and Feedback

We welcome contributions and feedback on this project. If you have suggestions for improvement or have found an issue, please open an issue or submit a pull request on the repository.

## License

This project is licensed under the MIT License 
