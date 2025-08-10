# Automated-Sentiment-Prediction-for-Swiggy-Reviews-via-Recurrent-Neural-Networks
The Swiggy reviews dataset is processed by cleaning the text (lowercasing, removing special characters) and creating a binary sentiment label: 1 for positive (Avg Rating &gt; 3.5), 0 for negative.

Text data is converted to sequences of integers using a tokenizer and padded so that all reviews have the same length.

The data is split into train/validation/test sets.

The core model is a simple RNN built with TensorFlow/Keras: an embedding layer maps words to vectors, an RNN layer captures sequential context, and a dense output layer with sigmoid activation predicts sentiment.

The model is trained to predict sentiment labels, and its performance is measured by accuracy on the test set.

A Python function is provided to predict the sentiment of any new review text using the trained model, returning either "Positive" or "Negative" with a probability score.
