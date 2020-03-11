# Assignment 1: Spam filtering using a naive Bayes classifier
The full assignment description is available [here](https://nld.ict.usc.edu/cs544-spring2020/assignment1/CS544-assignment1.pdf) as a PDF. Below is an explanation of what each file does:

* nblearn.py: learns the naive Bayes model from labeled data
* nbclassify.py: uses the model on new data
* nbevaluate.py: outputs precision, recall, and F1 scores

Part of our assignment was also to get experimental with our classifier and see if we could yield improvements in precision, recall, and F1 score metrics:
* nbclassify-only-lowercase: experiment was to convert everything to lowercase, and see if case-insensitive classification worked better (it was slightly worse)
* nblearn-only-alpha-tokens: experiment was to only consider alphabetical tokens, and ignore numerical tokens and special characters, and see if this worked better (surprisingly, it was much worse)
