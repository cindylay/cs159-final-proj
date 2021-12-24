# Stack Overflow Question Quality Classification Using Deep Learning Techniques
## Abstract
Community Question Answering (CQA) forums like Stack Overflow play an important role to support developers of all experience levels. Thus, it is essential to establish an automatic quality control metric to filter high-quality questions better than current manual moderation methods.  

In this research, we apply different natural language processing and deep learning techniques to classify high-quality questions based on linguistic features and assigned tags. Using random forests, we evaluate question features most influential to the quality of the posts. In accordance with our findings, we conclude that an approach that combines deep learning and natural language processing methods serves as an accurate solution to the automated quality classification problem for Stack Overflow. We found that bi-directional LSTM and CNN had higher accuracies than BERT although BERT had higher precision and recall. Furthermore, we found that when evaluating the dataset using sentiment analysis, Neural Network Classifcation had an accuracy of about 46\% while our Random Forest Classifier had an accuracy of about 51\% and found tags to be the most influential feature to predicting post quality.
## Usage:
1. Data visualization: [Experiments/display-data.ipynb](https://github.com/cindylay/cs159-final-proj/blob/main/Experiments/display-data.ipynb)
2. Text classification using BERT: [Experiments/fine_tuning_bert_text_classification.ipynb](https://github.com/cindylay/cs159-final-proj/blob/main/Experiments/fine_tuning_bert_text_classification.ipynb)
3. Text classification using Bi-directional LSTM and CNN: [Experiments/text_classification__comaprison.ipynb](https://github.com/cindylay/cs159-final-proj/blob/main/Experiments/text_classification__comaprison.ipynb)
4. Neural Net and Random Forest Text Classification with Feature Importances Ranking: [Experiments/sentiment_analysis.ipynb](https://github.com/cindylay/cs159-final-proj/blob/main/Experiments/sentiment_analysis.ipynb)
## Research Paper:
The research paper is included within this GitHub Repository titled: *Stack Overflow Question Quality Classification*. Click [here](https://github.com/cindylay/cs159-final-proj/blob/main/Stack%20Overflow%20Question%20Quality%20Classification.pdf) to access the PDF.
## Repository Structure
```
.
├── BERT_DATA_DIR <- Contains the train and valid setes for the Stack Overflow questions dataset
│   └── train.csv
│   └── valid.csv
├── Experiments <- Contains all of the experiments (reference methods section)
│   └── display-data.ipynb <- Dataset analysis
│   └── fine_tuning_bert_text_classification.ipynb <- BERT Classification Model
│   └── sentiment_analysis.ipynb <- Sentiment Analysis Scoring and Tags Based Approach
│   └── text_classification__comaprison.ipynb <- Bi-directional LSTM Classification
├── .gitignore 
├── README.md
└── Stack Overflow Question Quality Classification.pdf <- The research paper
Model / CNN Classification Model
```
## Methods
To best determine question quality we decided on two different methods. Our first approach was text classification using Bi-directional Encoder Representation from Transformers (BERT), as well as Bi-directional Long-Short Term Memory (BLSTM) and Convolutional Neural Networks (CNN). Our second method incorporates sentiment analysis on text attributes of the data to transform the data into numerical categories such that neural net classification and Random forest classification can be applied. Furthermore, through the second approach, Random Forests will be used to generate a ranking for dataset attributes based on their feature importance. Comparing these two approaches, we investigate which most effectively classifies high-quality stack overflow questions.
### Linguistic Characteristics Based Approach
For our first approach, for the deep learning approaches to be compared fairly, we set up the models so that the number of trainable parameters is close to each other. The following is the models studied in this project and how they were set up:

1. BERT: Using our pre-processed dataset, we leverage that each post is linked to a range of post qualities. Since each row of posts holds a different form from the text source, we need to clean each part of the data to apply a proper <start> and <end> portion to note the post text. Importing version two of the pre-trained uncased BERT Model on [TensorFlow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2), we tokenized the words using the official TensorFlow BERT [model asset](https://github.com/tensorflow/models/tree/master/official/nlp/bert). Provided that each line of the dataset is composed of the raw body text and its label, we process the text to BERT input features: Input Word Ids, Input Masks, Segment Ids. The output of BERT for our classification task will be a pooled output of shape [batch\_size, 768] with representations for the entire input sequences. Configuration parameters: maximum length of input sequences is 150 tokens, training batch size of 32 samples, and an adam optimizer with a learning rate of 2e-5. Although we wanted to increase the maximum input sequence length to match the others, BERT without training has over 110 million parameters, resulting in our lack of memory resources.

2. Bi-directional LSTM: There are two bi-directional LSTM layers stacked and the model consists of an Embedding layer as its input. The LSTM layers use around 64 hidden neurons whereas the first LSTM layers return a sequence that can be directly fed into the second layer. The final layer is a dense layer using a soft-max activation function to ensure that the output is in a probabilistic format. Configuration parameters: Adam optimizer with a learning rate of 1e-4, training batch size of 32 samples, the maximum length of input sequences is 360 tokens. The learning rate is reduced depending on the progress of the validation loss.

3. CNN: Consists of a single convolutional layer. The input layer contains an embedding layer and has the same properties as the previous one. The pre-processed data is flattened, resulting in a similar dense layer to the Bi-directional LSTM final layer. Configuration parameters: Adam optimizer with a learning rate of 1e-4, training batch size of 32 samples, the maximum length of input sequences is 360 tokens. The learning rate is reduced depending on the progress of the validation loss.
### Sentiment Analysis Scoring and Tags Based Approach

Using our pre-processed dataset, we first convert the dataframe to a numpy float64 array. Then, the data is permutated such that different data values exist in the training and testing sets for each iteration. 

1. Neural Net Classification: To keep the feature values in the -1 to 1 range, we standardize feature values by removing the mean and scaling to unit variance. Using the [multi-layer perceptron classifier provided by sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), we train the classifier with the following configuration: a hidden layer size of (9,9), a hyperbolic tan activation function for the hidden layer, the stochastic gradient descent solver for weight optimization, and a constant learning rate of 0.1. 

2. Random Forest Classification: Picking random data points from our training set, we build a decision tree associated with these data points. To optimize the performance of the model, cross-validation is used to split the training set into model-building and model-validation subsets. Test different numbers of decision trees and depths by iterating through the number of decision trees between 50 and 300, and a depth between 1 and 20. After establishing the optimal number of decision trees and the depth, re-build the model and test the model against the test set. Then, using the [feature_importances attribute](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) of the Random Forest classifier, determine which features contribute most to the quality classification task. 
### Evaluation Metrics
For any deep learning model, achieving a 'good fit' on the model is crucial. To evaluate the performances of each of the models, we will be using three statistical metrics: accuracy, precision, and recall. In our dataset, given that each quality label is of equal importance, we believe classification accuracy is the most effective. Other statistical measures we are considering are precision and recall as precision allows us to identify a measure of result relevancy, while recall allows us to measure the number of truly relevant results that the model returns.

## Results/ Discussions / Conclusion
Please reference the research paper[research paper](https://github.com/cindylay/cs159-final-proj/blob/main/Stack%20Overflow%20Question%20Quality%20Classification.pdf) 