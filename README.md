# Ted-Talks-Views-Prediction

## PROBLEM STATEMENT
Our objective is to predict the views of a TED talk uploaded on the TEDx website. For this
purpose; we have been provided with the following attributes which are to be used to
predict the views on a particular video.

* Information about the Speakers
* Underlying topics of the video’s talk
* Type of Event in which the video was recorded
* Recorded and Published Date of the video
* Native language of the video and languages in which video is available
* Comments, Duration and web-address of the video
* Related talks
* Description and Transcript of the video

## DATA ACQUISITION
The first step in the pipeline aims at importing our dataset into our environment. In our
case; we will import the dataset containing information about 4005 unique TED talks.
Each TED talk has 18 features to predict the views.

## EXPLORATORY DATA ANALYSIS

* We observe that we have 4005 entries in our dataset and some of the columns
contain null values.
* It is checked that there are no duplicate rows in our dataset which means that
we have data on 4005 unique TED talks.
* Columns containing information about speakers and languages in form of
dictionaries and lists are input into the dataset as strings.
* ‘Transcript’ and ‘Description’ contain large amount of textual data.
* The following table describes the numerical columns in the dataset.

### UNIVARIATE ANALYSIS

Following are some important observations from univariate analysis:
![alt text](https://github.com/fahadmehfooz/Ted-Talks-Views-Prediction/blob/main/images/Univariate.png)


### BIVARIATE ANALYSIS

Following are some important observations from bivariate analysis:
![alt text](https://github.com/fahadmehfooz/Ted-Talks-Views-Prediction/blob/main/images/Bivariate.png)

##  DATA CLEANING

* On analyzing the numerical features in our dataset; we found outliers in most of the
features.

## TRAIN TEST SPLIT
After cleaning our data; the dataset is split into Train - Test datasets. This is done to ensure
that our test dataset is completely isolated and there is no information leakage during the
training process of machine learning models.

## DATA PREPROCESSING AND FEATURE ENGINEERING

In this stage, we are creating two types of features:
1) Numerical Features
2) Numerical Word2Vector embedded feature vectors from corpus

For numerical features:

* Features like all_speakers, occupations, about_speakers are first filled with with a
value as ‘others’ where Nan was present. After that, these features are converted to a
dictionary representation from a string of dictionary representation.
* Features like published_date and recorded_date are converted to datetime.
* Total_days_since_published is also created using published_date and recored_date.
* New features like day, month, year and week_day are created using datetime
objects created in previous step.
* More features like speaker_1_average_views, topic_wise_average_views,
unique_topics and event_average_views are introduced in the data by
grouping data according to speaker_1, topics, and event respectively.

For numerical word2Vector embedded features:

* Used Google’s pre-trained word2vec model.
* Created a corpus using transcript feature.
* Cleaned the corpus by removing stop-words, digits, punctuation marks, etc.
* After cleaning the corpus, feature_vectors are created with dimension as 300.

## DATA MODELLING

Many models were trained, from simple parametric models like Linear Regression to
tree based models. It was observed that linear regression did not performed up to the
mark and tree based models generally outperformed them. The gist of top performing
regression models is given.

1). LGBM Regressor - It is a boosting technique that uses tree based learning algorithm. It
grows tree leaf wise rather than level wise.

2). XGBoost Regressor - It is also a boosting technique that uses gradient descent algorithm
to minimize the loss when adding new tree models.

3). Stacked Regressor - CatBoost, LGBM, RandomForest and XGBoost were chosen as the
base models for the Stacked Regressor and LGBM as the final one. It tried to combine
them in such a way that scores increase.

4). Voting Regressor - The estimators used for voting were the same ones being used in
Stacking.

## RESULTS OF DATA MODELLING

LGBM is chosen as the final model for our regression problem owing to best test
result and close Train and Test R2-scores.

## CONCLUSION

* The main objective was to build a predictive model, which could help in
predicting the views of the videos uploaded on the TEDx website.
* We have built a model where it is able to predict what views next TEDx video
would get. The model was able to predict with an R2 score of 0.861 on the test
data.
* Model Interpretation also shows how each feature contributes to the predicted
views.
* We built a baseline model and then improved on that.
* We have done modelling using: LGBM, XGBOOST, stacked model, voting model,
RandomForest and CATBOOST.
* We have used different error metrics like MAE, MSE and RMSE. The errors were
minimum for LGBM.
* Hyper-parameter tuning helped us to get rid of overfitting.
* We interpreted the model using SHAP.

## CHALLENGES FACED
* Creating a quality corpus was difficult here. We tried with multiple features like
description, transcript, etc. for how many features should be taken.
* We tried TF-IDF, countvectorizer, and word2vec using gensim but weren't able
to create a good corpus which was fitting nicely to data.
* Features were created using a mix of feature vectors from the corpus and
numerical columns.
* We also tried topic modelling here but again it did not give good results.
* A lot of feature engineering was required.
* Because of a large number of features, we were facing some overfitting. So,
reaching to an optimal model was challenging.

## FUTURE SCOPE OF WORK

* We would want to build a quality corpus using most of the textual features here.
* Time features are available, we could also try time series modelling.
* Since the data has textual features and as sequence is important in text, we
could also try a BiDirectional LSTM as it could give good results.
* Creating Application and Model Deployment.
* Various Other regressors can be used for this problem.
