---
layout: post
title:      "Churn in Telecom's Dataset: The Right Questions"
date:       2021-05-27 14:10:27 -0400
permalink:  churn_in_telecoms_dataset_the_right_questions
---


My task was a popular one on Kaggle - build a classifier to predict churn, whether or not a customer will soon cancel their service.  The dataset provided some basic information on each account  such as state, telephone number, number of calls/charges and their times of day, and number of customer service calls made from each account.  Right away I wanted to know the overall churn rate to get started.

```
churn_rate = sum(df['churn']) / len(df['churn'])

```

At 14.5%, this exceeded industry standard acceptable rates of 5-7%, so it would be important to not only build as accurate a classifier as possible, but also understand which features were contributing the most to churn rate.  Some EDA showed high numbers of customer service calls highly correlated to churn, while having an international plan seemed to keep customers loyal.  The dataset was quite clean, so after some light feature engineering I built a baseline model using a Random Forest Classifier and GridSearchCV to help find its best hyperparameters.

```
#Creating predictor and target variables
X = new_df.drop(columns=['churn'], axis=1)
y = new_df.churn

RF = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,  random_state=42)

#Setting GridSearch parameters
grid_param = {'n_estimators': [50,100], 
              'class_weight': ['balanced'], 
              'criterion': ['gini', 'entropy'], 
              'max_depth': [2,4,6], 
              'min_samples_split': [2, 5, 10], 
              'min_samples_leaf': [1, 3, 5]
              } 
							
#GridSearch fit
grid_search = GridSearchCV(RF, grid_param, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)
```

The idea was to focus on recall, that is to minimize false negatives or our model falsely predicting a customer will not churn.  The reverse would be inconvenient but not nearly as consequential from a business perspective.  To that end I relied heavily on sklearn.metrics packages such as precision and recall scores and confusion matrices.

<img src="https://github.com/JonahFlateman/dsc-mod-3-project-v2-1-online-ds-sp-000/blob/master/image/confusionmatrix.png?raw=true">

I was fairly pleased with the above confusion matrix on the test set and my metrics gave me a recall score of approximately 0.85.  After trying out a few more models using Logistic Regression, K-Nearest Neighbors, and AdaBoost, it was Gradient Boosting that produced numbers similar to my Random Forest model, so I proceeded to focus only on those two for the rest of the process.

I decided to bring back two previously-deleted features, state and area code, and one-hot encode each for the model's next iterations.  This kept recall the same but increased precision for the Random Forest model, and like the previous iteration this model seemed to have the best fit.  Using SMOTE to reduce class imbalance and a GridSearch to try and tune the hyperparameters for my Gradient Boosting Classifier seemed to work too well:

```
X = new_df.drop(columns=['churn'], axis=1)
y = new_df.churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,  random_state=42 )
smote = SMOTE()
gbt_clf = GradientBoostingClassifier()
X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train)
gbt_clf.fit(X_train_resampled, y_train_resampled)

pipeline = make_pipeline(smote, gbt_clf)

pipeline.fit(X_train_resampled, y_train_resampled)

grid_param = { 'gradientboostingclassifier__max_depth': [6], 
              'gradientboostingclassifier__min_samples_split': [4], 
              'gradientboostingclassifier__min_samples_leaf': [1],
              'gradientboostingclassifier__n_estimators': [200]
              } 
grid_search = GridSearchCV(pipeline, grid_param, cv=3, scoring=make_scorer(recall_score), verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
```

It should be noted that these grid parameters were entered manually after several search runs.  Being that GridSearch can take some time depending on how many parameters are entered, I wanted my notebook to be able to be run fairly quickly.  This fit produced no false positives or negatives in the training set and a recall score of 0.83 in the test set - not bad but the model was overfit.  Still, this along with the Random Forest model could be used with some additional tuning and depending on the business problem at hand.  Let's say we chose the Random Forest model and wanted to make some business recommendations:

<img src="https://github.com/JonahFlateman/dsc-mod-3-project-v2-1-online-ds-sp-000/blob/master/image/featureimportance.png?raw=true">

<img src="https://github.com/JonahFlateman/dsc-mod-3-project-v2-1-online-ds-sp-000/blob/master/image/customerchurnbar.png?raw=true">

As customer service calls leads here in terms of importance to our model, we can see from the bar graph that four or more calls highly increases probability of churn.  We could make sure to place special attention on certain accounts which are approaching four calls to better understand the needs of these customers.  This could be done in conjunction with a reduction in day rates or new offers for plans specific to daytime minutes use - as these are highly correlated to churn rate, we would need to balance the value of the individual accounts with any special offers we make to these customers.

Lastly, to help decrypt the numbers into some practical terms we used LIME (Local Interpretable Model-agnostic Explanations) to produce easy-to-understand tables:



```
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['no churn', 'churn'],
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=X_test.iloc[2], 
    predict_fn=best_estimator.predict_proba
)

exp.show_in_notebook(show_table=True)
```

<img src="https://github.com/JonahFlateman/dsc-mod-3-project-v2-1-online-ds-sp-000/blob/master/image/limetable.png?raw=true">

This shows us the prediction probabilities, the most important features, and the values of the top variables.  In this case we used the Gradient Boosting Classifier since it did not include any dummy variables which are less useful for a table such as this.
