## Telecom Customer Churn Prediction

### Churn
- Customer churn, also known as customer attrition, occurs when customers stop doing business with a company. <br>
- The companies are interested in identifying segments of these customers because the price for acquiring a new customer is usually higher than retaining the old one.

### Context
- "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."

### Content

- [The Orange Telecom's Churn Dataset](https://www.kaggle.com/mnassrib/telecom-churn-datasets), which consists of cleaned customer activity data (features), along with a churn label specifying whether a customer canceled the subscription, will be used to develop predictive models. 
Two datasets are made available here: The churn-80 and churn-20 datasets can be downloaded.

- The two sets are from the same batch, but have been split by an 80/20 ratio.
As more data is often desirable for developing ML models, let's use the larger set (that is, churn-80) for training and cross-validation purposes, and the smaller set (that is, churn-20) for final testing and model performance evaluation.

### Machine Learning Techniques

1. **Logistic Regression**<br>
* When working with our data that accumulates to a binary separation, we want to classify our observations as the customer “will churn” or “won’t churn” from the platform.<br>
* A logistic regression model will try to guess the probability of belonging to one group or another.<br>
* The logistic regression is essentially an extension of a linear regression, only the predicted outcome value is between [0, 1]. <br>
* The model will identify relationships between our target feature, Churn, and our remaining features to apply probabilistic calculations for determining which class the customer should belong to.<br>


2. **Gradient Boosting Classifier**<br>
- GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 
- In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 
- Binary classification is a special case where only a single regression tree is induced.

### Key Drivers in Churn
- Using Gradient Boosting Classifier Feature Importances are caluclated which indicate the major facots leading Customer Churn.
- Business can improve the retention rate on those lines<br>


![Key Drivers in Churn](https://github.com/adharangaonkar/DataScience_Projects/blob/master/Telecom%20Churn%20Prediction/images/key_drivers.png)
