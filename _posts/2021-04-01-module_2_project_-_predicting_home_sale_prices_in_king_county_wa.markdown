---
layout: post
title:      "Module 2 Project - Predicting Home Sale Prices in King County, WA"
date:       2021-04-01 14:22:16 -0400
permalink:  module_2_project_-_predicting_home_sale_prices_in_king_county_wa
---


This project involved creating a linear regression model to predict sale prices for homes in King County, WA bases on a 2014-2015 dataset.  From the get-go I imported every library I could think of that might be useful in a regression model, mostly from scikit-learn.  I imported my DataFrame and gave it an overview, going over in my head what I might want to keep and discard.

The dataset consisted of a mixure of continuous variables, many related to square footage, and categorical variables mainly identifying information about the home (bathroom, bedrooms, condition, etc.)  NaN values weren't too much of an issue, however I did spot a few columns that could benefit from engineering.

```
#change yr_renovated to categorical variable
df['yr_renovated'].fillna(0, inplace=True)
df['yr_renovated'] = np.where(df['yr_renovated'].between(2006,2015), 3, df['yr_renovated']).astype(int)
df['yr_renovated'] = np.where(df['yr_renovated'].between(1996,2005), 2, df['yr_renovated']).astype(int)
df['yr_renovated'] = np.where(df['yr_renovated'].between(1986,1995), 1, df['yr_renovated']).astype(int)
df['yr_renovated'] = np.where(df['yr_renovated'].between(1934,1985), 0, df['yr_renovated']).astype(int)
df.rename(columns={'yr_renovated': 'renovated'}, inplace=True)
```

The existing 'yr_renovated' column listed either a year or a 0, and I decided to categorize these based on decade of renovation.  Pre-1985 renovations didn't have too much of an effect on sale prices, and so these could be categorized with the non-renovated homes.  While the vast majority of homes ended up with a 0, the effects of any renovation at all were relevant with final sale price.

<img src="https://github.com/JonahFlateman/dsc-mod-2-project-v2-1-online-ds-sp-000/blob/master/renovated.png"  alt="Renovation vs. Sale Price" width=600>

Using regplots to check for linearity helped identify interior square footage, or 'sqft_living' as a reliable continuous variable.

<img src="https://github.com/JonahFlateman/dsc-mod-2-project-v2-1-online-ds-sp-000/blob/master/sqft_living.png"  alt='Interior Sqare Footage vs. Sale Price" width=600>

After checking for multicollinearity, I dropped all other square footage columns.  Keeping most other categorical variables, I engineered 'zipcode_category' by taking the mean of each price per ZIP and splitting each home into four quantiles.

```
#Creating unique dataframe 
df_zip = pd.concat([frequency, mean], axis=1)
df_zip['zipcode'] = df_zip.index
df_zip.columns = ['frequency', 'price', 'zipcode']

#Use function to split by quantile and apply to original dataframe
first_quantile = df_zip[df_zip.price < 347892.25]    
second_quantile = df_zip[df_zip.price.between(347892.25, 475155.5)] 
third_quantile = df_zip[df_zip.price.between(475155.5, 633188.00)] 
fourth_quantile = df_zip[df_zip.price > 633188.00]

def zipcode_category(zipcode):
    if zipcode in first_quantile.index:
        return 1
    elif zipcode in second_quantile.index:
        return 2
    elif zipcode in third_quantile.index:
        return 3
    else:
        return 4
				
df['zipcode_category'] = df.zipcode.apply(zipcode_category).astype(int)
```

Ready to begin modeling, I used log transformations on my two remaining continuous variables, 'sqft_living' and 'price,' and created dummy variables for the other categoricals.  Using statsmodels to run an OLS regression model gave me an R-squared of 0.783, however a good number of my predictors still had p-values greater than 0.05.  Using stepwise selection, I called a function which eliminated these high p-values and then used it to re-run the model using only the remaining predictors.

```
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = included[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
		
X = df_final.drop('price', axis=1)
y = df_final['price']

result = stepwise_selection(X, y, verbose = True)
print('resulting features:')
print(result)
```

Our R-squared was now 0.782 - not an increase but I could now be sure that only accurate predictors were being used.  The last step was to validate the model by creating our train-test splits.

```
#Creating training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_with_intercept, y)
print(len(X_train), len(X_test), len(y_train), len(y_test))

#Fitting the model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
```

And calculating our mean squared error, using cross-validation to produce five results.

```
#Calculating mean squared errors
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)

#Cross validating the mse
mse = make_scorer(mean_squared_error)
cv_5_results = cross_val_score(linreg, X_with_intercept, y, cv=5, scoring=mse)
print(cv_5_results)
print(cv_5_results.mean())
```

The training and testing MSEs were 21.5 and 22.6, respectively.  The MSE cross-validation produced a mean of 21.9.  Though I was satisfied with this model, some things that could be fine-tuned running it in the future involve possibly taking another look at square footage and seeing what effects our other variables (lot footage, basement footage) would possible have.  The dataset consisted of 2014-2015 only and a larger set (5, 10, 20 years) could show us some interesting trends and force us to re-think our predictors.



