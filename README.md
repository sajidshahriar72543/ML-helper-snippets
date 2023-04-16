### Mount gdrive
```py
from google.colab import drive
drive.mount('/content/drive')
dataset_name = pd.read_csv('path/to/file')
df = dataset_name
```
### Export and Download datasets 
```py
from google.colab import files
df.to_csv('df_UPDATED.csv')
files.download('df_UPDATED.csv')
```
### Null columns
```py
df.isnull().sum()

# manual function
# def get_nulls(df):
#     dict_nulls = {}
#     for col in  df.columns:
#         dict_nulls[col]=df[col].isnull().sum()

#     df_nulls = pd.DataFrame(data=list(dict_nulls.values()), 
#                             index=list(dict_nulls.keys()), 
#                             columns=['#nulls'])
#     return df_nulls

# get_nulls(df)
```
### Percentage of Null values
```py
# Approach 1
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

percent_missing

# Approach 2


def get_nulls_percentage(df):    
    dict_nulls = {}
    for col in  df.columns:
        percentage_null_values = str(round(df[col].isnull().sum()/len(df),2))+\
        "%"
        dict_nulls[col] = percentage_null_values
    
    df_nulls = pd.DataFrame(data=list(dict_nulls.values()), 
                            index=list(dict_nulls.keys()), 
                            columns=['% nulls'])
    return df_nulls
    
get_nulls_percentage(df)
    
# Aproach 3
percent_missing = df.isnull().sum() * 100 / len(df)

percent_missing

# Aproach 4
df.isnull().mean() * 100
```
### Heatmap checking
```py
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
```
### Dataset information
```py
df.info()
```
### Check Categorical variables
```py
print("Categorical Variables")
cat_var = df.select_dtypes(include="object").columns
for col in cat_var:
  print(col)
```
### Replace null values with mode (common entry) for catagorical variables
```py
for cat_col in cat_var:
  df[cat_col] = df[cat_col].fillna(df[cat_col].mode()[0])

for cat_col in cat_var:
  print(cat_col, df[cat_col].isnull().sum())
```
### Check Numerical variables
```py
print("Numerical Variables")
num_var = train._get_numeric_data().columns
for col in num_var:
  print(col)
```
### Replace null values with mean for numerical variables
```py
for num_col in num_var:
    df[num_col] = df[num_col].fillna(df[num_col].mean())

for num_col in num_var:
  print(num_col, df[num_col].isnull().sum())
```
### Dropping  null valued columns
```py
df.drop(['Column_name'],axis=1,inplace=True)
```
### Function to remove unique valued columns
```py
# defining the function
def remove_distinct_value_features(df):
    return [e for e in df.columns if df[e].nunique() == df.shape[0]]

# calling the function
drop_col = remove_distinct_value_features(df)
drop_col

# updating the dataset
cols = [e for e in df.columns if e not in drop_col]
df = df[cols]
df
```
### Remove duplicated columns
```py
df =df.loc[:,~df.columns.duplicated()]
```
### Concat datasets
```py
concat_df=pd.concat([first_df,second_df],axis=0)

# second_df is the df that will be concatenated with the first 
```
### Split a dataset
```py
# till_record = num of record till the split

df_Train=final_df.iloc[:till_record,:]
df_Test=final_df.iloc[till_record:,:]
```
### Fix `KNeighborsClassifier does not accept missing values encoded as NaN natively` error
The KNeighborsClassifier in scikit-learn does not accept missing values (encoded as NaN) natively. If your dataset has missing values, you need to handle them before fitting the classifier.

One common way to handle missing values is to impute them, which means replacing missing values with some other value. There are several strategies for imputation, such as replacing missing values with the mean, median, or mode of the non-missing values in the same feature.

Here is an example of how you can use scikit-learn's SimpleImputer class to impute missing values with the mean:
```py
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)

# Create a SimpleImputer object to impute missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform both the training and test data using the fitted imputer
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a KNeighborsClassifier object and fit it on the imputed training data
knn = KNeighborsClassifier()
knn.fit(X_train_imputed, y_train)

# Evaluate the classifier on the imputed test data
score = knn.score(X_test_imputed, y_test)
print(score)
```
In this example, the SimpleImputer class is used to impute missing values with the mean. The fit method is called on the imputer object to fit it on the training data, and the transform method is called on both the training and test data to transform them using the fitted imputer. The KNeighborsClassifier object is then fit on the imputed training data, and the score method is used to evaluate the classifier on the imputed test data.

Note that there are other strategies for imputing missing values, and the choice of strategy may depend on the specific characteristics of your dataset. <br>
***Source: chatGPT***