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
