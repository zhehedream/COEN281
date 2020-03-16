#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Data/pseudo_facebook.csv")


# In[3]:


df.tail()


# In[4]:


categorical = ('gender')


# In[5]:


df['gender'] = df['gender'].astype('category')


# In[6]:


def get_spans(df, partition, scale=None):
    """
    :param        df: the dataframe for which to calculate the spans
    :param partition: the partition for which to calculate the spans
    :param     scale: if given, the spans of each column will be divided
                      by the value in `scale` for that column
    :        returns: The spans of all columns in the partition
    """
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans


# In[7]:


full_spans = get_spans(df, df.index)
full_spans


# In[8]:


def split(df, partition, column):
    """
    :param        df: The dataframe to split
    :param partition: The partition to split
    :param    column: The column along which to split
    :        returns: A tuple containing a split of the original partition
    """
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


# In[9]:


def is_k_anonymous(df, partition, sensitive_column, k=3):
    """
    :param               df: The dataframe on which to check the partition.
    :param        partition: The partition of the dataframe to check.
    :param sensitive_column: The name of the sensitive column
    :param                k: The desired k
    :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
    """
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    """
    :param               df: The dataframe to be partitioned.
    :param  feature_columns: A list of column names along which to partition the dataset.
    :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
    :param            scale: The column spans as generated before.
    :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
    :returns               : A list of valid partitions that cover the entire dataframe.
    """
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


# In[10]:


# we apply our partitioning method to two columns of our dataset, using "income" as the sensitive attribute
feature_columns = ['friend_count', 'likes']
sensitive_column = 'age'
finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)


# In[11]:


# we get the number of partitions that were created
len(finished_partitions)


# In[12]:


def agg_categorical_column(series):
    return [','.join(set(series))]

def agg_numerical_column(series):
    return [series.mean()]


# In[13]:


def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)


# In[14]:


dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)


# In[15]:


# we sort the resulting dataframe using the feature columns and the sensitive attribute
k_res=dfn.sort_values(feature_columns+[sensitive_column])
k_res


# In[16]:


k_hidden_failure=0
for index, row in k_res.iterrows():
    if row['count']<3:
        k_hidden_failure=k_hidden_failure+row['count']
print(k_hidden_failure)


# In[40]:


import pandas as pd
ori = pd.read_csv("Data/pseudo_facebook.csv")
diff_out = pd.read_csv("Data/diff_out.csv")
diff_out


# In[18]:


def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


# In[19]:


diff_df = dataframe_difference(ori, diff_out, 'both')
d_hidden_failure=0
for index, row in diff_df.iterrows():
    d_hidden_failure=d_hidden_failure+1
print(d_hidden_failure)


# In[20]:


dict = {}
i = 0
for index, row in k_res.iterrows():
    for x in range(int(row['count'])):
        dict[i] = {'friend_count' : row['friend_count'] , 'likes' : row['likes'], 'age' : row['age']} 
        i = i + 1
k_df = pd.DataFrame.from_dict(dict, "index")
k_df


# In[31]:


from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
 eps = .2, 
 metric="euclidean", 
 min_samples = 3,
 n_jobs = -1)
o_normal = 0
o_total = 0
d_res=ori
d_res = d_res.drop(["dob_day", "dob_year", "dob_month"
                   , "gender", "tenure", "friendships_initiated"
                   , "likes_received", "mobile_likes", "mobile_likes_received"
                   , "www_likes", "www_likes_received", "userid"], axis=1)
clusters = outlier_detection.fit_predict(d_res)
for i in clusters:
    o_total = o_total + 1
    if i==-1:
        o_normal = o_normal + 1
print((o_total - o_normal) / o_total)
o_total
o_total - o_normal


# In[28]:


from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
 eps = .2, 
 metric="euclidean", 
 min_samples = 3,
 n_jobs = -1)
clusters = outlier_detection.fit_predict(k_df)
k_normal = 0
k_total = 0
for i in clusters:
    k_total = k_total + 1
    if i==-1:
        k_normal = k_normal + 1
print((k_total - k_normal) / k_total)
print(k_total)
print(k_normal)


# In[41]:


d_res=ori
d_res = d_res.drop(["dob_day", "dob_year", "dob_month"
                   , "gender", "tenure", "friendships_initiated"
                   , "likes_received", "mobile_likes", "mobile_likes_received"
                   , "www_likes", "www_likes_received", "userid"], axis=1)
d_res
diff_out = diff_out.drop(["dob_day", "dob_year", "dob_month"
                   , "gender", "tenure", "friendships_initiated"
                   , "likes_received", "mobile_likes", "mobile_likes_received"
                   , "www_likes", "www_likes_received", "userid"], axis=1)
diff_out


# In[37]:


from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
 eps = .2, 
 metric="euclidean", 
 min_samples = 3,
 n_jobs = -1)
clusters = outlier_detection.fit_predict(diff_out)
d_normal = 0
d_total = 0
for i in clusters:
    d_total = d_total + 1
    if i==-1:
        d_normal = d_normal + 1
print((d_total - d_normal) / d_total)
d_total - d_normal


# In[24]:


list = []
for index, row in k_res.iterrows():
    for x in range(int(row['count'])):
        list.append(row['age'])
k_density = pd.Series(list)
ax = k_density.plot.kde()


# In[43]:


list = []
for index, row in d_res.iterrows():
    list.append(row['age'])
d_density = pd.Series(list)
ax = d_density.plot.kde()
d_xy = ax.get_lines()[0].get_xydata()
hist = d_density.hist(bins=10)


# In[26]:


list = []
for index, row in ori.iterrows():
    list.append(row['age'])
o_density = pd.Series(list)
ax = o_density.plot.kde()
o_xy = ax.get_lines()[0].get_xydata()


# In[27]:


step = o_xy[1][0] - o_xy[0][0]
d_ap = 0
for x in range(len(o_xy)):
    d_ap = d_ap + step * abs(o_xy[x][1] - d_xy[x][1])
print(d_ap)


# In[ ]:




