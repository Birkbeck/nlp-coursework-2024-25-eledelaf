# Part 2:Feature Extraction and Classification
"""
Read the hansard40000.csv dataset in the texts directory into a dataframe. 
Sub-set and rename the dataframe as follows:
"""
import pandas as pd

df = pd.read_csv("/Users/elenadelafuente/Desktop/MASTER/2 trimestre/Natural Lenguage Processing/Assesment/p2-texts/hansard40000.csv")
#print(df.head())

"""
Rename the ‘Labour (Co-op)’ value in ‘party’ column to ‘Labour’
"""
df = df.replace({"Labour (Co-op)": "Labour"})

"""
remove any rows where the value of the ‘party’ column is not one of the
four most common party names
"""
parties_4 = list(df['party'].value_counts()[:4].index)
filtered_df = df[df['party'].isin(parties_4)]
filtered_df = filtered_df.drop("speakername", axis=1)
print(filtered_df.head())
