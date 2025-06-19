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
f_df = df[df['party'].isin(parties_4)]
f_df = f_df.drop("speakername", axis=1)
print(f_df.head())

"""
remove any rows where the value in the ‘speech_class’ column is not
‘Speech’.
"""
f_df = f_df.drop(f_df[f_df["speech_class"] != "Speech"].index)

