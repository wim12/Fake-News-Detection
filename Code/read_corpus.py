from dask import dataframe as dd
import dask.dataframe as dd
from slugify import slugify
import pandas as pd

df = dd.read_csv(r'E:\Users\Will\Desktop\Fake news Detection\test\news_cleaned_2018_02_13.csv', engine='python' , encoding="utf8",
                 error_bad_lines=False, dtype={'Unnamed: 0': 'object',
                                               'id': 'object',
                                               'keywords': 'object',
                                               'source': 'object',
                                               'summary': 'object'})

print(df.shape)
df = df.set_index("Unnamed: 0")
print(df.head())

df.drop(columns=['url','scraped_at','inserted_at','updated_at','authors','keywords','meta_keywords','meta_description','tags','summary','source'], axis=1)
#df.drop(columns='scraped_at', axis=1)
#df.drop(columns='inserted_at', axis=1)
#df.drop(columns='updated_at', axis=1)
#df.drop(columns='authors', axis=1)
#df.drop(columns='keywords', axis=1)
#df.drop(columns='meta_keywords', axis=1)
#df.drop(columns='meta_description', axis=1)
#df.drop(columns='tags', axis=1)
#df.drop(columns='summary', axis=1)
#df.drop(columns='source', axis=1)

df.to_csv('corpusoutput*.csv', header=False, index=False)
