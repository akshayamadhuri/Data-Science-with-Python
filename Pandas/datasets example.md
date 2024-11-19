## Exploring Practical Application of Pandas: Titanic Dataset from Seaborn

Let's dive in and see how Pandas can be applied to real-life datasets! To show this, we will use the Titanic dataset provided by the Seaborn library and show you some quick examples of how you can start analyzing it using Pandas.

Seaborn provides a direct function to load the dataset, making it very easy to load the dataset into the Pandas DataFrame:

```Python

import pandas as pd
import seaborn as sns

# Load the titanic dataset into a Pandas DataFrame
titanic = sns.load_dataset('titanic')

# Look at the first 3 rows of the DataFrame
print(titanic.head(3))

"""
   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True

[3 rows x 15 columns]
"""
```
As titanic is just a Pandas DataFrame, you can apply to it any operations we've learned before!
