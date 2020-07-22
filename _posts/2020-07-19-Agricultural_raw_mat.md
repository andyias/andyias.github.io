---
title: "Agricultural Products Sales"
date: 2020-07-19
tags: [data wrangling, data cleaning, data science]
header:
  image: ""
excerpt: "Data Wrangling, Data Cleaning, Data Science"
mathjax: "False"
---
# Agricultural Products Sales
This post is comprised of a data cleaning project. The dataset can be found [here](https://www.kaggle.com/kianwee/agricultural-raw-material-prices-19902020)
This dataset consists of the prices of farm products and their percentage change per month.
The code to this can be found [here](https://github.com/andyias/DataCleaning-Projects/blob/master/Agricultural_raw_mat.ipynb)
First off,lets start  by importing the neccessary libraries for  data analysis(In this case I used Pandas and Numpy).

```python
import numpy as np
import pandas as pd
```
Next is to read the dataset which is stored as a csv file.

```python
# import data
data = pd.read_csv('./data/agricultural_raw_material.csv')
```
The very first inspection of the data.

```python
#inspect the data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Coarse wool Price</th>
      <th>Coarse wool price % Change</th>
      <th>Copra Price</th>
      <th>Copra price % Change</th>
      <th>Cotton Price</th>
      <th>Cotton price % Change</th>
      <th>Fine wool Price</th>
      <th>Fine wool price % Change</th>
      <th>Hard log Price</th>
      <th>...</th>
      <th>Plywood Price</th>
      <th>Plywood price % Change</th>
      <th>Rubber Price</th>
      <th>Rubber price % Change</th>
      <th>Softlog Price</th>
      <th>Softlog price % Change</th>
      <th>Soft sawnwood Price</th>
      <th>Soft sawnwood price % Change</th>
      <th>Wood pulp Price</th>
      <th>Wood pulp price % Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apr-90</td>
      <td>482.34</td>
      <td>-</td>
      <td>236</td>
      <td>-</td>
      <td>1.83</td>
      <td>-</td>
      <td>1,071.63</td>
      <td>-</td>
      <td>161.20</td>
      <td>...</td>
      <td>312.36</td>
      <td>-</td>
      <td>0.84</td>
      <td>-</td>
      <td>120.66</td>
      <td>-</td>
      <td>218.76</td>
      <td>-</td>
      <td>829.29</td>
      <td>-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>May-90</td>
      <td>447.26</td>
      <td>-7.27%</td>
      <td>234</td>
      <td>-0.85%</td>
      <td>1.89</td>
      <td>3.28%</td>
      <td>1,057.18</td>
      <td>-1.35%</td>
      <td>172.86</td>
      <td>...</td>
      <td>350.12</td>
      <td>12.09%</td>
      <td>0.85</td>
      <td>1.19%</td>
      <td>124.28</td>
      <td>3.00%</td>
      <td>213.00</td>
      <td>-2.63%</td>
      <td>842.51</td>
      <td>1.59%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jun-90</td>
      <td>440.99</td>
      <td>-1.40%</td>
      <td>216</td>
      <td>-7.69%</td>
      <td>1.99</td>
      <td>5.29%</td>
      <td>898.24</td>
      <td>-15.03%</td>
      <td>181.67</td>
      <td>...</td>
      <td>373.94</td>
      <td>6.80%</td>
      <td>0.85</td>
      <td>0.00%</td>
      <td>129.45</td>
      <td>4.16%</td>
      <td>200.00</td>
      <td>-6.10%</td>
      <td>831.35</td>
      <td>-1.32%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jul-90</td>
      <td>418.44</td>
      <td>-5.11%</td>
      <td>205</td>
      <td>-5.09%</td>
      <td>2.01</td>
      <td>1.01%</td>
      <td>895.83</td>
      <td>-0.27%</td>
      <td>187.96</td>
      <td>...</td>
      <td>378.48</td>
      <td>1.21%</td>
      <td>0.86</td>
      <td>1.18%</td>
      <td>124.23</td>
      <td>-4.03%</td>
      <td>210.05</td>
      <td>5.03%</td>
      <td>798.83</td>
      <td>-3.91%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aug-90</td>
      <td>418.44</td>
      <td>0.00%</td>
      <td>198</td>
      <td>-3.41%</td>
      <td>1.79</td>
      <td>-10.95%</td>
      <td>951.22</td>
      <td>6.18%</td>
      <td>186.13</td>
      <td>...</td>
      <td>364.60</td>
      <td>-3.67%</td>
      <td>0.88</td>
      <td>2.33%</td>
      <td>129.70</td>
      <td>4.40%</td>
      <td>208.30</td>
      <td>-0.83%</td>
      <td>818.74</td>
      <td>2.49%</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


Checking the properties of the data.

```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 361 entries, 0 to 360
    Data columns (total 25 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   Month                         361 non-null    object 
     1   Coarse wool Price             327 non-null    object 
     2   Coarse wool price % Change    327 non-null    object 
     3   Copra Price                   339 non-null    object 
     4   Copra price % Change          339 non-null    object 
     5   Cotton Price                  361 non-null    float64
     6   Cotton price % Change         361 non-null    object 
     7   Fine wool Price               327 non-null    object 
     8   Fine wool price % Change      327 non-null    object 
     9   Hard log Price                361 non-null    float64
     10  Hard log price % Change       361 non-null    object 
     11  Hard sawnwood Price           327 non-null    float64
     12  Hard sawnwood price % Change  327 non-null    object 
     13  Hide Price                    327 non-null    float64
     14  Hide price % change           327 non-null    object 
     15  Plywood Price                 361 non-null    float64
     16  Plywood price % Change        361 non-null    object 
     17  Rubber Price                  361 non-null    float64
     18  Rubber price % Change         361 non-null    object 
     19  Softlog Price                 327 non-null    float64
     20  Softlog price % Change        327 non-null    object 
     21  Soft sawnwood Price           327 non-null    float64
     22  Soft sawnwood price % Change  327 non-null    object 
     23  Wood pulp Price               360 non-null    float64
     24  Wood pulp price % Change      360 non-null    object 
    dtypes: float64(9), object(16)
    memory usage: 70.6+ KB
    
There are actually alot of null values in the lower part of our dataset. This can adversely affect results in the machine learning project. Here, I decided to drop  of the null values.

```python
data.dropna(inplace = True)
```
Columns of the data set.

```python
# inspect the column labels
data.columns
```




    Index(['Month', 'Coarse wool Price', 'Coarse wool price % Change',
           'Copra Price', 'Copra price % Change', 'Cotton Price',
           'Cotton price % Change', 'Fine wool Price', 'Fine wool price % Change',
           'Hard log Price', 'Hard log price % Change', 'Hard sawnwood Price',
           'Hard sawnwood price % Change', 'Hide Price', 'Hide price % change',
           'Plywood Price', 'Plywood price % Change', 'Rubber Price',
           'Rubber price % Change', 'Softlog Price', 'Softlog price % Change',
           'Soft sawnwood Price', 'Soft sawnwood price % Change',
           'Wood pulp Price', 'Wood pulp price % Change'],
          dtype='object')


There are spaces in  between the column names, this needs to be corrected.

```python
# renaming the columns
data.rename(mapper  ={'Coarse wool Price':'Coarse_wool_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Coarse wool price % Change':'Coarse_wool_perc_Change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Copra Price':'Copra_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Copra price % Change':'Copra_perc_Change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Cotton Price':'Cotton_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Cotton price % Change':'Cotton_perc_Change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Fine wool Price':'Fine_wool_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Fine wool price % Change':'Fine_wool_perc_Change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Hard log Price':'Hard_log_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Hard log price % Change':'Hard_log_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Hard sawnwood Price':'Hard_sawnwood_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Hard sawnwood price % Change':'Hard_sawnwood_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Hide Price':'Hide_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Hide price % change':'Hide_price_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Plywood Price':'Plywood_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Plywood price % Change':'Plywood_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Rubber Price':'Rubber_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Rubber price % Change':'Rubber_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Softlog Price':'Softlog_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Softlog price % Change':'Softlog_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Soft sawnwood Price':'Soft_sawnwood_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Soft sawnwood price % Change':'Soft_sawnwood_perc_change'}, axis = 1, inplace =True)

data.rename(mapper  ={'Wood pulp Price':'Wood_pulp_price'}, axis = 1, inplace =True)
data.rename(mapper  ={'Wood pulp price % Change':'Wood_pulp_perc_change'}, axis = 1, inplace =True)
```
What  the dataset looks like after renaming columns.

```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Coarse_wool_price</th>
      <th>Coarse_wool_perc_Change</th>
      <th>Copra_price</th>
      <th>Copra_perc_Change</th>
      <th>Cotton_price</th>
      <th>Cotton_perc_Change</th>
      <th>Fine_wool_price</th>
      <th>Fine_wool_perc_Change</th>
      <th>Hard_log_price</th>
      <th>...</th>
      <th>Plywood_price</th>
      <th>Plywood_perc_change</th>
      <th>Rubber_price</th>
      <th>Rubber_perc_change</th>
      <th>Softlog_price</th>
      <th>Softlog_perc_change</th>
      <th>Soft_sawnwood_price</th>
      <th>Soft_sawnwood_perc_change</th>
      <th>Wood_pulp_price</th>
      <th>Wood_pulp_perc_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Apr-90</td>
      <td>482.34</td>
      <td>NaN</td>
      <td>236</td>
      <td>NaN</td>
      <td>1.83</td>
      <td>NaN</td>
      <td>1071.63</td>
      <td>NaN</td>
      <td>161.2</td>
      <td>...</td>
      <td>312.36</td>
      <td>-</td>
      <td>0.84</td>
      <td>-</td>
      <td>120.66</td>
      <td>-</td>
      <td>218.76</td>
      <td>-</td>
      <td>829.29</td>
      <td>-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>May-90</td>
      <td>447.26</td>
      <td>7.27</td>
      <td>234</td>
      <td>0.85</td>
      <td>1.89</td>
      <td>3.28</td>
      <td>1057.18</td>
      <td>1.35</td>
      <td>172.86</td>
      <td>...</td>
      <td>350.12</td>
      <td>12.09%</td>
      <td>0.85</td>
      <td>1.19%</td>
      <td>124.28</td>
      <td>3.00%</td>
      <td>213</td>
      <td>-2.63%</td>
      <td>842.51</td>
      <td>1.59%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jun-90</td>
      <td>440.99</td>
      <td>1.40</td>
      <td>216</td>
      <td>7.69</td>
      <td>1.99</td>
      <td>5.29</td>
      <td>898.24</td>
      <td>15.03</td>
      <td>181.67</td>
      <td>...</td>
      <td>373.94</td>
      <td>6.80%</td>
      <td>0.85</td>
      <td>0.00%</td>
      <td>129.45</td>
      <td>4.16%</td>
      <td>200</td>
      <td>-6.10%</td>
      <td>831.35</td>
      <td>-1.32%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jul-90</td>
      <td>418.44</td>
      <td>5.11</td>
      <td>205</td>
      <td>5.09</td>
      <td>2.01</td>
      <td>1.01</td>
      <td>895.83</td>
      <td>0.27</td>
      <td>187.96</td>
      <td>...</td>
      <td>378.48</td>
      <td>1.21%</td>
      <td>0.86</td>
      <td>1.18%</td>
      <td>124.23</td>
      <td>-4.03%</td>
      <td>210.05</td>
      <td>5.03%</td>
      <td>798.83</td>
      <td>-3.91%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aug-90</td>
      <td>418.44</td>
      <td>0.00</td>
      <td>198</td>
      <td>3.41</td>
      <td>1.79</td>
      <td>10.95</td>
      <td>951.22</td>
      <td>6.18</td>
      <td>186.13</td>
      <td>...</td>
      <td>364.6</td>
      <td>-3.67%</td>
      <td>0.88</td>
      <td>2.33%</td>
      <td>129.7</td>
      <td>4.40%</td>
      <td>208.3</td>
      <td>-0.83%</td>
      <td>818.74</td>
      <td>2.49%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sep-90</td>
      <td>412.18</td>
      <td>1.50</td>
      <td>196</td>
      <td>1.01</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>936.77</td>
      <td>1.52</td>
      <td>185.33</td>
      <td>...</td>
      <td>384.92</td>
      <td>5.57%</td>
      <td>0.9</td>
      <td>2.27%</td>
      <td>129.78</td>
      <td>0.06%</td>
      <td>199.59</td>
      <td>-4.18%</td>
      <td>811.62</td>
      <td>-0.87%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Oct-90</td>
      <td>394.64</td>
      <td>4.26</td>
      <td>198</td>
      <td>1.02</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>901.85</td>
      <td>3.73</td>
      <td>189.76</td>
      <td>...</td>
      <td>409.31</td>
      <td>6.34%</td>
      <td>0.9</td>
      <td>0.00%</td>
      <td>121.31</td>
      <td>-6.53%</td>
      <td>206.98</td>
      <td>3.70%</td>
      <td>807.46</td>
      <td>-0.51%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nov-90</td>
      <td>334.5</td>
      <td>15.24</td>
      <td>236</td>
      <td>19.19</td>
      <td>1.82</td>
      <td>1.68</td>
      <td>888.61</td>
      <td>1.47</td>
      <td>179.02</td>
      <td>...</td>
      <td>375.74</td>
      <td>-8.20%</td>
      <td>0.9</td>
      <td>0.00%</td>
      <td>130.5</td>
      <td>7.58%</td>
      <td>206.64</td>
      <td>-0.16%</td>
      <td>773.37</td>
      <td>-4.22%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dec-90</td>
      <td>328.24</td>
      <td>1.87</td>
      <td>237</td>
      <td>0.42</td>
      <td>1.85</td>
      <td>1.65</td>
      <td>870.55</td>
      <td>2.03</td>
      <td>171.13</td>
      <td>...</td>
      <td>363.16</td>
      <td>-3.35%</td>
      <td>0.88</td>
      <td>-2.22%</td>
      <td>119.35</td>
      <td>-8.54%</td>
      <td>198.22</td>
      <td>-4.07%</td>
      <td>741.29</td>
      <td>-4.15%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jan-91</td>
      <td>319.47</td>
      <td>2.67</td>
      <td>233</td>
      <td>1.69</td>
      <td>1.85</td>
      <td>0.00</td>
      <td>887.41</td>
      <td>1.94</td>
      <td>169.19</td>
      <td>...</td>
      <td>362.26</td>
      <td>-0.25%</td>
      <td>0.87</td>
      <td>-1.14%</td>
      <td>126.14</td>
      <td>5.69%</td>
      <td>186.94</td>
      <td>-5.69%</td>
      <td>721.85</td>
      <td>-2.62%</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 25 columns</p>
</div>



### Cleaning the data(removing symbols etc.)
After due observation, it was noticed that our data set has a lot of mixups(Symbols etc.) alongside the values  and the datatypes of the columns are not proper.
```python
# drop the first row
data.drop(0, inplace=True)
```


```python
data.Coarse_wool_price = data.Coarse_wool_price.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Coarse_wool_price = data.Coarse_wool_price.astype('float64')
```


```python
data.Coarse_wool_perc_Change = data.Coarse_wool_perc_Change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Coarse_wool_perc_Change = data.Coarse_wool_perc_Change.astype('float64')
```


```python
data.Copra_price = data.Copra_price.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Copra_price = data.Copra_price.astype('float64')
```


```python
data.Copra_perc_Change = data.Copra_perc_Change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Copra_perc_Change = data.Copra_perc_Change.astype('float64')
```


```python
data.Cotton_price = data.Cotton_price.astype('float64')
```


```python
data.Cotton_perc_Change = data.Cotton_perc_Change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Cotton_perc_Change = data.Cotton_perc_Change.astype('float64')
```


```python
data.Fine_wool_price= data.Fine_wool_price.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Fine_wool_price= data.Fine_wool_price.astype('float64')
```


```python
data.Fine_wool_perc_Change = data.Fine_wool_perc_Change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Fine_wool_perc_Change = data.Fine_wool_perc_Change.astype('float64')
```


```python
data.Hard_log_price = data.Hard_log_price.astype('float64')
```


```python
data.Hard_log_perc_change = data.Hard_log_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Hard_log_perc_change = data.Hard_log_perc_change.astype('float64')
```


```python
data.Hard_sawnwood_price = data.Hard_sawnwood_price.astype('float64')
```


```python
data.Hard_sawnwood_perc_change = data.Hard_sawnwood_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Hard_sawnwood_perc_change = data.Hard_sawnwood_perc_change.astype('float64')
```


```python
data.Hide_price = data.Hide_price.astype('float64')
```


```python
data.Hide_price_perc_change = data.Hide_price_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Hide_price_perc_change = data.Hide_price_perc_change.astype('float64')
```


```python
data.Plywood_price = data.Plywood_price.astype('float64')
```


```python
data.Plywood_perc_change = data.Plywood_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Plywood_perc_change = data.Plywood_perc_change.astype('float64')
```


```python
data.Rubber_price = data.Rubber_price.astype('float64')
```


```python
data.Rubber_perc_change = data.Rubber_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Rubber_perc_change = data.Rubber_perc_change.astype('float64')
```


```python
data.Soft_sawnwood_price = data.Soft_sawnwood_price.astype('float64')
```


```python
data.Soft_sawnwood_perc_change = data.Soft_sawnwood_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Soft_sawnwood_perc_change = data.Soft_sawnwood_perc_change.astype('float64')
```


```python
data.Softlog_price = data.Softlog_price.astype('float64')
```


```python
data.Softlog_perc_change = data.Softlog_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Softlog_perc_change = data.Softlog_perc_change.astype('float64')
```


```python
data.Wood_pulp_price = data.Wood_pulp_price.astype('float64')
```


```python
data.Wood_pulp_perc_change = data.Wood_pulp_perc_change.str.replace(',','').str.extract(r'(\d+.\d+)')
data.Wood_pulp_perc_change = data.Wood_pulp_perc_change.astype('float64')
```
Take a look  at it.

```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Coarse_wool_price</th>
      <th>Coarse_wool_perc_Change</th>
      <th>Copra_price</th>
      <th>Copra_perc_Change</th>
      <th>Cotton_price</th>
      <th>Cotton_perc_Change</th>
      <th>Fine_wool_price</th>
      <th>Fine_wool_perc_Change</th>
      <th>Hard_log_price</th>
      <th>...</th>
      <th>Plywood_price</th>
      <th>Plywood_perc_change</th>
      <th>Rubber_price</th>
      <th>Rubber_perc_change</th>
      <th>Softlog_price</th>
      <th>Softlog_perc_change</th>
      <th>Soft_sawnwood_price</th>
      <th>Soft_sawnwood_perc_change</th>
      <th>Wood_pulp_price</th>
      <th>Wood_pulp_perc_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>May-90</td>
      <td>447.26</td>
      <td>7.27</td>
      <td>234.0</td>
      <td>0.85</td>
      <td>1.89</td>
      <td>3.28</td>
      <td>1057.18</td>
      <td>1.35</td>
      <td>172.86</td>
      <td>...</td>
      <td>350.12</td>
      <td>12.09</td>
      <td>0.85</td>
      <td>1.19</td>
      <td>124.28</td>
      <td>3.00</td>
      <td>213.00</td>
      <td>2.63</td>
      <td>842.51</td>
      <td>1.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jun-90</td>
      <td>440.99</td>
      <td>1.40</td>
      <td>216.0</td>
      <td>7.69</td>
      <td>1.99</td>
      <td>5.29</td>
      <td>898.24</td>
      <td>15.03</td>
      <td>181.67</td>
      <td>...</td>
      <td>373.94</td>
      <td>6.80</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>129.45</td>
      <td>4.16</td>
      <td>200.00</td>
      <td>6.10</td>
      <td>831.35</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jul-90</td>
      <td>418.44</td>
      <td>5.11</td>
      <td>205.0</td>
      <td>5.09</td>
      <td>2.01</td>
      <td>1.01</td>
      <td>895.83</td>
      <td>0.27</td>
      <td>187.96</td>
      <td>...</td>
      <td>378.48</td>
      <td>1.21</td>
      <td>0.86</td>
      <td>1.18</td>
      <td>124.23</td>
      <td>4.03</td>
      <td>210.05</td>
      <td>5.03</td>
      <td>798.83</td>
      <td>3.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aug-90</td>
      <td>418.44</td>
      <td>0.00</td>
      <td>198.0</td>
      <td>3.41</td>
      <td>1.79</td>
      <td>10.95</td>
      <td>951.22</td>
      <td>6.18</td>
      <td>186.13</td>
      <td>...</td>
      <td>364.60</td>
      <td>3.67</td>
      <td>0.88</td>
      <td>2.33</td>
      <td>129.70</td>
      <td>4.40</td>
      <td>208.30</td>
      <td>0.83</td>
      <td>818.74</td>
      <td>2.49</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sep-90</td>
      <td>412.18</td>
      <td>1.50</td>
      <td>196.0</td>
      <td>1.01</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>936.77</td>
      <td>1.52</td>
      <td>185.33</td>
      <td>...</td>
      <td>384.92</td>
      <td>5.57</td>
      <td>0.90</td>
      <td>2.27</td>
      <td>129.78</td>
      <td>0.06</td>
      <td>199.59</td>
      <td>4.18</td>
      <td>811.62</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Oct-90</td>
      <td>394.64</td>
      <td>4.26</td>
      <td>198.0</td>
      <td>1.02</td>
      <td>1.79</td>
      <td>0.00</td>
      <td>901.85</td>
      <td>3.73</td>
      <td>189.76</td>
      <td>...</td>
      <td>409.31</td>
      <td>6.34</td>
      <td>0.90</td>
      <td>0.00</td>
      <td>121.31</td>
      <td>6.53</td>
      <td>206.98</td>
      <td>3.70</td>
      <td>807.46</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nov-90</td>
      <td>334.50</td>
      <td>15.24</td>
      <td>236.0</td>
      <td>19.19</td>
      <td>1.82</td>
      <td>1.68</td>
      <td>888.61</td>
      <td>1.47</td>
      <td>179.02</td>
      <td>...</td>
      <td>375.74</td>
      <td>8.20</td>
      <td>0.90</td>
      <td>0.00</td>
      <td>130.50</td>
      <td>7.58</td>
      <td>206.64</td>
      <td>0.16</td>
      <td>773.37</td>
      <td>4.22</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dec-90</td>
      <td>328.24</td>
      <td>1.87</td>
      <td>237.0</td>
      <td>0.42</td>
      <td>1.85</td>
      <td>1.65</td>
      <td>870.55</td>
      <td>2.03</td>
      <td>171.13</td>
      <td>...</td>
      <td>363.16</td>
      <td>3.35</td>
      <td>0.88</td>
      <td>2.22</td>
      <td>119.35</td>
      <td>8.54</td>
      <td>198.22</td>
      <td>4.07</td>
      <td>741.29</td>
      <td>4.15</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jan-91</td>
      <td>319.47</td>
      <td>2.67</td>
      <td>233.0</td>
      <td>1.69</td>
      <td>1.85</td>
      <td>0.00</td>
      <td>887.41</td>
      <td>1.94</td>
      <td>169.19</td>
      <td>...</td>
      <td>362.26</td>
      <td>0.25</td>
      <td>0.87</td>
      <td>1.14</td>
      <td>126.14</td>
      <td>5.69</td>
      <td>186.94</td>
      <td>5.69</td>
      <td>721.85</td>
      <td>2.62</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Feb-91</td>
      <td>323.23</td>
      <td>1.18</td>
      <td>226.0</td>
      <td>3.00</td>
      <td>1.87</td>
      <td>1.08</td>
      <td>596.02</td>
      <td>32.84</td>
      <td>176.93</td>
      <td>...</td>
      <td>371.70</td>
      <td>2.61</td>
      <td>0.85</td>
      <td>2.30</td>
      <td>126.77</td>
      <td>0.50</td>
      <td>220.67</td>
      <td>18.04</td>
      <td>706.81</td>
      <td>2.08</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 25 columns</p>
</div>




```python
data = data.convert_dtypes()
```


```python
data.dtypes
```




    Month                         string
    Coarse_wool_price            float64
    Coarse_wool_perc_Change      float64
    Copra_price                  float64
    Copra_perc_Change            float64
    Cotton_price                 float64
    Cotton_perc_Change           float64
    Fine_wool_price              float64
    Fine_wool_perc_Change        float64
    Hard_log_price               float64
    Hard_log_perc_change         float64
    Hard_sawnwood_price          float64
    Hard_sawnwood_perc_change    float64
    Hide_price                   float64
    Hide_price_perc_change       float64
    Plywood_price                float64
    Plywood_perc_change          float64
    Rubber_price                 float64
    Rubber_perc_change           float64
    Softlog_price                float64
    Softlog_perc_change          float64
    Soft_sawnwood_price          float64
    Soft_sawnwood_perc_change    float64
    Wood_pulp_price              float64
    Wood_pulp_perc_change        float64
    dtype: object


The prices of Cotton and Rubber are unlike other price columns so in order to have accurate results, I multiplied them by 100. Since others also are similar. Note that this does not affect other values in the other columns.


```python
data.Cotton_price = data.iloc[:,5].mul(100)
```


```python
data.Rubber_price = data.iloc[:,17].mul(100)
```
Take a look at the summary statistics for the data set.

```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coarse_wool_price</th>
      <th>Coarse_wool_perc_Change</th>
      <th>Copra_price</th>
      <th>Copra_perc_Change</th>
      <th>Cotton_price</th>
      <th>Cotton_perc_Change</th>
      <th>Fine_wool_price</th>
      <th>Fine_wool_perc_Change</th>
      <th>Hard_log_price</th>
      <th>Hard_log_perc_change</th>
      <th>...</th>
      <th>Plywood_price</th>
      <th>Plywood_perc_change</th>
      <th>Rubber_price</th>
      <th>Rubber_perc_change</th>
      <th>Softlog_price</th>
      <th>Softlog_perc_change</th>
      <th>Soft_sawnwood_price</th>
      <th>Soft_sawnwood_perc_change</th>
      <th>Wood_pulp_price</th>
      <th>Wood_pulp_perc_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.00000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>...</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
      <td>326.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>626.775429</td>
      <td>3.845307</td>
      <td>530.047761</td>
      <td>5.562362</td>
      <td>162.07362</td>
      <td>3.999663</td>
      <td>849.440092</td>
      <td>4.803497</td>
      <td>249.253620</td>
      <td>3.257914</td>
      <td>...</td>
      <td>510.027178</td>
      <td>2.268466</td>
      <td>166.328221</td>
      <td>5.499755</td>
      <td>164.662025</td>
      <td>5.442454</td>
      <td>291.283497</td>
      <td>4.979663</td>
      <td>678.212362</td>
      <td>2.800000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>299.992828</td>
      <td>3.628820</td>
      <td>264.001641</td>
      <td>5.091321</td>
      <td>53.34573</td>
      <td>3.930971</td>
      <td>285.248110</td>
      <td>4.698225</td>
      <td>68.553994</td>
      <td>3.423842</td>
      <td>...</td>
      <td>93.188458</td>
      <td>2.659446</td>
      <td>106.834450</td>
      <td>5.185055</td>
      <td>25.519555</td>
      <td>4.768734</td>
      <td>33.929470</td>
      <td>5.773700</td>
      <td>158.315029</td>
      <td>2.890238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>247.090000</td>
      <td>0.000000</td>
      <td>182.000000</td>
      <td>0.000000</td>
      <td>82.00000</td>
      <td>0.000000</td>
      <td>417.470000</td>
      <td>0.000000</td>
      <td>133.280000</td>
      <td>0.010000</td>
      <td>...</td>
      <td>335.250000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>119.350000</td>
      <td>0.000000</td>
      <td>183.610000</td>
      <td>0.000000</td>
      <td>384.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>368.490000</td>
      <td>1.300000</td>
      <td>371.000000</td>
      <td>1.895000</td>
      <td>127.25000</td>
      <td>1.377500</td>
      <td>646.257500</td>
      <td>1.557500</td>
      <td>195.275000</td>
      <td>1.190000</td>
      <td>...</td>
      <td>434.727500</td>
      <td>0.402500</td>
      <td>84.250000</td>
      <td>1.617500</td>
      <td>146.117500</td>
      <td>2.152500</td>
      <td>277.717500</td>
      <td>1.905000</td>
      <td>544.632500</td>
      <td>0.662500</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>526.890000</td>
      <td>2.910000</td>
      <td>449.500000</td>
      <td>4.350000</td>
      <td>154.00000</td>
      <td>2.840000</td>
      <td>747.555000</td>
      <td>3.470000</td>
      <td>247.550000</td>
      <td>2.335000</td>
      <td>...</td>
      <td>512.495000</td>
      <td>1.470000</td>
      <td>133.500000</td>
      <td>3.910000</td>
      <td>160.430000</td>
      <td>4.230000</td>
      <td>294.975000</td>
      <td>3.695000</td>
      <td>662.160000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>848.795000</td>
      <td>5.157500</td>
      <td>657.125000</td>
      <td>7.775000</td>
      <td>183.00000</td>
      <td>5.407500</td>
      <td>1016.352500</td>
      <td>6.362500</td>
      <td>287.025000</td>
      <td>4.285000</td>
      <td>...</td>
      <td>582.035000</td>
      <td>3.147500</td>
      <td>215.750000</td>
      <td>7.397500</td>
      <td>180.345000</td>
      <td>7.457500</td>
      <td>310.887500</td>
      <td>6.122500</td>
      <td>832.245000</td>
      <td>4.107500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1391.470000</td>
      <td>22.250000</td>
      <td>1503.000000</td>
      <td>31.820000</td>
      <td>506.00000</td>
      <td>23.640000</td>
      <td>1865.440000</td>
      <td>32.840000</td>
      <td>520.810000</td>
      <td>34.190000</td>
      <td>...</td>
      <td>751.810000</td>
      <td>19.500000</td>
      <td>626.000000</td>
      <td>32.160000</td>
      <td>259.970000</td>
      <td>33.210000</td>
      <td>372.600000</td>
      <td>65.240000</td>
      <td>966.490000</td>
      <td>21.570000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 24 columns</p>
</div>




```python
data.set_index('Month')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coarse_wool_price</th>
      <th>Coarse_wool_perc_Change</th>
      <th>Copra_price</th>
      <th>Copra_perc_Change</th>
      <th>Cotton_price</th>
      <th>Cotton_perc_Change</th>
      <th>Fine_wool_price</th>
      <th>Fine_wool_perc_Change</th>
      <th>Hard_log_price</th>
      <th>Hard_log_perc_change</th>
      <th>...</th>
      <th>Plywood_price</th>
      <th>Plywood_perc_change</th>
      <th>Rubber_price</th>
      <th>Rubber_perc_change</th>
      <th>Softlog_price</th>
      <th>Softlog_perc_change</th>
      <th>Soft_sawnwood_price</th>
      <th>Soft_sawnwood_perc_change</th>
      <th>Wood_pulp_price</th>
      <th>Wood_pulp_perc_change</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>May-90</th>
      <td>447.26</td>
      <td>7.27</td>
      <td>234.00</td>
      <td>0.85</td>
      <td>189.0</td>
      <td>3.28</td>
      <td>1057.18</td>
      <td>1.35</td>
      <td>172.86</td>
      <td>7.23</td>
      <td>...</td>
      <td>350.12</td>
      <td>12.09</td>
      <td>85.0</td>
      <td>1.19</td>
      <td>124.28</td>
      <td>3.00</td>
      <td>213.00</td>
      <td>2.63</td>
      <td>842.51</td>
      <td>1.59</td>
    </tr>
    <tr>
      <th>Jun-90</th>
      <td>440.99</td>
      <td>1.40</td>
      <td>216.00</td>
      <td>7.69</td>
      <td>199.0</td>
      <td>5.29</td>
      <td>898.24</td>
      <td>15.03</td>
      <td>181.67</td>
      <td>5.10</td>
      <td>...</td>
      <td>373.94</td>
      <td>6.80</td>
      <td>85.0</td>
      <td>0.00</td>
      <td>129.45</td>
      <td>4.16</td>
      <td>200.00</td>
      <td>6.10</td>
      <td>831.35</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>Jul-90</th>
      <td>418.44</td>
      <td>5.11</td>
      <td>205.00</td>
      <td>5.09</td>
      <td>201.0</td>
      <td>1.01</td>
      <td>895.83</td>
      <td>0.27</td>
      <td>187.96</td>
      <td>3.46</td>
      <td>...</td>
      <td>378.48</td>
      <td>1.21</td>
      <td>86.0</td>
      <td>1.18</td>
      <td>124.23</td>
      <td>4.03</td>
      <td>210.05</td>
      <td>5.03</td>
      <td>798.83</td>
      <td>3.91</td>
    </tr>
    <tr>
      <th>Aug-90</th>
      <td>418.44</td>
      <td>0.00</td>
      <td>198.00</td>
      <td>3.41</td>
      <td>179.0</td>
      <td>10.95</td>
      <td>951.22</td>
      <td>6.18</td>
      <td>186.13</td>
      <td>0.97</td>
      <td>...</td>
      <td>364.60</td>
      <td>3.67</td>
      <td>88.0</td>
      <td>2.33</td>
      <td>129.70</td>
      <td>4.40</td>
      <td>208.30</td>
      <td>0.83</td>
      <td>818.74</td>
      <td>2.49</td>
    </tr>
    <tr>
      <th>Sep-90</th>
      <td>412.18</td>
      <td>1.50</td>
      <td>196.00</td>
      <td>1.01</td>
      <td>179.0</td>
      <td>0.00</td>
      <td>936.77</td>
      <td>1.52</td>
      <td>185.33</td>
      <td>0.43</td>
      <td>...</td>
      <td>384.92</td>
      <td>5.57</td>
      <td>90.0</td>
      <td>2.27</td>
      <td>129.78</td>
      <td>0.06</td>
      <td>199.59</td>
      <td>4.18</td>
      <td>811.62</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Feb-17</th>
      <td>1029.58</td>
      <td>0.18</td>
      <td>1146.25</td>
      <td>6.43</td>
      <td>188.0</td>
      <td>3.30</td>
      <td>1368.14</td>
      <td>6.06</td>
      <td>263.45</td>
      <td>1.88</td>
      <td>...</td>
      <td>483.23</td>
      <td>1.88</td>
      <td>271.0</td>
      <td>5.86</td>
      <td>157.58</td>
      <td>7.39</td>
      <td>287.43</td>
      <td>7.73</td>
      <td>875.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Mar-17</th>
      <td>1059.60</td>
      <td>2.92</td>
      <td>1016.00</td>
      <td>11.36</td>
      <td>191.0</td>
      <td>1.60</td>
      <td>1454.83</td>
      <td>6.34</td>
      <td>263.48</td>
      <td>0.01</td>
      <td>...</td>
      <td>483.27</td>
      <td>0.01</td>
      <td>235.0</td>
      <td>13.28</td>
      <td>160.05</td>
      <td>1.57</td>
      <td>300.42</td>
      <td>4.52</td>
      <td>875.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Apr-17</th>
      <td>991.12</td>
      <td>6.46</td>
      <td>1044.00</td>
      <td>2.76</td>
      <td>192.0</td>
      <td>0.52</td>
      <td>1404.98</td>
      <td>3.43</td>
      <td>270.34</td>
      <td>2.60</td>
      <td>...</td>
      <td>495.87</td>
      <td>2.61</td>
      <td>221.0</td>
      <td>5.96</td>
      <td>159.84</td>
      <td>0.13</td>
      <td>306.60</td>
      <td>2.06</td>
      <td>875.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>May-17</th>
      <td>1019.95</td>
      <td>2.91</td>
      <td>1112.50</td>
      <td>6.56</td>
      <td>195.0</td>
      <td>1.56</td>
      <td>1433.47</td>
      <td>2.03</td>
      <td>265.28</td>
      <td>1.87</td>
      <td>...</td>
      <td>486.59</td>
      <td>1.87</td>
      <td>210.0</td>
      <td>4.98</td>
      <td>159.84</td>
      <td>0.00</td>
      <td>306.60</td>
      <td>0.00</td>
      <td>875.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Jun-17</th>
      <td>1065.81</td>
      <td>4.50</td>
      <td>1119.00</td>
      <td>0.58</td>
      <td>187.0</td>
      <td>4.10</td>
      <td>1403.83</td>
      <td>2.07</td>
      <td>268.39</td>
      <td>1.17</td>
      <td>...</td>
      <td>492.29</td>
      <td>1.17</td>
      <td>172.0</td>
      <td>18.10</td>
      <td>159.84</td>
      <td>0.00</td>
      <td>306.60</td>
      <td>0.00</td>
      <td>875.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>326 rows × 24 columns</p>
</div>

Finally, I saved the cleaned data set into another csv file.

```python
data.to_csv('Agricultural_raw_materials_cleaned.csv', index = False)
```
