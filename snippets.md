
##Visualization Tips

#SETTING TICK LABELS**
Control the y-ticks by specifying two arguments:
```python
plt.yticks([0,1,2], ["one","two","three"])
```
In this example, the ticks corresponding to the numbers 0, 1 and 2 will be replaced by one, two and three, respectively.

Let's do a similar thing for the x-axis of your world development chart, with the xticks() function. The tick values 1000, 10000 and 100000 should be replaced by 1k, 10k and 100k. To this end, two lists have already been created for you: tick_val and tick_lab.

```python
# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_value = [1000, 10000, 100000]
tick_label = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_value, tick_label)

# After customizing, display the plot
plt.show()
```

#SETTING SCATTER SIZES

Right now, the scatter plot is just a cloud of blue dots, indistinguishable from each other. Let's change this. Wouldn't it be nice if the size of the dots corresponds to the population?

To accomplish this, there is a list pop loaded in your workspace. It contains population numbers for each country expressed in millions. You can see that this list is added to the scatter method, as the argument s, for size.

```python
# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop. All items in the np array will be doubled!
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Display the plot
plt.show()
```

#SETTING PLOT COLORS

To do this, a list col has been created for you in dictionary form below. It's a list with a color for each corresponding country, depending on the continent the country is part of.

How did we make the list col you ask? The Gapminder data contains a list continent with the continent each country belongs to. A dictionary is constructed that maps continents onto colors:

col = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}

Change the **opacity** of the bubbles by setting the alpha argument to 0.8 inside plt.scatter(). Alpha can be set from zero to one, where zero is totally transparent, and one is not at all transparent.

```python
# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c=col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Show the plot
plt.show()
```

#Additional Customizations

You'll see that there are two plt.text() functions now. They add the words "India" and "China" in the plot
```python

# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations adds labels to the India and China points
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call which makes a grid behind the whole plot
plt.grid(True)

# Show the plot
plt.show()
```

##Pandas Tips

Setting row index when importing a csv:

```python
df = pd.read_csv('filename.csv', index_col = 0)
```
Specify row labels:
```python
# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels
```
Indexing a dataframe without turning it into a pandas series:
```python
#WRONG:
df['column name']

#RIGHT:
df[['column name']]
```
Slicing rows of dataframes:

```python
#slices the second to the fourth rows
df[1:4]
```
Loc vs iloc indexing:

loc = based on labels
iloc = based on indeger indexing
```python
#provides all info from a single row based on label as a DATAFRAME
df.loc[['index label']]
df.loc[['A', 'B', 'C']]

#to specify the columns, you add a comma - it will return a subsetted dataframe
df.loc[['Seattle', 'New York', 'Atlanta'], ['population', 'state']]

#keep all rows but subset dataframe by columns using just a colon and column names
df.loc[:, ['population', 'state']]

#comparing loc and iloc: The two below accomplish the same thing!
df.loc[['first index label']] 
df.iloc[[0]]

#The two below could accomplish the same thing if they were the first rows and columns
df.loc[['Seattle', 'New York', 'Atlanta'], ['population', 'state']]
df.iloc[[0,1,2], [0,1]]
```
Find a specific row in a specific column using loc
```python
# Print out the drives_right value of the row corresponding to Morocco (its row label is MOR)
print(cars.loc[['MOR'],['drives_right']])
#Multiple rows and columns
# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country','drives_right']])
```
Segementing by Series vs by Dataframe
```python
# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])
# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])
```

#Comparison Operators using Numpy

**Boolean Operators with Numpy Arrays**

In a numpy array, you can use comparison operators that will work against each item in the array
```python
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)

# It appears that the living room and bedroom in my_house are smaller than the corresponding areas in your_house!
```
**Boolean Operators with Variables**

```python
# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < your_kitchen*3)
```
**Boolean Operators with Variables NUMPY**
```python
# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11))
```

**CUT FUNCTION**

Bin values into discrete intervals. Use cut when you need to segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges
```python
pd.cut(x, bins, right: bool = True, labels=None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise')
```

#FILTERING PANDAS DATA FRAMES

```python
#Get a single column as a Pandas Series:
df['area']

#Add a conditional operator which returns a series containing booleans:
df['area'] > 10

#Save the series to a variable name:
is_huge = df['area'] > 10

#Subset the dataframe returns a dataframe
df[is_huge]

#OR just do it all at once!
df[df['area'] > 10]] 
```

# WHILE LOOP BASICS:
```python
# Initialize offset
offset = 8

# Code the while loop that runs as long as offset is not equal to 0
while offset != 0:
    #create a print statement to show how many times it loops
    print("correcting...")
    #decrease the value of offset by 1
    offset= offset -1
    #print offset each time to see it's value
    print(offset)

#It will print 'correcting...' 8 times
```
CONDITIONALS WITH WHILE LOOPS. 

```python
#if the offset above was negative, it would run forever. Add conditionals!
# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset >0:
      offset=offset-1
    else : 
      offset=offset+1   
    print(offset)
```

# FOR LOOP BASICS

**Using ENUMERATE**
```python
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate() and update print()
for index, area in enumerate(areas):
    #make the index start at 1 instead of 0
    index=index+1
    #print the index and value as a string for each iterations
    print("room" + str(index)+":"+ str(area))
```
**Iterate through a list of lists**
```python
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for x in house:
    #index the sublists to get each sublist item
    print("the " + x[0] + " is " + str(x[1]) + " sqm")

#return 'the hallway is 11.25sqm' etc for each room
```

**Iterate through dicionaries**

If you try to loop through using just key, value you will get a Value Error. 
Instead, you can use **dictionary.items** in the for loop!
Remember that dictionaries are NOT ordered so they will iterate out of order.

```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print("the capital of" + str(key) + " is " + str(value))
```
**Iterate through Numpy array**

```python
for value in np.nditer(my_array):
    print(value)
```

**Iterate over PANDAS dataframe**
```python
for label, row in df.iterrows():
    print(label)
    print(row)

#Subset with specific rows
for label, row in df.iterrows():
    print(label + ":" + row['column_name'])
```
**Adding a column AND iterating over pandas dataframe**

use the apply function

```python
#EVERY INEFFICIENT
for label, row in df.iterrows():
    #creating a series on every iteration
    df.loc[label, 'name_length'] = len(row['country'])
#check that the column was created correctly using an unindented print statement
print(df)

#Use APPLY instead
df['name_length']= df['country'].apply(len)

#To apply a method (like '.upper()' here is an example:
cars['COUNTRY'] = cars['country'].apply(str.upper)

```

# ITERATING OVER AND APPENDING TO A DICTIONARY

```python
# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1 
    if entry in langs_count.keys():
        #Using the +=, you first GET the value at that entry currently and SET it to that value plus the new one. If entry was previously 2, it is now 3.
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)
```

# FLEXIBLE ARGUMENTS IN PYTHON FUNCTIONS

**Using Multiple Default Arguments**
```python
# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Make echo_word uppercase if intense is True
    if intense is True:
        # Make uppercase and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo('Hey', echo=5, intense = True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo('Hey', intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)

#HEYHEYHEYHEYHEY!!!
#HEY!!!
```

**Using ARGS**

Pass one or multiple arguments to the function.

```python
# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ""

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

#luke
#lukeleiahanobidarth
```

**Using KWARGS**

What makes **kwargs different is that it allows you to pass a variable number of keyword arguments to functions. Within the function definition, kwargs is a dictionary.

```python
# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name = 'luke', affiliation ='jedi', status ='missing')

#BEGIN: REPORT

#name: luke
#affiliation: jedi
#status: missing

#END REPORT

# Second call to report_status()
report_status(name='anakin', affiliation='sith lord', status='deceased')

#BEGIN: REPORT

#name: anakin
#affiliation: sith lord
#status: deceased

#END REPORT
```

# FLEXIBLE ARGUMENTS WITH PANDAS

**Using default arguments**
```python
# Define count_entries()
def count_entries(df, col_name = 'lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, col_name = 'source')

# Print result1 and result2
print(result1)
print(result2)

#This allows you to count occurances in the 'language' column, and then also in the 'source' column!
```
**Using *args to deterine counts of row occurances for columns (keys)**

```python
# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
        print(col)
        
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): for one column
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): for two columns!
result2 = count_entries(tweets_df, 'lang','source')

# Print result1 and result2
print(result1)
print(result2)



