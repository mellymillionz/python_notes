#Understanding Random with Numpy

Climbing the Empire State building. Useing steps up or down as dependent upon the dice roll

```python
# Numpy is imported, seed is set

# Starting step
step = 50

# Roll the dice. Specify numbers on the dice (between 1 and 6)
dice = np.random.randint(1, 7)

print(dice)

# Finish the control construct. 
#If less than or equal to 2, step down
#if greater or equal to 5, step up.
#Else, a random number of steps is generated
if dice <= 2 :
    step = step - 1
elif dice <=5:
    step = step + 1
else :
    step = step + np.random.randint(1,7) #numbers 1-6 on the dice

# Print out dice and step
print(dice)
print(step)
```

#RANDOM WALK

If you use a die to determine next step, its a random step.
If you use a die to determine many next steps, its a random walk.
(financial status of a gambler for example)

**Need to gradually build a list with a for loop:**
```python
import numpy as np
np.random.seed(123)

outcomes = []
#setting a for loop in range x means it will run 10 times
for x in range(10): 
    coin = np.random.randint(0,2) #between 0 and 1 (heads and tails)
    if coin == 0:
        outcomes.append('heads')
    else:
        outcomes.append('tails')
print(outcomes)
#this is NOT a random walk because it is just a list of random numbers, its not based on the thing that came before.
```


