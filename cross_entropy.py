import numpy as np

# function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

'''Cross entropy:
if we have 3 doors and we need to find what is the probability of gift under
door1, door2 or door 3.
Let's suppose p(gift behind door 1) = 0.8
p(gift behind door 2) = 0.7
p(gift behind door 3) = 0.1
for door 1 & door 2 it's more likely that there is a gift with prob as 0.8 and 0.7 resp.
but more like li probability that there is no gift behind door 3 with prob(1-p(gift)) = 0.9
As the events are independent then more likely probability is 0.8*0.7*0.9 = 0.504(roughly 50%)
Cross entropy = -log(probability)
let's form the generalized formula for cross entropy
p1 - probability that there is gift behind 1st door
p2 - probability that there is gift behind 2nd door
p3 - probability that there is gift behind 3rd door
y - no of gifts present behind the door
y1=1(present behind 1st door), y2=1(present behind the 2nd door), y3=0(no present behind the 3rd door)
Cross_entropy = [(y1,y2,y3),(p1,p2,p3)]
Formula : -Summation(from i=1 to m)((yi*ln(pi))+(1-yi)*(1-pi))
if y1 = 1 i.e. gift behind door 1 then, the 2nd term in above formula will become 0
if y1 = 0 i.e. no gift behind the door 1 then, 1st term in above formula will become 0


Generalization : Cross entropy is in relation to probability
if probability of an event is more likely : it has less cross entropy i.e. low error
if probability of an even is less likely : it has more cross entropy i.e. more error
'''
# Function takes input of two list i.e. Y- labels and P- probability
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    entropy = - np.sum(Y*np.log(P) + (1-Y)*np.log(1-P))
    return entropy

Y=[1,0,1,1]
P=[0.4,0.6,0.1,0.5]
print("Input values for Y are : ",Y)
print("Input probability values are : ",P)
print("Result for cross entropy is : ",cross_entropy(Y,P))