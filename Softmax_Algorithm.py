import numpy as np

'''In case of classification problem, where we have more than 2 classes
for e.x. in case we want to find the classes of animals like whether the animal is a duck, beaver
or walgrus.
In this case with the help of some linear equation which is classified on the basis of some
features. We get the scores.
Now, in order to change those scores to probabilities, we apply softmax function
Softmax fn. is defined by, let's say we have n classes
score for n classes is given by : z1, z2, z3, _ _ _ _, zn
probability(an object belongs to class i) = e^zi/e^z1+e^z2+e^z3+_ _ _ _+e^zn
'''
# function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
L=[5,6,7]
print("Trying for L =", L)
def softmax(L):
    result = []
    sum_exp_deno = np.sum(np.exp(L))
    for i in range(len(L)):
        score = np.divide(np.exp(L[i]), sum_exp_deno)
        result.append(score)
    return result

print("Result of softmax fn. is : ", softmax(L))
