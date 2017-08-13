# Deep_Learning_Algorithms

### Softmax algorithm: 
In case of classification problem, where we have more than 2 classes for e.x. if we want to find the classes of animals like 
whether the animal is a duck, beaver or walgrus.
With the help of some linear equation which is classified on the basis of some features that belong to all the 3 animals, scores are derived.
In order to change those scores to probabilities, we apply softmax function:
Softmax fn. is defined by, let's say we have n classes
score for n classes is given by : z1, z2, z3, _ _ _ _, zn.
Probability(an object belongs to class i) = e^zi/e^z1+e^z2+e^z3+_ _ _ _+e^zn.
* Output : Trying for L=[5,6,7].
The correct answer is
[0.090030573170380462, 0.24472847105479764, 0.6652409557748219]
And your code returned
[0.090030573170380462, 0.24472847105479764, 0.6652409557748219]
