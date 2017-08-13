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


### Cross Entropy:
Cross entropy:
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


### Logistic Regression & Gradient Descent

Below, you'll be given the code for the functions that calculate the sigmoid, the derivative of the sigmoid, the errors, and the prediction. 
You'll be asked to write the code for the functions dErrors and gradientDescentStep, which do the following:

dErrors: This function should receive X and y, and return a list of errors given by the formula for the gradient 
(y−y^)(x​1​​ ,…,x​n​ ).
gradientDescentStep: In this function, you receive X, y,W and b, and you need to upgrade the weights and the bias by subtracting the coordinates of the gradient, given in the function dErrors.
If you get stuck, feel free to look at solution.py. But give it a try first!

Then, as before, click on test run to graph the solution that the perceptron algorithm gives you. It'll actually draw a set of dotted lines, that show how the algorithm approaches to the best solution, given by the black solid line. This will also show a plot of the error, and you can see how it decreases as we get closer and closer to an optimal solution.


![Output](/graph_output.PNG)
![Output](/graph.PNG)