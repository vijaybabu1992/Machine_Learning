## Exercises
In this chapter we have covered some of the most important concepts in Machine Learning.

1. How would you define Machine Learning?
   -- Machine Learning is about building systems that can learn from data.
   Learning means getting better at some task, given some performance measure.
   
2. Can you name four types of problems where it shines?
  -- Machine Learning is great for complex problems for which we have no algorithmic solution,
   to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments,
   and finally to help humans learn (e.g., data mining).
   
6. What is a labeled training set?
 --A labeled training set is a training set that contains the desired solution (a.k.a. a
label) for each instance.

8. What are the two most common supervised tasks?
  -- The two most common supervised tasks are regression and classification.
   
10. Can you name four common unsupervised tasks?
    -- Common unsupervised tasks include clustering, visualization, dimensionality
reduction, and association rule learning.

12. What type of Machine Learning algorithm would you use to allow a robot to
walk in various unknown terrains?
-- Reinforcement Learning is likely to perform best if we want a robot to learn to
walk in various unknown terrains since this is typically the type of problem that
Reinforcement Learning tackles. It might be possible to express the problem as a
supervised or semisupervised learning problem, but it would be less natural.

14. What type of algorithm would you use to segment your customers into multiple
groups?
-- If you don’t know how to define the groups, then you can use a clustering algo‐
rithm (unsupervised learning) to segment your customers into clusters of similar
customers.
However, if you know what groups you would like to have, then you
can feed many examples of each group to a classification algorithm (supervised
learning), and it will classify all your customers into these groups.

16. Would you frame the problem of spam detection as a supervised learning prob‐
lem or an unsupervised learning problem?
-- Spam detection is a typical supervised learning problem: the algorithm is fed
many emails along with their label (spam or not spam).

18. What is an online learning system?
    -- An online learning system can learn incrementally, as opposed to a batch learn‐
ing system. This makes it capable of adapting rapidly to both changing data and
autonomous systems, and of training on very large quantities of data.

20. What is out-of-core learning?
    -- Out-of-core algorithms can handle vast quantities of data that cannot fit in a
computer’s main memory. An out-of-core learning algorithm chops the data into
mini-batches and uses online learning techniques to learn from these mini-
batches.

22. What type of learning algorithm relies on a similarity measure to make predic‐
tions?
-- An instance-based learning system learns the training data by heart; then, when
given a new instance, it uses a similarity measure to find the most similar learned
instances and uses them to make predictions.

24. What is the difference between a model parameter and a learning algorithm’s
hyperparameter?
-- A model has one or more model parameters that determine what it will predict
given a new instance (e.g., the slope of a linear model). A learning algorithm tries
to find optimal values for these parameters such that the model generalizes well
to new instances. A hyperparameter is a parameter of the learning algorithm
itself, not of the model (e.g., the amount of regularization to apply).

26. What do model-based learning algorithms search for? What is the most common
strategy they use to succeed? How do they make predictions?
-- Model-based learning algorithms search for an optimal value for the model
parameters such that the model will generalize well to new instances. We usually
train such systems by minimizing a cost function that measures how bad the sys‐
tem is at making predictions on the training data, plus a penalty for model com‐
plexity if the model is regularized. To make predictions, we feed the new
instance’s features into the model’s prediction function, using the parameter val‐
ues found by the learning algorithm.

28. Can you name four of the main challenges in Machine Learning?
    -- Some of the main challenges in Machine Learning are the lack of data, poor data
quality, nonrepresentative data, uninformative features, excessively simple mod‐
els that underfit the training data, and excessively complex models that overfit
the data.

30. If your model performs great on the training data but generalizes poorly to new
instances, what is happening? Can you name three possible solutions?
-- If a model performs great on the training data but generalizes poorly to new
instances, the model is likely overfitting the training data (or we got extremely
lucky on the training data). Possible solutions to overfitting are getting more
data, simplifying the model (selecting a simpler algorithm, reducing the number
of parameters or features used, or regularizing the model), or reducing the noise
in the training data.

32. What is a test set and why would you want to use it?
    -- A test set is used to estimate the generalization error that a model will make on
new instances, before the model is launched in production.

34. What is the purpose of a validation set?
    -- A validation set is used to compare models. It makes it possible to select the best
model and tune the hyperparameters.

36. What can go wrong if you tune hyperparameters using the test set?
    -- If you tune hyperparameters using the test set, you risk overfitting the test set,
and the generalization error you measure will be optimistic (you may launch a
model that performs worse than you expect).

38. What is cross-validation and why would you prefer it to a validation set?
    -- Cross-validation is a technique that makes it possible to compare models (for
model selection and hyperparameter tuning) without the need for a separate vali‐
dation set. This saves precious training data.
