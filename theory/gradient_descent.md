# **Gradient Descent**

## Motivation
For most of the Machine Learning models, the objective is to minimise the loss function. It means finding an equation which best fits the training sample. For small and simple datasets, it is possible to find single step solution but in real world scenarios data is more complex. Gradient Descent provides an iterative approach to minimise loss function by finding optimal model parameters.

## Definition
Gradient Descent is an algorithm that is used to minimise the cost function by iteratively adjusting the parameters(weights and bias) in the direction of steepest descent of the function's gradient ie. loss function wrt to model parameters.

## Problem Formulation
Given a dataset

  D = { (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ) }

Each pair consists of:
- xᵢ → input features
- yᵢ → actual output

- **Input vectors**

  Each input xᵢ is a vector of d real numbers.

- **Target values**

  Each target yᵢ is a real number that is the actual output for xᵢ.

- **Model parameters**

  The model has p adjustable parameters grouped into a vector θ(* *theta* *) .

  We define a paramteric model :

  ŷᵢ = f(xᵢ, θ)

## Loss Function
To find how well the equation fits the dataset we compute the loss function.

For a single data point, loss function is-
 l(ŷᵢ , yᵢ) = ŷᵢ - yᵢ = f(xᵢ, θ) - yᵢ


For entire dataset - 
L(θ) = (1/n) Σ l(ŷᵢ , yᵢ)

But since loss both positive and negative should be added in total loss and not canceling each other during summation eg loss of +5 and -5 is actually a total loss of 10 not 0 when added, hence we square the loss and divide the sum by the total number of records in dataset.

  **L(θ) = (1/n) Σ (f(xᵢ, θ) - yᵢ)<sup>2</sup>**

Our objective is: minimize L(θ)

So we need to find value for θ such that loss is minimum. So we will find ∇L(θ)(partial derivative of L wrt θ).This will tell us whether the vector of L wrt to a particular parameter of θ has increasing slope(∇L(θ)>0) or decreasing slope(∇L(θ)<0) corresponding to a particular initial value of θ.

If ∇L(θ)>0 that means we need to reduce θ by small amount in order to go towards the minimum value of L.
Else we need to increase θ by small amount.

## Gradient Descent Paramter Update Equation

**θ = θ - α ∇L(θ)**

where α is learning rate

For each parameter in θ, we need to update that parameter for a number of iterations and calculate L(θ) for each iteration.
If L(θ) keeps decreasing then gradient descent is working correctly for chosen α.

If L(θ) keeps increasing and decreasing betwwen iterations α chosen is very large.

## Variants of Gradient Descent

The main difference between gradient descent variants is:

**How much data is used to compute the gradient at each update step.**


### 1 Batch Gradient Descent

In Batch Gradient Descent, the gradient is computed using the entire dataset.

This means:

∇L(θ) is calculated using all n training examples before updating parameters.

#### How it works:

1. Compute predictions for all training examples.
2. Compute total loss.
3. Compute gradient using the full dataset.
4. Update parameters once.

#### Characteristics:

- Stable updates.
- Computationally expensive for large datasets.
- Requires the entire dataset in memory.


### 7.2 Stochastic Gradient Descent (SGD)

In Stochastic Gradient Descent, the gradient is computed using only one training example at a time.

Instead of computing the full loss, we approximate it using a single sample.

#### How it works:

For each training example:

1. Compute prediction for one example.
2. Compute its loss.
3. Compute gradient from that single example.
4. Update parameters immediately.
5. Then take the next example.

#### Characteristics:

- Faster updates.
- Much lower memory requirement.
- Noisy updates (loss may fluctuate).
- It may oscillate near the minimum instead of finding exact value.

#### When it is useful:

- Very large datasets


### 7.3 Mini-Batch Gradient Descent

Mini-batch gradient descent is a combination between Batch GD and SGD.

Instead of using:
- All data (batch)
- One example (SGD)

We use a small subset of data.

#### How it works:

1. Divide dataset into small batches.
2. For each batch:
   - Compute predictions
   - Compute batch loss
   - Compute gradient
   - Update parameters

#### Characteristics:

- Faster than batch gradient descent.
- Less noisy than SGD.
- Most commonly used method in practice.



