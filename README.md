# evil-softmax

This repository explores the limitations of the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) in reinforcement learning. In particular, it finds cases when the vanilla policy gradient method converges to sub-optimal solutions, even with linear policies and perfect sampling.

# Background

Sometimes, one wants a neural network to output a discrete probability distribution. For example, if we are classifying an image, we want a probability for each type of object that might be in the image. The [softmax function](https://en.wikipedia.org/wiki/Softmax_function) is often used to this end.

In the domain of reinforcement learning, we can use a softmax to decide which action to take at each time-step. For example, our agent might be able to press four different buttons, and we want it to decide which button to press based on the output of a neural network. By using the softmax function, we can apply a [policy gradient method](http://www.scholarpedia.org/article/Policy_gradient_methods) to train the policy.
