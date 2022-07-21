# 02 - Classification

- [02 - Classification](#02---classification)
- [Types](#types)
- [Classification inputs & outputs](#classification-inputs--outputs)

# Types

What is a classification problem?

**Binary Classification:** The task of classifying the elements of a set into two groups (each called class) on the basis of a classification rule.

_e.g._ is this email spam or not spam?

**Multiclass Classification:** The problem of classifying instances into one of three or more classes. aka multinomial classification.

_e.g._ is this a photo of sushi, steak, or pizza?

**Multilabel Classification:** The problem of finding a model that maps inputs x to binary vectors y; that is, it assigns a value of 0 or 1 for each element (label) in y.

_e.g._ What tags does this web article need?

# Classification inputs & outputs

**Images:**

- W: width
- H: height
- C: color channels

**Tensor shape:** [batch_size, width, height, color_channels]
batch_size = number of images in the batch

| Hyperparameter           | Binary Classification                                                                                         | Multiclass Classification                                             |
| :----------------------- | :------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------- |
| Input Layer Shape        | Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart dissease prediction) | Same as binary classification                                         |
| Hidden Layer(s)          | Problem specific, minimum = 1, maximum = unlimited                                                            | Same as binary classification                                         |
| Neurons per Hidden Layer | Problem specific, generally 10 to 100                                                                         | Same as binary classification                                         |
| Output Layer Shape       | 1 (one class or the other)                                                                                    | 1 per class (e.g. 3 for food, person or dog photo)                    |
| Hidden Activation        | Usually ReLU (rectified linear unit)                                                                          | Same as binary classification                                         |
| Output Activation        | Sigmoid                                                                                                       | Softmax                                                               |
| Loss Function            | Cross entropy (tf.keras.losses.BinaryCrossentropy in TensorFlow)                                              | Cross entropy (tf.keras.losses.CategoricalCrossentropy in TensorFlow) |
| Optimizer                | SGD (stochastic gradient descent), Adam                                                                       | Same as binary classification                                         |
