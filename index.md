---
layout: default
title: Deep Learning for Computer Vision
---

# Deep Learning for Computer Vision

### Second semester 2022

Andrés Marrugo, PhD       
*Universidad Tecnológica de Bolívar*

##  Aims and Scope

This semester course will provide a practical introduction to neural networks and deep learning. Covered topics covered will include: linear classifiers; multilayer neural networks; stochastic gradient descent and backpropagation; convolutional neural networks and their applications to computer vision tasks such as object detection and dense image labeling. Course work will consist of programming tasks in Python.

At the end of the lectures, one would be able to:

- Gain experience in computer vision research.
- Get exposure to a variety of topics and become familiar with modern machine learning and deep learning techniques.
- Learn to write and present works of a technical-academic nature.
- Gain hands-on experience developing machine learning systems.

<!-- Prior knowledge of this course includes probability, linear algebra, and calculus. Programming experience in MATLAB is desirable, but not required. -->


<!-- This semester course is an introduction to computer vision. It is aimed at graduate students in the Faculty of Engineering. We will focus on the practical and theoretical aspects of techniques in computer vision. -->

<!-- At the end of the lectures, one would be able to:

- Have clear idea of challenges in computer vision due to increasing use in mobile applications.
- Understand many different computer vision algorithms and approaches.
- Implement computer vision algorithms for mid-level vision tasks. -->


## Useful Resources
 
We will be using Jupyter Python notebooks as a numerical computing and graphical platform for solving many problems in the course. To avoid installing Jupyter Python locally, I encourage you to use Google Colab. 

- [Python Tutorial](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)
- [Introduction to colab](https://colab.research.google.com/notebooks/welcome.ipynb)

### Tutorials, review materials

- [Linear algebra review](http://www.cse.ucsd.edu/classes/wi05/cse252a/linear_algebra_review.pdf)
- [Random variables review](http://www.cse.ucsd.edu/classes/wi05/cse252a/random_var_review.pdf)
- [Linear algebra with Numpy](https://github.com/agmarrugo/computer-vision-utb/blob/main/notebooks/00_Linear_algebra_with_Numpy.ipynb)
- [Manipulating images in Python/OpenCV](https://github.com/agmarrugo/computer-vision-utb/blob/main/notebooks/01_Image_Processing_in_Python_Final.ipynb)
- [Data analysis with Pandas](https://github.com/drvinceknight/Python-Mathematics-Handbook/blob/master/03-Data-analysis-with-Pandas.ipynb)
- [Visualisation with matplotlib](https://github.com/drvinceknight/Python-Mathematics-Handbook/blob/master/04-Visualisation-with-matplotlib.ipynb)



## Outline

This website will be updated as we go along.

### Lecture 1: Introduction

[Lecture 1 slides]({{site.url}}lectures/lec01_intro.pdf)

#### Additional Reading

- [Chapter 1 - I. Goodfellow, Y. Bengio and A. Courville](https://www.deeplearningbook.org)


### Lecture 2: Introduction to machine learning

In this lecture, we cover the general aspects of machine learning.

[Lecture 2 slides]({{site.url}}lectures/lec02_ml_intro.pdf)

#### Recommended Readings

- [Chapter 5 - I. Goodfellow, Y. Bengio and A. Courville](https://www.deeplearningbook.org)

### Lecture 3: Linear classifiers

In this two-part lecture, we cover linear classifiers.

- [Lecture 3 slides]({{site.url}}lectures/lec03_linear_part1.pdf)

### Lecture 4: Linear classifiers

In this two-part lecture, we cover linear classifiers.

- [Lecture 4 slides]({{site.url}}lectures/lec04_linear_part2.pdf)

### In-class activity

We will be working out the code from two chapters of the book Deep Learning for Computer Vision by Adrian Rosenbrock. Download the pdfs and launch the notebooks.

#### Book chapters

- [Chapter 8]({{site.url}}pdfs/parameterized-learning.pdf)
- [Chapter 9]({{site.url}}pdfs/optimization-methods-regularization.pdf)

#### Notebooks

- [Parameterized learning example](https://github.com/opi-lab/DL4CV/blob/main/notebooks/parameterized-learning.ipynb)
- [Optimization methods](https://github.com/opi-lab/DL4CV/blob/main/notebooks/optimization_examples.ipynb)

#### What to submit
Upload the two notebooks with corresponding annotations and output in html format. 

- [Upload link](https://www.dropbox.com/request/LJt4Lvi0MHRhzUxNIO2z)

### Lecture 5: Nonlinear classifiers

In this lecture, we cover nonlinear classifiers: the “Shallow” approach using Kernel support vector machines (SVMs) and the “Deep” approach using Multi-layer neural networks. We also cover how to control classifier complexity, hyperparameters, bias-variance tradeoff, overfitting and underfitting, and hyperparameter search in practice.

- [Lecture 5 slides]({{site.url}}lectures/lec05_nonlinear_classifiers.pdf)

### In-class activity

We will be implementing our first neural network. It is Logistic Regression with a Neural Network mindset based on the course by Andrew Ng. Launch the notebook in Colab. 

 - [Notebook](https://github.com/opi-lab/DL4CV/blob/main/notebooks/logistic_regression_example.ipynb)
 - [Logistic regression slides](http://cs230.stanford.edu/files/C1M2.pdf)

#### What to submit
Upload the notebook with corresponding annotations and output in HTML format. 

- [Upload link](https://www.dropbox.com/request/TbGpgVmLv21E8CNtmpKb)

### Lecture 6: Backpropagation

In this lecture, we cover the backpropagation algorithm used to update the network parameters.

- [Lecture 6 slides]({{site.url}}lectures/lec06_backprop.pdf)

### Lecture 7: Convolutional neural networks

In this lecture, we take on the fundamentals of convolutional neural networks (CNNs).

- [Lecture 7 slides]({{site.url}}lectures/lec07_cnn.pdf)