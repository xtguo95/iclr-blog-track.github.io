---
layout: post
title: Attention Solves TSP: Insights on Solving Real-world Instances
authors: Guo, Xiaotong, Massachusetts Institute of Technology
tags: [Traveling Salesman Problem, Combinatorial Optimization, Attention Network]
---

This post discusses a recent publication at ICLR 2019 [[Kool et al., 2019]](#Kool) on applying attention layers for
solving the well-known combinatorial optimization problem, Traveling Salesman Problem (TSP).

On the other hand, this post is inspired by a recent research challenge held by Amazon, [Amazon Last Mile Routing](https://routingchallenge.mit.edu/),
where the author participated with two excellent teammates Qingyi Wang and Baichuan Mo. The challenge encourages paticipants
to develop  innovative approaches leveraging non-conventional methods (e.g., machine learning) to produce solutions to the route sequencing problem,
which could outperform traditional, optimization-driven operations research methods in terms of solution quality and computational cost.

As a part of the exploration process for competing in this challenge, author's team implemented the attention-based model proposed
by Kool et al. and gained insights on how the proposed model performed on real-world routing problems. The author will summarize
the attention-based model and discuss its performances on real-world instances and comparisons with traditional optimization-driven
methods.

## Background

### Traveling Salesman Problem (TSP)

The Traveling Salesman Problem (TSP) is a traditional combinatorial optimization problem, which is also an NP-hard problem
that can not be solved with polynomial time algorithms. The TSP requires to find the shortest distance route that visits all
given locations and return to the original location.

![TSP instance]({{ site.url }}/public/images/2022-05-03-attention-solves-tsp/TSP.png)

[comment]: <> (*<p align="center">Figure 1. TSP instance </p>*)

TSP is typically solved with optimization-driven approaches where TSP is modelled as an Integer Linear Program (ILP) and solved
with off-the-shelf ILP solvers, e.g., Gurobi and CPLEX. However, large-scale TSP instances can not be solved optimally with
solvers. Therefore, many researchers have focused on developing heuristics to solve the large-scale TSP instances, where their
proposed algorithms are not guaranteed to be optimal. Some interesting real-world solved large-scale TSP problems can be found
[here](https://www.math.uwaterloo.ca/tsp/uk/index.html). For instance, an optimal TSP tour to visit 49,687 pubs in UK have
been found through optimization-driven methods.

### DNN to solve TSP

On the other hand, the success of DNN in the past decade has drawn the attention of applying the DNN into solving the combinatorial
optimization problem.

### Attention Mechanisms and Transformers

## Model Structure and Training Algorithm

### Encoder Structure

### Decoder Structure

### Training Algorithm

## Experiments

## Discussions

## References

<a name="Kool">Wouter Kool, Herke van Hoof, Max Welling. Attention, Learn to Solve Routing Problems! The International Conference on Learning Representations (ICLR), 2019. </a>
