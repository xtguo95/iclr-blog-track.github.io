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

{% include uk49_tour.html %}

### DNN to solve TSP

### Attention Mechanisms and Transformers

## Model Structure and Training Algorithm

### Encoder Structure

### Decoder Structure

### Training Algorithm

## Experiments

## Discussions

## References

<a name="Kool">Wouter Kool, Herke van Hoof, Max Welling. Attention, Learn to Solve Routing Problems! The International Conference on Learning Representations (ICLR), 2019. </a>
