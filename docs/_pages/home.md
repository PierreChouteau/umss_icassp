---
title: ""
layout: single
permalink: /
author_profile: false
classes: wide
header:
  overlay_color: "#000"
  overlay_filter: "0.6"
#   overlay_image: /images/particles.jpg
excerpt: #"A fully differentiable model for unsupervised singing voice separation"
---

Welcome to the demo website. All the mixtures examples presented in [Audio Examples](./audio.md) are taken from the test set. 


# Abstract

<html>
<div style="text-align: justify">
<p>
A novel model was recently proposed by <a href="https://ieeexplore.ieee.org/document/10058592" target="_blank" rel="noopener noreferrer">Schulze-Forster & al.</a> for unsupervised music source separation. This model allows to tackle some of the major shortcomings of some modern source separation frameworks. Specifically, it eliminates the need for isolated sources during training, performs efficiently with limited data, and can handle homogeneous sources (such as singing voice). Nevertheless, this model relies on an external multipitch estimator and incorporates an adhoc voice assignment procedure. In this paper, we
propose to extend this framework and to build a complete, fully differentiable model by integrating a multipitch estimator and a novel differentiable voice assignment module within the core model. We show the merits of our approach though a set of experiments, and we highlight in particular its potential for processing diverse and unseen data.
</p>

</div>
</html>

*Index Terms - Unsupervised source separation, multiple singing voices, differentiable models, deep learning*
