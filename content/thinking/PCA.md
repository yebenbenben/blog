---
title: "Dimension Reductionm and Feature Extraction"
date: 2021-06-13T02:50:16-04:00
categories: ["Math"]
tags: ["Data_Science", "Stats"]
slug: PCA
draft: false



---

# Principal Component Analysis
PCA is a wild used technique for dimension reduction, data compression and feature extraction.

There are two comman used definition of PCA which lead to the same algorithm.

* PCA can be defined as orthogonal projection of data to the lower dimension linear space, known as *principal subspce* where the varience of the projected data is maximized.
* PCA can also be defined as finding a low dimension linear projection which minimize the mean squared distance.

# Probablistic PCA (PPCA)

* Advantage: PPCA is better in handeling missing data. It is shown in Bishp's book that with randomnly 30% of the data being truncated, the PPCA can still achieve a similar result as the traditional PCA without truncated data.