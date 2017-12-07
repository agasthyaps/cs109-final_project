---
title: Alzheimer’s Disease and Cognitive Impairment Prediction
nav_include:0
---


  ![alt text](https://www.alz.org/braintour/images/alzheimer_brain.jpg "Alzheimer's Brain"){: margin: 20px}





## Problem Statement and Motivation

Alzheimer's is a type of dementia that causes impairments and gradual decline in memory, thinking and behaviour. It is an irreversible process typically occurring in people over the age of 60 and is found to relate with accumulation of Amyloid Beta protein in the patients' brain. While some drugs and non-drug treatments help in controlling cognitive and behavioral symptoms, the disease currently has no cure. This has resulted in Alzheimer's being the third leading cause of death in old people and heavy economic and social impact of the disease.



Early detection of Alzheimer's can greatly impact the potency of any therapy. There are multiple diagnostiic tests, yet the diagnosis cannot be made with confidence between Cognitively Normal (CN), Mild cognitive impairment (MCI) and Dementia (AD) categories. In this project, we evaluate the effectiveness of individual cognitive tests in Dementia detection. The team was motivated to create a meta-classifier which builds on top of existing tests and can serve as a standalone entry-test to determine an algorithm for better identification, especially of the messy MCI category. We also study how and which cognitive tests are useful in detecting brain measurements.

​    

## Literature Review and Related Work

The Alzheimer’s Disease Neuroimaging Initiative (ADNI) data used in this project and curated by [USC](http://adni.loni.usc.edu/), also provides the [Procedures Manual](http://adni.loni.usc.edu/wp-content/uploads/2012/10/ADNI3-Procedures-Manual_v3.0_20170627.pdf) containing literature about the protocol used to conduct each cognitive tests used. 

Other web and journal-based resources referred are:

1. Moradi, Hallikainen, Hänninen, & Tohka. (2017). Rey's Auditory Verbal Learning Test scores can be predicted from whole brain MRI in Alzheimer's disease. NeuroImage: Clinical, 13, 415-427. 

   RAVLT Immediate and percentage of forgetting are chosen for the study here because they ‘highlight different aspects of episodic memory, learning and delayed memory’.

2. Science Direct Article on “Gray Matter”, http://www.sciencedirect.com/topics/medicine-and-dentistry/gray-matterDescribes 

   Predicting RAVLT scores from MRI based gray matter density images by applying elastic net linear regression forming a multivariate brain atrophy pattern predicting the RAVLT score.

3. Farias, Sarah Tomaszewski et al. “The Measurement of Everyday Cognition (ECog): Development and Validation of a Short Form.” Alzheimer’s & dementia : the journal of the Alzheimer’s Association 7.6 (2011): 593–601. PMC. Web. 7 Dec. 2017.

   Helped understanding the way Everyday cognition tests work and what is the role of the informant/study partner.

4. Chen et al., 2011. Characterizing Alzheimer's disease using a hypometabolic convergence index. Neuroimage, 56(1), pp.52–60. 

   We used this article as a reference while creating our own meta-indicator, inspired by the logic of the HCI.

5. Teng, E., Becker, B. W., Woo, E., Knopman, D. S., Cummings, J. L., & Lu, P. H. (2010). Utility of the Functional Activities Questionnaire for distinguishing mild cognitive impairment from very mild Alzheimer’s disease. Alzheimer disease and associated disorders, 24(4), 348.

   Helped understand how and why the FAQ test is administered

   ​



## Team

* [Mehul Smriti Raje](https://github.com/mraje16) (ME in CSE, SEAS, Harvard University)
* [Agasthya Shenoy](https://github.com/agasthyaps) (MEd, GSEd, Harvard University)
* [Neeti Nayak](https://github.com/neetinayak) (MDE, GSD, Harvard University)
