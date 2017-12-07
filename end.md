---
title: Results
notebook: end.ipynb
nav_include: 3

---

## Results, Conclusions, and Future Work

The figure proposed in [Modeling](https://mraje16.github.io/cs109-final_project/modeling.html) summarizes the final algorithm we proposed to detect and classify the different stages of dementia. The strength of the result is that it can give a more thorough classification of MCI, and can be a single comprehensive test score to determine future course of treatment. The weakness of the test lies in the lack of a predictive ability, i.e. while it can classify the patient based on the current tests and other information, it cannot determine the prospective diagnosis over the coming months, which is a vital point of consideration especially for Alzheimer’s patients. 

**In future, we would have liked to conduct the following analyses:**
1. We take longitudinal data into account and more thoroughly treat the decline or alleviation from/to Dementia of the same patient. This can possibly be done by creating polynomial features for the interactions between different test dates and then  creating a feature based on “months since baseline”. 
2. We can look the brain imaging data to determine if our created meta-test algorithm is able to predict the change in diagnosis over a fixed period of time. 
3. It will be interesting to determine which test most accurately correlates with the change in the  brain shape. 
4. We would like to use HCi/BCI as response variable and cognitive tests as predictor to understand which cog-test provides insight into which brain measurement.
