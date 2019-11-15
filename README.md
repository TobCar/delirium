# Predicting delirium in ICU patients using time series physiological data and convolutional neural networks
Authors: Tobias Carryer, Maxwell Berkowitz, Shreyansh Anand, Logan Wilkinson, Justin Harrison, John Gord Boyd

[Abstract](https://cccf.multilearning.com/cccf/2019/eposters/285175/tobias.carryer.predicting.delirium.in.icu.patients.using.time.series.html?f=listing%3D0%2Abrowseby%3D8%2Asortby%3D1%2Asearch%3Ddelirium%2Bneural+networks) was published at the Canadian Critical Care Forum in November 2019.

## Abstract

**Introduction**: Delirium is a preventable yet highly common complication of critical illness, affecting up to 80% of patients who require invasive life sustaining therapies (Barr, Fraser, Puntillo, Ely, Gélinas, Dasta, et al., 2013). To predict whether the patient will experience delirium, nurses can apply the PRE-DELIRIC model (Wassenaar, Schoonhoven, Devlin, van Haren, Slooter, Jorens, et al., 2018) when a patient is admitted into the ICU. The PRE-DELIRIC model achieves a sensitivity and specificity of 60% and 65% respectively, leaving much room for improvement. Auto-DelRAS (Moon, Y. Jin, T. Jin, and Lee, 2018) is another approach, it uses electronic health records and logistic regression to achieve a sensitivity of 63% and a specificity of 90%. To the best of our knowledge, no previous research has used high fidelity time series data to predict delirium.
 
**Objectives**: Develop a machine learning model to predict whether a patient will experience delirium at any point during their stay in the ICU given derived regional cerebral oxygenation (rSO2), peripheral oxygen saturation (SpO2), heart rate (HR), and mean arterial pressure (MAP) time series data.
 
**Methods**: We derived our data from the Cerebral Oxygenation and Neurological outcomes FOllowing CriticAL illness (CONFOCAL, NCT02344043 www.clinicaltrials.gov) study, a single centre observational study assessing the link between near-infrared spectroscopy, rSO2, and delirium in patients admitted with respiratory failure and/or shock to a tertiary ICU. Participants were divided into training (n=37), cross validation (n=37), and withheld test (n=6) sets. The training and cross validation sets contained the same 37 subjects. For each subject trained on, the initial 80% of the collected data went in the training set and the remaining 20% went in the cross validation set. We normalized rSO2, SpO2, HR, and MAP for the whole data set, regardless of what subject contributed each value. Then, we augmented each of the time series into Gramian Angular Summation/Difference Fields and Markov Transition Fields (Wang and Oates, 2015), producing a 128x128x3 volume for every 128 minute rolling window of subject data. The volumes for the four time series were stacked together into 128x128x12 volumes. A convolutional neural network trained on the volumes and predicted whether the data was from a patient who experienced delirium at any point in the study. The model architecture was made up of a 2D convolutional layer followed by batch normalization, global average pooling, and a densely connected layer.
 
**Results**: The cross validation set achieved a sensitivity of 82.609% and a specificity of 71.429%. Similarly, the test set achieved a sensitivity of 100% and a specificity of 66%, verifying variance was minimized. Most notably, 84% of the patients in the cross validation set and 67% of patients in the test set were accurately classified 100% of the time even when using data from days prior to a patient entering delirium.

**Conclusion**: High fidelity physiological time series data can be used to predict delirium with higher sensitivity and specificity than attempts with other data sources. Our predictions were consistently accurate regardless of the length of time before the onset of delirium. We propose the consistency is evidence the interactions between rSO2, SpO2, HR, and MAP put patients at risk of delirium. Further analysis into the patterns our model learned may shed insight into the pathophysiology of delirium.

## Running the project

`pipeline.py` contains the pipeline we used to train the model.

`testing_the_model.py` evaluates the model on the cross validation set and the withheld test set.
