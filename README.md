# Intensive Care-Urgency (IC-U)

This is the GitHub repo for my capstone project, IC-U, for The Data Incubator. IC-U is a web-based app that uses patient demographic and clinical information recorded during his/her stay at a hospital to predict an urgency score for potential intensive care needs and an etimated length of stay in the ICU. To see the app in action, go to [ https://ic-u.herokuapp.com/](https://ic-u.herokuapp.com/). 

## Introduction
ICU care costs and short-term capacity are major pain points for hospitals across the country. Using the MIMIC-III database and a combination of random- forest, ridge, and logistic regression, IC-U culminates in an web app that uses patient demographic and clinical information to produce an urgency score (aptly called "Intensive Care - Urgency", or IC-U) for needed intensive care and an estimated length-of-stay in the ICU. 

By predicting intensive care urgency and length of stay, hospitals can better optimize patient care, flow, and logistics, and better prepare for anticipated spikes in ICU capacity needs. In turn, this information has the potential to improve patient outcomes and decrease operating costs. 

Visit the MIMIC-III database here: [https://mimic.physionet.org](https://mimic.physionet.org).

## Motivation
### The problem
Over 5 million patients annually require some form of intensive care, which requires continual and often invasive monitoring, as well as near 1-to-1 nurse-to-patient staffing in order to treat patients effectively. As a result of the complicated nature of ICU care, ICUs are one of the leading drivers of hospital costs in the US today. Hospitals are currently spending $80B a year in ICU care, and these costs are expected to double by 2030. 

Additionally, median ICU occupancy rates in the US already sit around 75%, and can be as high as 86%. Given that the number of ICU beds in a hospital can range from as low as 6 to 67 beds, many hospitals face extreme challenges managing spikes in ICU needs. 

It is no surprise then that both ICU operating costs and occupancy rates are major pain points for hospitals across the country and that innovations are necessary to help drive costs down.

### My solution
The basis of IC-U is to quantify a risk factor (called the "IC-U" factor) reflecting intensive care urgency for patients that present to a hospital and predict an estimated length of stay in the ICU. My solution would take in a patient’s health and demographic information and estimate (1) the likelihood that s/he may need immediate intensive care and (2) the likely length of stay (LOS) ultimately required in the ICU.

### Value proposition
Intensive Care Urgency and LOS estimates can help hospital staff to more quickly identify patients most in need of intensive care and predict ICU occupancy rates in the near term. This will allow hospitals to optimize patient triaging and monitoring and better manage staffing and capacity concerns. Combined, these benefits could both improve patient care and reduce operating costs.

## Model Design
IC-U uses the MIMIC-III data base, a freely accessible dataset representing data from ~60,000 ICU stays across 40,000 patients. To quantify intensive care urgency, I calculate the time between hospital admission and ultimate ICU admission and bin these times into 4 distinct categories:

- Immediate: ICU admission <1 hour from hospital admission
- Urgent: ICU admission <24 hours from hospital admission
- Questionable: ICU admission <5 days from hospital admission
- Stable: ICU admission >5 days from hospital admission

Two separate models are then fit to the data:
- A multi-class "one-versus-all" logistic regression classifies patients into one of the four urgency categories.
- A gradient-boosting regression estimates the anticipated LOS in the ICU.

## How it Works
IC-U works as follows:

- Hospital staff inputs patient information into a web form
- Upon submission, the predictive models analyze the patient data and compute both the IC-U factor and an estimated length of stay.
- Finally, the app displays the results with a description of the main contributors to the IC-U score.

With more information about the size and occupancy of the hospital, these results can be utilized to optimize patient care and staffing and serve as an early warning system for ICU capacity issues.
And as more information for a patient is gathered, the inputs can be updated and a new IC-U factor and LOS can be calculated and displayed as needed.

## Exploratory notebooks
- explore_chartEvents_and_icustays.py
- explore_drugCombos_and_icustays.ipynb
- explore_drugCombos_and_icustays.py
- explore_features_and_timetoicustay.ipynb
- explore_features_and_timetoicustay_updated.ipynb
- plot_summary_drugCombos_icustays.ipynb
- plot_summary_chartEvents_and_icustays.ipynb

## Processing notebooks
- batchGetStats.py
- batch_load_and_sort_notes.ipynb
- batch_merge_notes_with_df.ipynb
- batchidentify.py
- customTransformers.py

## Model building notebooks
- pipeline_LOS_models.ipynb
- pipeline_base_models.ipynb
- pipeline_ensemble_sameday_classifier.ipynb
- pipeline_select_data.ipynb
- pipeline_urgency_score.ipynb
- pipeline_urgency_speedup_and_optimize.ipynb


## Model evaluation notebooks
- urgency_model_evaluation.ipynb

## Scratch notebooks
- templatePipeline.ipynb
- scratch_organize_demographic_data_for_model.ipynb
- zzz_data_exploration.ipynb
- testFuzzyMatchScoreThresholds.ipynb

## Model Performance
The IC-U risk factor model has an overall accuracy of 65%. The most likely predictions for each risk category tended to be the true categories, as can be seen by the higher percentages along the diagonal in the confusion matrix. In general, the predictions for the riskier categories (urgent and immediate) tended to be more accurate than those for the less risky categories.

Feel free to interact with the figure below to explore the performance of the model in more detail. The dropdown menu allows you to view the performance breakdown of the model by raw counts, counts normalized by the true labels, counts normalized by the predicted labels, or counts normalized by the total number of entries in the test set. View the model performance [here](https://ic-u.herokuapp.com/performance).
