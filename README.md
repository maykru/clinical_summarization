# Clinical Summarization
(work in progress)
## Overview
This repository contains code for the paper 'Zero-shot Large Language Models for Long Clinical Text Summarization with Temporal Reasoning'

Link: https://arxiv.org/abs/2501.18724

## How to run
1. run get_target_population.py to get the filtered MIMIC files for the target population (age >= 65, LOS >= 3 days)
2. depending on task, run either get_chronologies_DS.py (discharge summarization) or get_chronologies_AP.py (assessment and plan generation)

TODO: explain script arguments etc, add instructions for running generation 

## Tasks
- Discharge summarization: given a patient chronology, generate the three sections of a discharge summary (Diagnosis, Brief Hospital Course, Discharge Instructions)
- A&P generation: given a patient chronology and previous progress notes, generate the Assessment and Plan sections of the current day's progress note