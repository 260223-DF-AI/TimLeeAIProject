# TimLeeAIProject
This project is based on Task G:
Track G: AutoIntel (Driver Safety & Diagnostics)
- The Problem: A fleet management company needs to monitor driver alertness and vehicle dashboard health.
- Vision Task: Classify driver status (Alert, Drowsy, Distracted) or dashboard warning lights.
- Reasoning Task: Use Dialog State management to handle driver alerts and ReAct to suggest the nearest rest stop or service center based on GPS and vehicle status.

This project uses the [statefarm distracted driver image database](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data).

How the app works:
1. post request triggers a series of 10 images to be sent to the CV model, passed through 1 at a time
2. CV model classifies image into categories:
    a. c0- safe driving
    b. c1- phone usage
    c. c2- radio
    d. c3- drinking
    e. c4- reaching
    f. c5- hair/makeup
    g. c6- turned to passenger
3. LLM gets confidence results from CV model, decides whether to immediately alert or not
4. after all images have been passed, LLM generates a report on incidents/safety of the trip
