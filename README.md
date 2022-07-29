# Gait_analysis
Running gait analysis from video

- Mediapipe: BlazePose part
  - mediapipe_opencv.py: Code that executes the model and predicts the results
- opencv: Openpose part
  - Code: (Needs the model to be executed before)
    - angles.py: Principal code that gets the metrics from the JSON outputs of the Openpose model
    - errors.py: Study of the errors
    - svm.py: Support vector machine method to try to get the metrics
    - svm_not_sr: Same as before but without the step rate metric
  - Images: Example images of some outputs
  - Execute: Command to execute openpose model for our problem
- Results
  - result_analysis.Rmd: R script to get the images and some error analysis using hypothesis tests.
  - Results.xlsx: Table with the results
  - Results_sum: Summary of the results
- TFM-CM.pdf: Explanation of the whole project (spanish)
