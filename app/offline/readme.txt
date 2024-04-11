In this directory are all the scripts for processing the raw results from the model.

- Examples of raw results are contained in the example_s3_files subdirectory
- analyze.py does most of the processing, which results in all the "results" files from the main directory
- Other subdirectories are for other offline analyses. An "offline analysis" is one where the data are fixed (i.e., items and training ratings), but different aspects of the tests are explored. For example: 
	* different_reps_offline_testing -- simulated model performance under different representations
	* redo_41_knn1_2 -- explore how KNN performs if the number of neighbors is set to different values
	* redo_62_no_norm_k1, redo_study_58 -- both of these fixed a bug where the app was outputing differently scaled predictions
	* reoptimize -- many scripts for figuring out which hyperparameter settings may perform well in various studies
	* understanding_prefs -- this is where the interpretation results from the paper were done
	
Note: these analysis files have their own requirements.txt.