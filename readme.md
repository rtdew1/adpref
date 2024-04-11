# Code to accompany: Adaptive Preference Measurement with Unstructured Data
## By Ryan Dew

This directory is broken into two subdirectories: 

1. app - This contains all the code to deploy our testing framework. 

	* For readers interested in just how the models work, please reference: app/backend/model. 
	* For the data used in the model, please reference: app/backend/data. Note: the embeddings and images provided by the company have been omitted for confidentiality. The VGG-19 embeddings are provided. 
	* Under app, the subdirectory "offline" contains all of the scripts to process results and run any offline analyses
	
2. results_and_analysis - This contains all of the results from using the app live, plus the analysis scripts to generate all plots and tables in the paper. 
	* To generate all the tables and figures from the paper, see the scripts under /analysis. 
	* All raw respondent data can be found in: results/raw 
	* Minimally processed (and much more user-friendly) versions of the results can be found in the other subdirectories of /results.