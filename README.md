# EMB3Rs-programming
Shared repository Linde&amp;Sergio


#####################################################################################
There are a total of 4 test cases: Pool Base, Pool EnergyBudget, Pool Network Base, Pool Network EnergyBudget

The codes uses a combination of functions: 
1. load_data()
	- Change the 'path' variable to where the 'Linde_data' folder is located
2. load_network() 	
	- Change the 'path' variable to where the 'Linde_data' folder is located
3. save_tocsv()
	- Change the 'path' variable to the folder you want the dataframes to be saved

The only variable that has to be changed to run these codes 

######################################################################################
Using the Plot_figures code
# -------------------------

1. Change the path where the 'CSV' files are located
2. Change the 'modelName_base' and 'modelName_EB' to network being analyzed
	(e.g: modelName_base = 'Pool_base'; modelName_EB = 'Pool_EB')


How it works:
1. Input two models (Names: (Pool_Base, Pool_EB), (PoolNetwork_Base, PoolNetwork_EB))
2. If model is 'Network' the market clearing price is computed as a weighted price of two zones.


##############################################################################################################
# Notes:
1. It seems that the there is a difference in the total consumed energy between the EB and the base
2. The difference is amplified in the network option
3. Some of the network pipes do not have 'Heat loss', therefore not able to compute total_loss for gis_data
