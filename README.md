# EECS545-project-STGCN
This is the final project for EECS545.  
Group Member:  
Anonymous before reviewing

## DataCollection
This is a file to generate matrices and plots with the data downloaded from the website.  
Data souce is http://pems.dot.ca.gov/.
The data collected is about the traffic speed and flow from Nov. 1 to Nov.30 in District 8. And to reduce the computation cost, we choose 200 stations as samples in most denest area.  
If you want to use a different dataset on the website, some parameters should be modified.

## Model
This folder contains STGCN model.  
The original code is from https://github.com/FelixOpolka/STGCN-PyTorch.  
We modified parts of the code and added a different function to capture the spatial features.
