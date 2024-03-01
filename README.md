# TripletNetworksIndoorLocalization

In this repository, the programmes used to adress robot localization can be found.

The main information about this work is described below:

- A triplet network architecture has been used, which is shown in program "triplet_network.py".
- Omnidirectional images converted to panoramic have been employed. 
- To validate the proposed method, we have chosen COLD database. It can be downloaded from this website: https://www.cas.kth.se/COLD/
- Two different localization approaches have been adressed:
  1. Hierarchical localization: the robot position is obtained in two steps (coarse step and fine step)
  2. Global Localization: the robot position is obtained in a single step
  
- The code is divided into two experiments: 
  * Experiment 1 adresses an exhaustive comparation amongst different triplet loss functions and its parameters. In this experiment, only Freiburg dataset has been used.
  * Experiment 2 explores the limits of the proposed architecture, testing it in three different environments simultaneously: Freiburg, Saarbrücken A and Saarbrücken B.
  * Inside each folder, the training and test programmes are included
  * Moreover, the programmes used to generate the datasets for each training and test process are also included
  * Additional code is provided as well, some of them are essential to run the programmes (triplet_network.py, losses.py) and others can be used to generate the figures shown in the manuscript.
 
