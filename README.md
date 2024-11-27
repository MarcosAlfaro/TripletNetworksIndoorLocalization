# Hierarchical localization with panoramic views and triplet loss functions
Marcos Alfaro, Juan José Cabreraa, María Flores, Oscar Reinoso and Luis Payá
Abstract:
The main objective of this paper is to tackle visual localization, which is essential for the safe navigation of mobile robots. The solution we propose employs panoramic images and triplet convolutional neural networks. We seek to exploit the properties of such architectures to address both hierarchical and global localization in indoor environments, which are prone to visual aliasing and other phenomena. Considering their importance in these architectures, a complete comparative evaluation of different triplet loss functions is performed. The experimental section proves that triplet networks can be trained with a relatively low number of images captured under a specific lighting condition and even so, the resulting networks are a robust tool to perform visual localization under dynamic conditions. Our approach has been evaluated against some of these effects, such as changes in the lighting conditions, occlusions, noise and motion blurring. Furthermore, to explore the limits of our approach, triplet networks have been tested in different indoor environments simultaneously. In all the cases, these architectures have demonstrated a great capability to generalize to diverse and challenging scenarios. The code used in the experiments is available at https://github.com/MarcosAlfaro/TripletNetworksIndoorLocalization.git.

In this repository, the programmes used to adress robot localization can be found.

The main information about this work is described below:

- A triplet network architecture has been used, which is shown in script "models.py".
- Omnidirectional images from a public database have been converted to panoramic. 
- To validate the proposed method, we have chosen COLD database. It can be downloaded from this website: https://www.cas.kth.se/COLD/
- Two different localization approaches have been adressed:
  1. Hierarchical localization: the robot position is obtained in two steps (coarse step and fine step)
  2. Global Localization: the robot position is obtained in a single step
  
- The code is divided in three experiments: 
  * Experiment 1 adresses an exhaustive comparation amongst different triplet loss functions and its parameters. In this experiment, only Freiburg dataset has been used.
  * Experiment 2 evaluates triplet networks against certain effects that appear frequently in images captured by mobile robots, which are: Gaussian noise, occlusions and motion blurring.
  * Experiment 3 tests triplet networks it in three different environments simultaneously: Freiburg, Saarbrücken A and Saarbrücken B.
  * Inside each folder, the training and test scripts are included
  * Moreover, the scripts used to generate the CSV files for each training and test process are also included
  * Additional code is provided as well, some of them are essential to run the programmes (models.py, losses.py) and others can be used to generate the figures shown in the manuscript.
  * config folder includes a YAML file where some parameters can be defined
 
