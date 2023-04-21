This repository implements different RL techniques on the LunarLander Module.

There are 4 algorithms implemented:
1. Sarsa
2. DQN
3. Dueling DQN
4. REINFORCE

All respective folders can be run individually and their is no cross-dependencies between the folders.

Each folder will have two types of main files.

1. main file without addition of gaussian noise in the observation
2. main file with addition of gaussian noise in the observation

All files have assumed that there could be engine failure and its probability is kept to be 20 percent.

The result obtained from these files are:
1. png image plotting reward function vs episodes
2. text file which will have rewards for all the episodes

There is a plotting script in Results folder which can be used to combine and obtain beautiful plots using plotly

