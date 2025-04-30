# Channel Charting in Real-World Coordinates with Distributed MIMO


This is the code for the results in the paper
"Channel Charting in Real-World Coordinates with Distributed MIMO", S. Taner, V. Palhares, and C. Studer
(c) 2025 Sueda Taner

email: taners@ethz.ch

### Important Information

If you are using this code (or parts of it) for a publication, then you _must_ cite the following paper:

S. Taner, V. Palhares and C. Studer, "Channel Charting in Real-World Coordinates with Distributed MIMO," in IEEE Transactions on Wireless Communications, 2025.

### How to use this code...

#### Step 1: Download and save the data

- From [dichasus-cf0x Dataset: Distributed Antenna Setup in Industrial Environment, Day 1](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2854&version=3.0), download the specifications file ```spec.json```, the channel files ```dichasus-cf02```, ```dichasus-cf03``` and ```dichasus-cf04``` (which should have the ```.tfrecord``` extension), and their offset estimates (which should be ```reftx-offsets-dichasus-cf0x.json```) into a folder called ```data_raw```.
- Run ```preprocess_dichasus.py```. This will store ```.np``` versions of the CSI, timestamps, and ground-truth positions extracted from the ```.tfrecord``` files in a folder called ```data``` along with AP positions and LoS bounding boxes for each AP.

#### Step 2: Channel charting

- Set your training parameters as explained on top of ```main.py``` and run for the results in the paper. This code does the following:
  - We separate the complete dataset into training and testing samples.
  - We train a neural network for channel charting or positioning according to the settings.
  - We test the channel charting neural network on the test set.


### Version history

Version 0.1: taners@ethz.ch - initial version for GitHub release.

## Acknowledgments
This project makes use of the following external data and code:
- [dichasus-cf0x Dataset: Distributed Antenna Setup in Industrial Environment, Day 1]([https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2854&version=3.0), accessed on 1/3/2025: Our code uses the ```cf02```, ```cf03```, and ```cf04``` datasets.
- [Dissimilarity Metric-Based Channel Charting](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/) by F. Euchner, accessed on 1/3/2025: We use this code for (i) pre-processing the data from  ```tfrecords``` and ```json``` files, and (ii) finding an affine transform that maps a channel chart to real-world positions using ground-truth position labels for our baseline B2.
