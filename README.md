Description
------------------------
Python3 code and data for the paper "A Deep Learning Model for Predicting NGS Sequencing Depth from DNA Sequence"

Each of the three folders ("NGS_human_SNP", "NGS_synthetic" and "HYB_DSP") contains a seperate copy of the python code of our deep learning model, as well as the corresponding dataset for training and validation (NGS human SNP panel, NGS synthetic panel and DNA strand displacement/hybridization). For details about running the code and interpreting the results, please refer to the "README.txt" inside each folder. 

We also provide the raw NGS data in the "Raw NGS data" folder, and the raw fluorescence data for calculated kinetics rate constants in the "Raw fluorescence data" folder. 


Requirements
------------------------
GPU is required. 
operating systems: tested on Windows10 and Amazon Linux 25.0
python (tested with version 3.7.6)
tensorflow (tested with version 1.15.0)
docopt (tested with version 0.6.2)
matplotlib (tested with version 3.2.2, for plotting results)

For the SNP panel, the training time for each epoch is roughly 10 seconds while taking less than 3 gigabytes memory of a graphics processing unit. For the other two datasets the training time is shorter. 


Installation
------------------------
It is recommended to use Anaconda/Miniconda to install python and all the required packages. The installation process would take a few minutes. 
On Linux for example: 
1. Get the latest Miniconda3 for Linux 64-bit:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

2. Install Miniconda3. Please follow the default installation settings. Restart the shell after installation.\
bash Miniconda3-latest-Linux-x86_64.sh\

3. Create and activate new python environment\
conda create --name tf1\
conda activate tf1

4. Install required packages\
conda install python=3.7.6\
conda install -c conda-forge docopt=0.6.2\
conda install -c anaconda tensorflow-gpu=1.15.0\
conda install -c conda-forge matplotlib=3.2.2
