# BMLS

The demo code of BMLS in the paper of "Bayesian multi-label learning with sparse features and labels, and label co-occurrences", _Artificial Intelligence and Statistics (AISTATS)_ 2018. [Paper](http://proceedings.mlr.press/v84/zhao18b.html)

# Run BMLS

0. The code is a mixture of Matlab and C++. The code has been tested in MacOS and Linux (Ubuntu). To run it on Windows, you need to re-compile all the .c files with MEX and a C++ complier.

1. Requirements: Matlab 2016b (or later).

2. We have offered the Bibtex dataset used in the paper, which is downloaded from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) and stored in MAT format:

**bibtex.mat** contains:
* X_tr, X_te: a N by D feature (sparse and binary) matrix for N instances with D features for training and testing, respectively.
* Y_tr, Y_te: a N by L label (sparse and binary) matrix for N instances with L labels for training and testing, respectively.

**bibtex_missing_label.mat** contains:
* Y_tr: a N by L label (sparse and binary) matrix for N instances with L labels for training, where we randomly removed 80% entries from the label matrix.

**bibtex_missing_instance.mat** contains:
* X_tr: a N by D feature (sparse and binary) matrix for N instances with D features for training, where we reduced the size of training instances to 20%. 
* Y_tr: a N by L label (sparse and binary) matrix for N instances with L labels for training, where we reduced the size of training instances to 20%.

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

3. Run the demos:

-```demo.m```: run the original version of BMLS on bibtex.

-```demo_missing_label.m```: run BMLS with label co-occurrences in the case with missing labels. 

-```demo_missing_instance.m```: run BMLS with label co-occurrences in the case with fewer traning instances. 

# Notes

1. ```CRT_sum_mex.c```, ```Multrnd_Matrix_mex_fast.c``` and ```Multrnd_mijk.c```, ```truncated_Poisson_rnd.m``` are borrowed from [NBP_PFA](https://mingyuanzhou.github.io/Softwares/NBP_PFA_v1.zip) and [EPM](https://github.com/mingyuanzhou/EPM), respectively, of [Mingyuan Zhou](https://mingyuanzhou.github.io). If you want to use the above code please cite the related papers.

2. The code has no support but if you find any bugs, please contact me by email (ethanhezhao@gmail.com).
