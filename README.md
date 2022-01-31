# ANN-CUDA-Pybind11
ANN Python module utilizing pybind11, cudnn and the CUDA programming language.

Instructions - Build from source:

 - Clone the repo and open the visual studio project.
 - Make sure your desired cuda, cudnn, and pybind11 versions are linked in VC++ Directories > Library Directories.
 - Compile the module.
 
Instructions - Run the module:

 - Add compiled module to the folder where your project is located or to your python site-packages.
 - Set up dataset as 4D numpy arrays (n, c, h, w)
 - Convert the arrays in to Tensor instances and then move to CUDA device.
 - Set up network architecture
 - Train network
 
A demo notebook is included to show how to run through all of the steps on the mnist dataset.

*If you're looking for a sequential/c++/cuda implementation, https://github.com/francislabountyjr/ANN-CUDA-Framework

**If you have any questions/concerns please reach me at labounty3d@gmail.com
