# post-processing-TO-results
The Python code performs a design optimization process: topology optimization (stage 1), geometry extraction (stage 2) and shape optimization (stage 3). This project started during my MSc. Thesis and the Delft University of Technology. After finishin my studies, the research continued. Resulting in this piece of example code for 2D case studies.

## Getting Started
The file variables.py allows the user to set all variables for the design optimization, e.g. the domain size and the type of case study. To run the process, simply run DesignOpt_2D.py. In your Windows command prompt, go to the directory where all files are located and simply type 'python DesignOpt_2D.py'.

### Prerequisites
The program depends on several other packages: Numpy, Scipy, Matplotlib, Pypardiso (optional, but faster direct solver than scipy's spsolve)

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
The topology optimization of stage 1 is a Python version of the 99-line Matlab code published by DTU (http://www.topopt.mek.dtu.dk/Apps-and-software/Topology-optimization-codes-written-in-Python).
A big thanks to J.C. Bakker, M. Barbera, N.D. Bos and S.J. van Elsloo for the help in improving the speed of the post-processing part (stage 2 and 3).
