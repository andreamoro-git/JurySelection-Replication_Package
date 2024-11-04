# JurySelection-Replication_Package

<p align="center">
!<img alt='minorities in juries' src='https://raw.githubusercontent.com/andreamoro-git/andreamoro-git.github.io/3f2975ad2b7474a7f84d1205d23ea4845cc5be3c/assets/images/std-logitnorm.png' />
</p>

Replication package for Andrea Moro and Martin Van der Linden: "Exclusion of Extreme Jurors and Minority Representation: The Effect of Jury Selection Procedures." (2023), Forthcoming, Journal of Law and Economics. The paper is available at https://arxiv.org/abs/2102.07222 or from the authors' websites

Bibtex Citation:

```
@article{moro-vanderlinden-juryselection-2024,
    title = "Exclusion of Extreme Jurors and Minority Representation: The Effect of Jury Selection Procedures",
    author = "Moro, Andrea and Martin {Van der Linden}",
    year = "2024",
    month = " May",
    journal = "The Journal of Law and Economics",
    volume = "67",
    pages = "295-336",
    url = "http://andreamoro.net/assets/papers/juryselection.pdf"
}
```


Overview
--------

The code in this replication package constructs the simulated juries used to generate figures, tables, and other results in Moro and Van der Linden (Journal of Law and Economics, Forthcoming 2023) using Python.

Data Availability and Provenance Statements
----------------------------

No external data is used in this project

Computational requirements
---------------------------

### Software Requirements

- Python 3.9.17
  A pip freeze text file with required packages is included under Environments
  In particular, the following are needed
  - `numpy`==1.25.2 
  - `matplotlib`==3.7.2 
  - `pandas`==1.5.3 
  - `scipy`==1.10.1 
  - `matplotlib-label-lines`==0.6.0 

We ran the code both using the Spyder 5.0.5 GUI, from Python's command line, and with the included Docker image

### Memory and Runtime Requirements

The code does not require large memory. The code was last run on a 1.7 GHz Quad-Core Intel Core i7 Intel-based laptop with MacOS version 11.4. Execution time was about one hour.

Description of programs/code
----------------------------

- execute_all.sh Executes all code in the appropriate order
- class_model_types.py contains the main Jurymodel class used to simulate juries.
- juryConstruction.py contains code to generate all simulated juries needed to generate the paper and appendices figures and tables. Juries are saved in pickle format under Simulations/
- juryPlotsAndResults.py contains code to generate all figures and tables. Output is saved under Exhibits/
- class_jury_Statdisc.py extension of Jurymodel for the statistical discrimination section
- juryStatdisc_sims.py
generates all simulated juries needed for the statistical discrimination section
- juryStatdisc_plots.py generates figures for the statistical discrimination section

### License for Code

- the code is © Moro and Van der Linden. Please contact the authors if you are interested in using or modifying this code.

Instructions to Replicators
---------------------------

### Using Docker

A Dockerfile is included under directory Environment to replicate an environment suitable for proper code execution

Execute run-docker.sh from the root directory. The file will generate a docker image called juryselectionimage and 
then execute all the necessary code (see run-docker.sh for details)

### Using a Python installation

A pip freeze text file with required packages is included under directory Environment. Execute ```execute_all.sh``` from a shell, or run the python files in the order indicated in Code/execute_all.sh in your preferred client/GUI using Code/ as your working directory

### Notes

a) To speed up computations, some jury construction commands use Python's multiprocessing package with 6 processors as default. You may change the nprocs variable at the beginning of juryConstruction.py (row 17) and juryStatdisc_sims.py (row 20) at your convenience

b) The code needs a LaTeX installation to format figure labels to match exactly the article figures. A minimal latex installation is included in the Dockerfile, fetched from TinyTeX. If for any reason the image fails to generate, you can comment out lines 14-23 in Environment/Dockerfile. The code will execute without LaTeX formatting

List of figures, tables and programs
---------------------------

| Figure/Table # | Program                  | Line | Output file
|----------------|--------------------------|-------------|-------------------------------|
| Figure 1       | TikZ-generated in LaTeX  |      |                                      |
| Figure 2       | TikZ-generated in LaTeX  |      |                                      |
| Figure 3       | juryPlotsAndResults.py   | 140  |  betapdfs.pdf                        |  
| Figure 4       | juryPlotsAndResults.py   | 219  |  prop1-beta-all.pdf                  | 
| Figure 5       | juryPlotsAndResults.py   | 302  |  prop2-uni.pdf                       | 
| Figure 6       | juryPlotsAndResults.py   | 378  |  counterall.pdf                      | 
| Figure 7       | juryPlotsAndResults.py   | 635  |  minority-representation.pdf         | 
| Figure 8       | juryPlotsAndResults.py   | 710  |  nchallenges-extr-minority.pdf       | 
| Figure 9       | juryStatdisc_plots.py    | 196  |  std-logitnorm.pdf                   | 
| Figure 10      | juryStatdisc_plots.py    | 251  |  std-density-1.pdf                   |
| Figure 11      | juryPlotsAndResults.py   | 793  |  median.pdf                          | 
| Figure 12      | juryPlotsAndResults.py   | 1018 |  balanced.pdf                        |
| Figure 13      | juryStatdisc_plots.py    | 320  |  std-beta.pdf                        | 

Note: figures for the external appendix are generated but not included in this table

## Acknowledgements
Social Science Data Editors template README file https://github.com/social-science-data-editors/template_README

Tip for generating the list of figures above: https://andreamoro.net/blog/2021/06/01/generate-list-figures-with-code-references.html
