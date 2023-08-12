# JurySelection-Replication_Package

Replication package for Andrea Moro and Martin Van der Linden: "Exclusion of Extreme Jurors and Minority Representation: The Effect of Jury Selection Procedures." (2023), Forthcoming, Journal of Law and Economics. The paper is available at https://arxiv.org/abs/2102.07222 or from the authors' websites

Overview
--------

The code in this replication package constructs the simulated juries used to generate figures, tables, and other results in Moro and Van der Linden (2023) using Python.

Data Availability and Provenance Statements
----------------------------

No external data is used in this project

Computational requirements
---------------------------

### Software Requirements

- Python 3.9.1
  A pip freeze text file with required packages is included under Environments
  In particular, the following are needed
  - `numpy` 1.20.2
  - `matplotlib` 3.6.2
  - `scipy` 1.6.2
  - `labellines` 0.6.0

We ran the code both using the Spyder 5.0.5 GUI, from Python's command line, and with the included Docker image

### Memory and Runtime Requirements

The code does not require large memory. The code was last run on a 1.7 GHz Quad-Core Intel Core i7 Intel-based laptop with MacOS version 11.4. Execution time was less than 30 minutes.

Description of programs/code
----------------------------

- class_model_types.py contains the main Jurymodel class used to simulate juries.
- juryConstruction.py contains code to generate all simulated juries needed to generate the paper figures and tables. Juries are saved in pickle format under output/
- juryPlotsAndResults.py contains code to generate all figures and tables. Output is saved under exhibits/
- class_jury_Statdisc.py
extension of class_model_types.py to run simulations for the statistical discrimination section
- juryStatdisc_sims.py
generates all simulated juries needed for the statistical discrimination section
- juryStatdisc_plots.py generates figures for the statistical discrimination section.

### License for Code

- the code is © Moro and Van der Linden. Please contact the authors if interested in using or modifying this code.

Instructions to Replicators
---------------------------

## Using Docker
A Dockerfile is included under directory Environments to replicate an environment suitable for proper code execution
After creating the docker image (with tag juryimage, that is run "docker build -t juryimage Environment/"), run the following command from the project root directory:

docker run --init -it -v $(PWD)/:/juryselection -w /juryselection/Code juryimage ./execute_all.sh

## Using a python installation
A pip freeze text file with required packages is included under Environments

## Notes

a) running the code with docker will not format some of the figure labels with LaTeX. To generate LaTeX-formatted figures you need to have a LaTeX distribution in your system and then run the python commands in the following sequence using Code as working directory

  1) python juryConstruction.py
  2) python juryPlotsAndResults.py
  3) python juryStatdisc_sims.py
  4) python juryStatdisc_plots.py

b) Some jury construction commands use the multiprocessing package with 6 processors to speed up computations. You may change the nprocs variable at the beginning of juryConstruction.py and juryStatdisc_sims.py at your convenience

List of figures, tables and programs
---------------------------

| Figure/Table # | Program                  | Line | Output file
|----------------|--------------------------|-------------|-------------------------------|
| Figure 1       | TixZ-generated in LaTeX  |      |                                      |
| Figure 3       | juryPlotsAndResults.py   | 140  |  betapdfs.pdf                        |  
| Figure 4       | juryPlotsAndResults.py   | 219  |  prop1-beta-all.pdf                  | 
| Figure 5       | juryPlotsAndResults.py   | 302  |  prop2-uni.pdf                       | 
| Figure 6       | juryPlotsAndResults.py   | 378  |  counterall.pdf                      | 
| Figure 7       | juryPlotsAndResults.py   | 635  |  minority-representation.pdf         | 
| Figure 8       | juryPlotsAndResults.py   | 710  |  nchallenges-extr-minority.pdf       | 
| Figure 9       | juryStatdisc_plots.py    | 196  |  std-logitnorm.pdf                   | 
| Figure 10      | juryStatdisc_plots.py    | 251  |  std-density-1.pdf                   |
| Figure 11      | juryPlotsAndResults.py   | 792  |  median.pdf                          | 
| Figure 12      | juryPlotsAndResults.py   | 1017 |  balanced.pdf                        |
| Figure 13      | juryStatdisc_plots.py    | 320  |  std-beta.pdf                        | 


## Acknowledgements
Social Science Data Editors template README file https://github.com/social-science-data-editors/template_README
