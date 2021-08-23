# JurySelection-Replication_Package

Replication package for Andrea Moro and Martin Van der Linden: "Exclusion of Extreme Jurors and Minority Representation: The Effect of Jury Selection Procedures." The paper is available at https://arxiv.org/abs/2102.07222 or from the authors' websites

Overview
--------

The code in this replication package constructs the simulated juries used to generate figures, tables, and other results in Moro and Van der Linden (2021) using Python.

Data Availability and Provenance Statements
----------------------------

No external data is used in this project

Computational requirements
---------------------------

### Software Requirements

- Python 3.9.1
  - `numpy` 1.20.2
  - `matplotlib` 3.3.4
  - `scipy` 1.6.2

We ran the code both using the Spyder 5.0.5 GUI and from Python's command line.

### Memory and Runtime Requirements

The code does not require large memory. The code was last run on a 1.7 GHz Quad-Core Intel Core i7 Intel-based laptop with MacOS version 11.4. Execution time was less than 30 minutes.

Description of programs/code
----------------------------

- class_model_types.py contains the main Jurymodel class used to simulate juries.
- juryConstruction.py contains code to generate all simulated juries needed to generate the paper figures and tables. Juries are saved in pickle format under output/
- juryPlotsAndResults.py contains code to generate all figures and tables. Output is saved under exhibits/

### License for Code

- the code is © Moro and Van der Linden. Please contact the authors if interested in using or modifying this code.

Instructions to Replicators
---------------------------
run the python commands in the following sequence

1) python juryConstruction.py
2) python juryPlotsAndResults.py

List of figures, tables and programs
---------------------------

| Figure/Table #    | Program                  | Line Number | Output file
|-------------------|--------------------------|-------------|---------------------------------------|
| Figure 3          | juryPlotsAndResults.py   | 67          |exhibits/beta_1-5_dist.pdf
| Figure 3          | juryPlotsAndResults.py   | 87          |exhibits/beta_2-4_dist.pdf
| Figure 3          | juryPlotsAndResults.py   | 111         |exhibits/beta_3-4_dist.pdf
| Figure 4          | juryPlotsAndResults.py   | 338         |exhibits/prop1-beta-1-5-75pcT1-ul.pdf
| Figure 4          | juryPlotsAndResults.py   | 370         |exhibits/prop1-beta-2-4-75pcT1-ul.pdf
| Figure 4          | juryPlotsAndResults.py   | 419         |exhibits/prop1-beta-3-4-75pcT1-ul.pdf
| Figure 5          | juryPlotsAndResults.py   | 419         |exhibits/prop2-uni.pdf
| Figure 6          | juryPlotsAndResults.py   | 631         |exhibits/counter.pdf
| Figure 6          | juryPlotsAndResults.py   | 645         |exhibits/counter_b.pdf
| Figure 7          | juryPlotsAndResults.py   | 731         |exhibits/nchallenges-extreme.pdf
| Figure 7          | juryPlotsAndResults.py   | 1124        |exhibits/nchallenges-minority.pdf
| Figure  8         | juryPlotsAndResults.py   | 1206        |exhibits/median.pdf
| Figure  B         | juryPlotsAndResults.py   | 895         |exhibits/uni-12-6-6.pdf
| Figure  B         | juryPlotsAndResults.py   | 1124        |exhibits/median-2-4.pdf
| Figure  B         | juryPlotsAndResults.py   | 1206        |exhibits/median-3-4.pdf
| Table 1           | juryPlotsAndResults.py   | 567         |exhibits/tables.tex
| Table 2           | juryPlotsAndResults.py   | 847         |exhibits/tables.tex
| Table 3           | juryPlotsAndResults.py   | 1040        |exhibits/tables.tex

## Acknowledgements
Social Science Data Editors template README file https://github.com/social-science-data-editors/template_README
