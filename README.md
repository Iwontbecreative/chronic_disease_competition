# Data competition: Predicting the importance of chronic diseases for health institutions

### What was this Data Competition about?

* Organizers: ANAP and ATIH (public health French institutions)
* Goal: predict the **mid-term impact of chronic diseases** for health establishments
* Kind of task: regression
* Metric: [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
* Platform: [Datascience.net](https://www.datascience.net)
    * **600 competitors**
    * **3 months**, ended on 14/10/2016
    * This repository details **my solution 9th on both private and public leaderboards**

More details can be found at the [competition's page](https://www.datascience.net/fr/challenge/28/details)

### The Data

* Train and Test data with logs of patient visits (age, year, health establishments, number of visits to other health institutions...)
* Information on each health establishment ([hospidiag](http://hospidiag.atih.sante.fr/cgi-bin/broker?_service=hospidiag&_debug=0&_program=hd.accueil_hd.sas) reports)
* All open data you could find and leverage

The data was provided as is, which meant there was still some data engineering needed to work with it

### Approach

The detailed approach can be found in French in this [ipython
notebook](https://github.com/Iwontbecreative/chronic_disease_competition/blob/master/chronic/Rapport%20Python.ipynb)
