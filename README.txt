REQUIREMENTS
============
Platform
  - Google Colab (notebooks 01-08 and 10-13), with Google Drive access
  - Python 3.9 or later (09_bench.py, run on a local computer)

Python libraries
  numpy, pandas, scipy, lightgbm, optuna, cmaes, shap, pyarrow, matplotlib
  (in Colab, each notebook installs what it needs in its first cell;
   locally: pip install numpy pandas scipy lightgbm optuna cmaes shap pyarrow matplotlib)

Google Drive folders
  My Drive/m5_data      the M5 dataset files (see DATASET)
  My Drive/m5_results   created automatically; every notebook saves its outputs here


DATASET
=======
Name     M5 Forecasting - Accuracy (Walmart daily sales data)
Source   Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Files used
  calendar.csv                 dates, weekdays, events and SNAP days
  sell_prices.csv              weekly selling price of each product in each store
  sales_train_evaluation.csv   daily unit sales of each product in each store
                               (d_1 to d_1941, 29 Jan 2011 to 22 May 2016)
  weights_evaluation.csv       sales-value weights used in the official M5 evaluation

Data used in this project
  Stores           CA_1, TX_1 and WI_1 (one store from each state)
  Products         9,119 product series (CA_1: 3,040, TX_1: 3,043, WI_1: 3,036)
                   in the categories Foods, Household and Hobbies
  Demand patterns  smooth 897, erratic 207, intermittent 6,935, lumpy 1,080
  Validation       last 56 days before the test period (used for all tuning decisions)
  Test period      62 days, 22 March 2016 to 22 May 2016 (next-day forecasts)


FOLDER CONTENTS
===============
01-13 code files   notebooks and script, numbered in the order they are run
data_files/
  eval_products.csv         the 9,119 evaluated products, their demand pattern and weight
  paper_figures_data.zip    data used to draw the paper figures


CODE FILES
==========
No.  File                                   Purpose                                         Main outputs (in m5_results)
---  -------------------------------------  ----------------------------------------------  -----------------------------------------
01   M5_Colab_Resume.ipynb                  Data preparation, demand classification,        baseline_dtf_results.csv, tuning_log.csv,
                                            base DTF, Opt 1 (CMA-ES tuning),                demand_pattern_classification.csv,
                                            Opt 2 (pattern pooling), Opt 3 (joint search)   optimization1/2/3 results
02   M5_Baseline_Predictions_Colab.ipynb    Base DTF daily test forecasts                   baseline_predictions.csv.gz
03   M5_Optimization4_Colab.ipynb           Stage 1: feature-group selection (TPE)          optimization4_predictions.csv.gz
04   M5_Optimization5_Colab.ipynb           Stage 2: joint hyperparameter and pooling       optimization5_predictions.csv.gz
                                            search (CMA-ES + median pruning)
05   M5_Optimization6_Colab.ipynb           Stage 3: 2-year history + sales-value weights   optimization6_predictions.csv.gz
06   M5_Optimization7_Colab.ipynb           Stage 3 on demand-pattern pooling               optimization7_predictions.csv.gz
07   M5_Optimization10_Colab.ipynb          Stage 3: hierarchical inputs                    optimization10_predictions.csv.gz
08   M5_Final_Results_Graphs_Colab.ipynb    Final TSO-DTF ensemble (Opt 4+5+6+7+10),        result tables and graphs
                                            all accuracy results and graphs
09   bench.py                               Statistical benchmarks: Naive, Seasonal naive,  benchmark and inventory results
                                            Moving average, Exponential smoothing,
                                            Croston, ADIDA
10   M5_ExtraChecks_C1_C3_v2_Colab.ipynb    Fairness check (5-seed base DTF) and            extra_checks_*.csv
                                            Diebold-Mariano test
11   M5_Paper_Figures_Colab.ipynb           Figures of the paper                            paper_figures.zip
12   M5_BasePaper_Figures_Colab.ipynb       Comparison figures with the base paper          figure files
13   M5_Dashboard_Colab.ipynb               Interactive decision-support dashboard          dashboard.html


HOW TO RUN
==========
1. Download the M5 files from Kaggle and copy them to My Drive/m5_data.
2. Copy data_files/eval_products.csv to My Drive/m5_results
   (create the folder if it does not exist).
3. Open notebook 01 in Google Colab and click Runtime -> Run all.
   Allow Google Drive access when asked and keep the tab open until it finishes.
   Notebook 01 can be re-run after a disconnection; finished steps are loaded from Drive.
4. Run notebooks 02 to 08 in order in the same way. Each notebook reads the outputs of the
   earlier ones from My Drive/m5_results.
5. Run 10, 12 and 13 in the same way. For 11, put data_files/paper_figures_data.zip in
   My Drive/m5_results before running.
6. 09_bench.py runs on a local computer: place it in a folder that contains m5full/ (the M5 files),
   figs/figdata/per_product.csv and the res/ forecast files, then run: python 09_bench.py

Approximate run times on free Colab: notebook 01 several hours (resumable);
03 and 06 about 45-75 minutes each; 04 about 1.5 hours; 10 about 1 hour;
08, 11, 12 and 13 about 5-10 minutes each (no model training).


MAIN RESULT
===========
Official M5 score (WRMSSE, lower is better)
  Base DTF   0.5563
  TSO-DTF    0.5392   (3.08% better)
