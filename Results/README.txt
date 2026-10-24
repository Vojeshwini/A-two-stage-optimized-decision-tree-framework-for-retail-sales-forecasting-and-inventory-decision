TSO-DTF - Project Results
=========================

01_final_results
  main_accuracy_metrics.csv     Official M5 score, WRMSSE, RMSSE, RMSE, MAE, WAPE and bias: base DTF vs TSO-DTF
  hierarchy_levels_wrmsse.csv   WRMSSE at each M5 hierarchy level and the improvement (%)
  benchmarks_and_inventory.csv  All benchmark methods with accuracy and inventory results
                                (inv = mean inventory, sl = service level, below = products below 95%,
                                 bwo / bwi = bullwhip of orders / inventory)
  diebold_mariano_tests.csv     Diebold-Mariano significance tests on the 62 daily losses
  wilcoxon_effect_size.csv      Share of products improved, Wilcoxon test and effect size
  fairness_check_5seeds.csv     Comparison with the base DTF averaged over five seeds

02_figures                      All figures used in the paper

03_forecasts                    Daily test forecasts (62 days, 9,119 products) of the base DTF and of the
                                five TSO-DTF models (Opt 4, 5, 6, 7, 10); TSO-DTF = their equal-weight average

04_optimization_logs            Tuning log, search histories of each optimization stage and the
                                demand-pattern classification of every product

05_dashboard                    Interactive decision-support dashboard (open dashboard.html in a browser)

Main result: official M5 score 0.5563 (base DTF) -> 0.5392 (TSO-DTF), 3.08% better.
