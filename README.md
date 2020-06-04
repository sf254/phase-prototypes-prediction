''Predicting phase prototypes of inorganic substances through transfer learning''

 CNNs were trained on a big dataset of 228,676 compounds from open quantum materials database (OQMD). The feature extractors of the well-trained CNNs were reused for feature extraction on a phase prototypes dataset containing 17,762 inorganic substances and involving 170 phase prototypes. Random forest was then utilized as classifier. 

Pickle formatted file 'comp_energy_pa_oqmdf2b.txt' and 'comp_volume_pa_oqmdf2.txt' contain Ef and V data from OQMD. Pickle formatted file 'phase-prototypes-dataset.txt' contains the phase-prototypes-dataset. Pickle formatted file 'element_property.txt' and 'Z_row_column.txt' contain information of 108 chemcial elements' atomic numbers, row numbers, and column number etc.

Codes 'CNN-OQMD-Ef' and 'CNN-OQMD-Ef' carry out training and testing. The best models are saved for next step prediction.

10 models, i.e. 'CNN-Ef-best-(0-4)Wb.h5' and 'CNN-V-best-(0-4)Wb.h5', were obtained after 5-fold cross-validation. 

'main-transfer-learning.py' read phase-prototypes-dataset, uses one model from 'CNN-Ef-best-(0-4)Wb.h5 or CNN-V-best-(0-4)Wb.h5 as features generators, and use random forest as classifier to fulfil the goal of transfer learning.

'main-non-transfer.py' read phase-prototypes-dataset, took composition vector as input, and use random forest as classifier.

