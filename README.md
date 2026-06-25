# COF-ML
A repository for training machine learnning classifiers for distinguishing electroacitive behavior of Covalent Organic Frameworks and downstream prediction on the CORE COF data base

# Brief description
dice\_wOCV\_wcap.xlsx: DFT-derived OCV and capacity-labelled dataset for a diverse collection of electroactive COFs (DICE) curated from the CORE COF database
feats.ipynb: Notebook to featurize a COF from the unrelaxed CIF structure
xgboost\_ocv\_classifier.py: Python script for training the models for distinguishing anodic vs. cathodic behavior of COFs and downstream prediction tasks. 
xgboost\_cap\_classifier.py: Python script for training the models for distinguishing COFs with high and low capacity and downstream prediction tasks. 
