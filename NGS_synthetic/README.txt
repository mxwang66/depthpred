Training and validation
------------------------
To run 20-fold cross-validation on NGS synthetic panel with our default model, use the command (inside "NGS_synthetic" folder):
	bash NGS_synthetic.sh
The training would take about two hours for each fold.  Comment out lines in "src/Yuan.sh" and change "n_fold" in "read_result_Yuan.py" for running fewer folds. 

Results
------------------------
The results would be saved in folder "logs", with "pred.csv" showing observed (first column) and predicted (second column) log10(reads depth) of all the 20 validation sets combined (stop epoch = 1000). "index_tar.csv" shows the sequence ID (first column) and the corresponding observed log10(reads depth) (second column). For the actual sequence of each sequence ID, please refer to "Sequences.xlsx". "pred.csv" and "index_tar.csv" have the same row order. "test_loss.csv" or "train_loss.csv" shows the loss of validation sets or train sets, with each column corresponding to one fold and each row corresponding to one epoch. 

Feature generation
------------------------
For features calculated with NUPACK (open probability, free energy), the NUPACK parameters are: material = DNA, temperature = 55C, sodium = 0.8M. 
