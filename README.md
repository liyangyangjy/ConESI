# ConESI
A model for enzyme-substrate pair prediction, applicable to genome-scale enzyme mining

# 1. Clone repository
```bash
git clone https://github.com/liyangyangjy/ConESI.git
cd ConESI/Codes
```

# 2. Install dependencies 
The prediction pipeline is fully compatible with CPU-only environments and does not require CUDA support. 
GPU acceleration is recommended for model training and large-scale inference to improve computational efficiency.
```bash
conda env create -f ConESI.yml
```

# 3. Predict (CPU or GPU)
```bash

#3.1 Generate embeddings

python generate_embeddings_for_pre.py --filename ../Data/predict_data.csv

#3.2 Prediction using the trained model (best_model.pt)

python predict_with_trained_model.py \
  --enzy_pt ../Data/predict_data_enzy.pt \
  --smiles_pt ../Data/predict_data_smiles.pt \
  --model_path best_model.pt \
  --output_file predictions.csv

#Run the above code and save the results to "predictions.csv".
```
# 4. Training from Scratch (CPU or GPU)
To train ConESI from scratch, run:
```bash
#4.1 Prepare data 

# Translate the enzyme-substrate pair training set into a triplet dataset for triplet loss function training.
# Dataset used for model training and fine-tuning(Data.zip) are available in the Zenodo repository, https://doi.org/10.5281/zenodo.21324639

python grouped_for_TripletLoss.py # Adjust the input files by modifying the 'file_list' variable in the script.

# Generate the CSV files named ESI_for_TripleLoss_by_enzymetrain_df-experimental-evidence-based_dataset.csv and ESP_for_TripleLoss_by_enzymetrain_df-phylogenetic-evidence-based_dataset.csv

#4.2 Genarate embeddings

# Calculate the embeddings for enzymes and small molecules.

python generate_embeddings_for_TripletLoss.py --filename ../Data/ESI_for_TripleLoss_by_enzymetrain_df-experimental-evidence-based_dataset.csv
python generate_embeddings_for_TripletLoss.py --filename ../Data/ESI_for_TripleLoss_by_enzymetrain_df-phylogenetic-evidence-based_dataset.csv

python generate_embeddings_for_train.py --filename ../Data/ESI_test_df-experimental-evidence-based_dataset.csv
python generate_embeddings_for_train.py --filename ../Data/ESI_val_df-phylogenetic-evidence-based_dataset.csv
python generate_embeddings_for_train.py --filename ../Data/ESI_test_df-experimental-evidence-based_dataset.csv
python generate_embeddings_for_train.py --filename ../Data/ESI_val_df-phylogenetic-evidence-based_dataset.csv

#4.3 Train

# Training of the model can be performed with the following code, using the data from ESI_test_df-phylogenetic-evidence-based_dataset.csv.

python train.py \
  --train_anchor ../Data/ESI_for_TripleLoss_by_enzymetrain_df-phylogenetic-evidence-based_dataset_anchor_enzy.pt \
  --train_positive ../Data/ESI_for_TripleLoss_by_enzymetrain_df-phylogenetic-evidence-based_dataset_positive_smiles.pt \
  --train_negative ../Data/ESI_for_TripleLoss_by_enzymetrain_df-phylogenetic-evidence-based_dataset_negative_smiles.pt \
  --val_enzy ../Data/ESI_val_df-phylogenetic-evidence-based_dataset_enzy.pt \
  --val_smiles ../Data/ESI_val_df-phylogenetic-evidence-based_dataset_smiles.pt \
  --val_y ../Data/ESI_val_df-phylogenetic-evidence-based_dataset_label.pt \
  --test_enzy ../Data/ESI_test_df-phylogenetic-evidence-based_dataset_enzy.pt \
  --test_smiles ../Data/ESI_test_df-phylogenetic-evidence-based_dataset_smiles.pt \
  --test_y ../Data/ESI_test_df-phylogenetic-evidence-based_dataset_label.pt

#4.4 Finetune(optional)

# Fine-tuning of the trained model can be performed with the following code, using the data from ESI_train_df-experimental-evidence-based_dataset.csv.

python finetune_from_best_save.py \
  --train_anchor ../Data/ESI_for_TripleLoss_by_enzymetrain_df-experimental-evidence-based_dataset_anchor_enzy.pt \
  --train_positive ../Data/ESI_for_TripleLoss_by_enzymetrain_df-experimental-evidence-based_dataset_positive_smiles.pt \
  --train_negative ../Data/ESI_for_TripleLoss_by_enzymetrain_df-experimental-evidence-based_dataset_negative_smiles.pt \
  --val_enzy ../Data/ESI_val_df-experimental-evidence-based_dataset_enzy.pt \
  --val_smiles ../Data/ESI_val_df-experimental-evidence-based_dataset_smiles.pt \
  --val_y ../Data/ESI_val_df-experimental-evidence-based_dataset_label.pt \
  --test_enzy ../Data/ESI_test_df-experimental-evidence-based_dataset_enzy.pt \
  --test_smiles ../Data/ESI_test_df-experimental-evidence-based_dataset_smiles.pt \
  --test_y ../Data/ESI_test_df-experimental-evidence-based_dataset_label.pt
  --ckpt best_model.pt
```
