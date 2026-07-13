import pandas as pd
from sklearn.utils import shuffle



def grouped_by_enzy(filename,output_file):
    #1. Load data and group

    # Read CSV
    df = pd.read_csv(filename)  

    # Group by Uniprot ID
    grouped = df.groupby("Uniprot ID")


    #2. Build triplets
    triplets = []

    for enzyme_id, group in grouped:
        # Separate positive samples (label == 1) and negative samples (label == 0)
        pos_samples = group[group["output"] == 1]
        neg_samples = group[group["output"] == 0]

        # Only construct triplets if both positive and negative samples exist
        if len(pos_samples) > 0 and len(neg_samples) > 0:
            for _, pos_row in pos_samples.iterrows():
                for _, neg_row in neg_samples.iterrows():
                    triplets.append({
                        "anchor_sequence": pos_row["Protein sequence"],      # enzyme 共享
                        "positive_smiles": pos_row["SMILES"],
                        "negative_smiles": neg_row["SMILES"]
                    })


    #3. Convert to DataFrame and save.
    triplet_df = pd.DataFrame(triplets)
    triplet_df = shuffle(triplet_df).reset_index(drop=True)

    triplet_df.to_csv(output_file, index=False)
    
    
file_list=['../Data/ESP_train_df-experimental-evidence-based_dataset.csv']
for filename in file_list:
    fn_list=filename.split('_')
    output_file=fn_list[0]+'_for_TripleLoss_by_enzyme'+fn_list[1]+'_'+fn_list[2]+'_'+fn_list[3]
    grouped_by_enzy(filename,output_file)