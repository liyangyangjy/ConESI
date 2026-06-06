# ConESI
A model for enzyme-substrate pair prediction, applicable to genome-scale enzyme mining

# 1. Clone repository
```bash
git clone https://github.com/liyangyangjy/ConESI.git
cd ConESI/Codes
```

# 2. Install dependencies 
```bash
pip install -r requirements.txt
```

# 3. Predict
```bash
python predict_with_trained_model.py \
  --enzy_pt ../Data/df_enzy.pt \
  --smiles_pt ../Data/df_smiles.pt \
  --model_path best_model.pt \
  --output_file predictions.csv
```
#Run the above code and save the results to "predictions.csv".
