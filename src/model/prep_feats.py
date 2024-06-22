import pandas as pd

# takes input dict and converts to a df with columns ordered as
# for the model to predict.
def prep_feats(input_feats: dict) -> pd.DataFrame:
    ordered_feats = [{
        'OverallQual': input_feats['OverallQual'],
        'GrLivArea': input_feats['GrLivArea'],
        'TotalBsmtSF': input_feats['TotalBsmtSF'],
        'BsmtFinSF1': input_feats['BsmtFinSF1'],
        'GarageArea': input_feats['GarageArea'],
        'YearBuilt': input_feats['YearBuilt']
    }]
    return pd.DataFrame(ordered_feats)
