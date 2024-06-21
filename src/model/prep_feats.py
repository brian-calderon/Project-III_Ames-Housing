import pandas as pd

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
