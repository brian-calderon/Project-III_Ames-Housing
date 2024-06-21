##
import pickle
import pandas as pd
import os

from src.model import feat_ids
from src.model import misc_ids

##
file_name = misc_ids.USER_INPUT_FILE
user_input = {}
user_input[feat_ids.OVERALL_QUALITY] = 10
## Read the user input file

# Open a file in binary read mode
with open(file_name+'.pkl', "rb") as file:
    # Deserialize the data from the file
    loaded_data = pickle.load(file)

##

with open(file_name + '.pkl', "wb") as file:
    pickle.dump(user_input, file)
#### ----------------------Loading the 6 Feat Model---------------------------------
# i.e. the model trained on all the features
# with open('../GB_Test_6feats.pkl', 'rb') as f:
#     gb_test_6feats = pickle.load(f)
# print(gb_test_6feats.feature_names_in_)
## check if file exists
import pickle
import pandas as pd
import os

from src.model import feat_ids
from src.model import misc_ids

user_input = misc_ids.USER_INPUT_FILE
model_name = misc_ids.MODEL_NAME
model_path = 'data/'+model_name+'.pkl'

if os.path.exists(misc_ids.USER_INPUT_FILE + '.pkl'):
    print(f'File {misc_ids.USER_INPUT_FILE}.pkl exists')
else:
    print(f'File {misc_ids.USER_INPUT_FILE}.pkl doesn\'t exist')


## Predict 6 feat model
import pickle
import pandas as pd
import os

from src.model import feat_ids
from src.model import misc_ids

user_input = misc_ids.USER_INPUT_FILE
model_name = misc_ids.MODEL_NAME
model_path = 'data/'+model_name+'.pkl'

with open(user_input + '.pkl', "rb") as file:
    loaded_data = pickle.load(file)

with open(model_path, 'rb') as f:
    model = pickle.load(f)


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

# NOTE: The top 6 features are:
# ['OverallQual' 'GrLivArea' 'TotalBsmtSF' 'BsmtFinSF1' 'GarageArea'
#      'YearBuilt']
temp_data = prep_feats(loaded_data)

predicted = model.predict(temp_data)
f'Predicted values is: ${predicted:.2f}'
