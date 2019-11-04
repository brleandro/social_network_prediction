""""Represents a controller for Profile Model."""
import pandas as pd
from model import predictor as pr
import os


def generate_files(input_path, output_path):
    """
    Generates xml without any API since it is a very simple one.
    :param input_path:
    :param output_path:
    :return:
    """
    profiles = pd.read_csv(f'{input_path}/Profile/Profile.csv')

    # call all predictors
    predicted = {f: pr.all_predictors[f].predict(profiles=profiles, base_folder=input_path)
                 for f in pr.features if f in pr.all_predictors}

    for i, (index, row) in enumerate(profiles.iterrows()):
        file = open(os.path.join(output_path, f'{row["userid"]}.xml'), 'w')
        # gender = pr.get_gender(predicted['gender'][row[1]]) if row[1] in predicted['gender'] else 'null'
        open_ = predicted['open'][row[1]] if row[1] in predicted['open'] else 'null'
        conscientious = predicted['conscientious'][row[1]] if row[1] in predicted['conscientious'] else 'null'
        extrovert = predicted['extrovert'][row[1]] if row[1] in predicted['extrovert'] else 'null'
        agreeable = predicted['agreeable'][row[1]] if row[1] in predicted['agreeable'] else 'null'
        neurotic = predicted['neurotic'][row[1]] if row[1] in predicted['neurotic'] else 'null'
        age = predicted['age_group'][row[1]] if row[1] in predicted['age_group'] else 'null'

        file.write(f'<user id="{row[1]}" ')
        file.write(f'age_group="{age}" ')
        # file.write(f'gender="{gender}" ')
        file.write(f'gender="null" ')
        file.write(f'extrovert="{extrovert}" ')
        file.write(f'neurotic="{neurotic}" ')
        file.write(f'agreeable="{agreeable}" ')
        file.write(f'open="{open_}" ')
        file.write(f'conscientious="{conscientious}" />')
        file.close()
