import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer


def load_data():
    #since the data is too large to use read_csv(it failed to read the data when i gave it the url)
    #here is the url to the dataset: https://drive.proton.me/urls/7X386KENPM#VFw2Gi2EfEOs 
    data = pd.read_csv("data/data.csv")

    data.columns = data.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(data.dtypes[data.dtypes == 'object'].index)

    for c in categorical_columns:
        data[c] = data[c].str.lower().str.replace(' ', '_')

    nan = ['mms', 'ernedc_(g/km)', 'electric_range_(km)', 'vf', 'enedc_(g/km) ',
           'at2_(mm)', 'at1_(mm)', 'w_(mm)', 'rlfi', 'de', 'z_(wh/km)', 'it',
           'erwltp_(g/km)', 'enedc_(g/km)']

    data = data[[col for col in data.columns if col not in nan]]

    features = ['mk', 'm_(kg)', 'mt', 'w_(mm)', 'at1_(mm)', 'at2_(mm)',
                'ft', 'fm', 'ec_(cm3)', 'ep_(kw)', 'fuel_consumption_', 'ech']
    features = [f for f in features if f in data.columns]

    categorical_columns = [col for col in categorical_columns if col in features]

    target = 'ewltp_(g/km)'

    select_data = data[features + [target]].copy()

    numerical_columns = list(select_data.dtypes[select_data.dtypes != 'object'].index)
    numerical_columns.remove(target)

    spelling_mappings = {
        'toyota/carpol' : 'toyota',
        'renault/carpol' : 'renault',
        'toyota/steeler' : 'toyota',
        'opel/carpol' : 'opel',
        'mercedes-amg' : 'mercedes-benz',
        'peugeot/carpol' : 'peugeot',
        'volkswageb' : 'volkswagen',
        'ford/fc_auto_system' : 'ford',
        'citroen/carpol' : 'citroen',
        'allied_vehicles_ltd.' : 'allied_vehicles_ltd',
        'volkswagen/v-van' : 'volkswagen',
        'cms_auto/mercedes-benz' : 'mercedes-benz',
        'ford_transit/frank-cars' : 'ford_transit',
        'ssangyong_kg_mobility' : 'ssangyong',
        'mercedes/v-van' : 'mercedes-benz',
        'ford/germaz' : 'ford',
        'volkswagen_amz_kutno' : 'volkswagen',
        'ford/frank-cars' : 'ford',
        'ford_transit/its_system' : 'ford_transit',
        'volkswagen/carpol' : 'volkswagen',
        'nisssan' : 'nissan',
        'fiat/carpol' : 'fiat',
        'volkswagen/zimny' : 'volkswagen',
        'mercedes-benz/mrc' : 'mercedes-benz',
        'ssang_yong' : 'ssangyong',
        'ford/auto_galeria' : 'ford',
        'alpina' : 'bmw',
        'suzki' : 'suzuki',
        'man/carpol' : 'man',
        'renault_/_multitel' : 'renault',
        'nissa' : 'nissan',
        'opek' : 'opel',
        'volkswage._vw' : 'volkswagen',
        'lexsus' : 'lexus',
        'mercede-benz': 'mercedes-benz',
        'ssangyong_kg_mobitity' : 'ssangyong',
        'mercedes' : 'mercedes-benz',
        'mercedes-benz/cms-auto' : 'mercedes-benz',
        'volkswagen/mobilcar' : 'volkswagen',
        'jaeccoo' : 'jaecoo',
        'ford_transit/auto_galeria' : 'ford_transit',
        'volkswagen/mrc' : 'volkswagen',
        'ssang-young' : 'ssangyong',
        'porche' : 'porsche',
        'omoda5' : 'omoda',
        'mag' : 'mg',
        'caterham' : 'caterham_cars_ltd',
    }

    select_data['mk'] = select_data['mk'].replace(spelling_mappings)
    return select_data, numerical_columns, categorical_columns, target

def train_model(data, numerical_cols, categorical_cols):
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=1)

    y_train = df_train['ewltp_(g/km)']
    y_test = df_test['ewltp_(g/km)']

    del df_train['ewltp_(g/km)']
    del df_test['ewltp_(g/km)']


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_encoder', TargetEncoder(handle_unknown='value'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('target', target_transformer, ['mk', 'ech']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'), ['ft', 'fm']),
            ('num', numeric_transformer, numerical_cols)
        ],
        remainder='drop'
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(random_state=42,
        learning_rate= 0.2,
        max_depth= 7,
        min_child_samples= 5,
        n_estimators= 500
                                ))
    ])


    pipe.fit(df_train, y_train)

    return pipe

def save_model(model, filename):

    joblib.dump(model, filename)
    print(f"Model Pipeline saved to: {filename}")


select_df, num, cat, target = load_data()
pipeline = train_model(select_df, num, cat)
save_model(pipeline, 'model.bin')


