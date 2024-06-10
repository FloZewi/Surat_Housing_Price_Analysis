import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Hilfsfunktion zum Umwandeln von Preisangaben in numerische Werte
def convert_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if 'Cr' in price_str:
        return float(price_str.replace('₹', '').replace(' Cr', '').replace(',', '').strip()) * 1e7
    elif 'Lac' in price_str:
        return float(price_str.replace('₹', '').replace(' Lac', '').replace(',', '').strip()) * 1e5
    else:
        return float(price_str.replace('₹', '').replace(',', '').strip())


# Hilfsfunktion zum Umwandeln von Flächenangaben in Quadratfuß
def convert_area(area_str):
    if pd.isna(area_str):
        return np.nan
    if 'sqm' in area_str:
        return float(area_str.replace(' sqm', '').replace(',', '').strip()) * 10.7639  # Convert sqm to sqft
    elif 'sqft' in area_str:
        return float(area_str.replace(' sqft', '').replace(',', '').strip())
    else:
        return np.nan


# Daten laden und bereinigen
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    # Behandle 'Call for Price' und andere nicht-numerische Werte in der Preisspalte
    data['price'] = data['price'].replace('Call for Price', np.nan)
    data['price'] = data['price'].apply(convert_price)

    # Anwendung der helper function, um die Flächenangaben zu bereinigen und zu konvertieren
    data['square_feet'] = data['square_feet'].apply(convert_area)

    data['bhk'] = data['property_name'].str.extract(r'(\d+) BHK')
    data['bhk'] = data['bhk'].fillna(0).astype(int)

    # Extrahieren der Etageninformationen in separate Spalten
    data[['floor', 'total_floors']] = data['floor'].str.extract(r'(\d+) out of (\d+)').astype(float)
    data = data.drop(columns=['property_name', 'areaWithType', 'description'])

    return data


# Explorative Datenanalyse
def exploratory_data_analysis(data):
    data[['price', 'square_feet', 'bhk']].hist(bins=30, figsize=(15, 10))
    plt.show()
    sns.pairplot(data[['price', 'square_feet', 'bhk']])
    plt.show()
    correlation_matrix = data[['price', 'square_feet', 'bhk']].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()


# Entfernen von Ausreißern
def remove_outliers(data):
    data_numeric = data.select_dtypes(include=[np.number])  # Nur numerische Spalten auswählen
    Q1 = data_numeric.quantile(0.25)
    Q3 = data_numeric.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data_numeric < (Q1 - 1.5 * IQR)) | (data_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]


# Feature Engineering
def feature_engineering(data):
    data['price_per_sqft'] = data['price'] / data['square_feet']
    data = pd.get_dummies(data, columns=['transaction', 'status', 'furnishing', 'facing'], drop_first=True)

    # Imputation der NaN-Werte mit dem Median
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    return data_imputed


# Modellbildung und Evaluierung
def build_and_evaluate_model(data):
    X = data.drop(columns=['price'])
    y = data['price']

    # Pipeline zur Datenvorverarbeitung und Modellbildung
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor())
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1],
        'model__max_depth': [3, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')

    y_pred = grid_search.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

    plt.scatter(y_test, y_pred)
    plt.xlabel('Wahre Werte')
    plt.ylabel('Vorhersagen')
    plt.title('Wahre Werte vs Vorhersagen')
    plt.show()


if __name__ == "__main__":
    file_path = 'surat_uncleaned.csv'
    data = load_and_clean_data(file_path)
    data = remove_outliers(data)
    exploratory_data_analysis(data)
    data = feature_engineering(data)
    build_and_evaluate_model(data)
