from imports_and_env_setup import *

def load_ames_housing_data():
    data = fetch_openml(name="house_prices", as_frame=True)
    X = data.data.copy()
    y = pd.to_numeric(data.target, errors="coerce").astype(float)

    if y.name in X.columns:
        raise ValueError("❌ Target leakage detected")

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    return X, y, num_cols, cat_cols
