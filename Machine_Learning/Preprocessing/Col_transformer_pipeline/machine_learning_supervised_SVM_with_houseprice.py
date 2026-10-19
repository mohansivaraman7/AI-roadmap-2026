import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline

df = pd.read_csv("house_price_bd.csv")
print(df)

df["Price_in_taka"] = df["Price_in_taka"].replace({r"[^\d.]": ""}, regex=True).astype(float)

X = df.drop(columns=["Price_in_taka", "Title", "Location"])
y = df["Price_in_taka"]

for col in X.select_dtypes(include=["float64", "int64"]).columns:
    X[col] = X[col].fillna(X[col].mean())

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

svr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svr_pipeline.fit(X_train, y_train)


train_score = svr_pipeline.score(X_train, y_train)
test_score = svr_pipeline.score(X_test, y_test)

print(train_score)
print(test_score)