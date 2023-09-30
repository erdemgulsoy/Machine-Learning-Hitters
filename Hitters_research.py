import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from config import regressor


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasetes/hitters.csv")


def grab_col_names(dataframe, cat_th=5, car_th=20) :
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int, float
        numerik fakat kategorik olan değişkenlerdir.
    car_th : int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri.

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi.
        num_cols : list
            Numerik değişkeen listesi.
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi.

    Notes
    ------
        cat_cols + num_cols +cat_but_car = toplam değişken sayısı.
        num_but_cat, cat_cols'un içerisinde.

    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].dtypes in ["int64", "float64", "uint8"] and df[col].nunique() < cat_th]

    cat_but_car = [col for col in df.columns if str(df[col].dtypes) in ["category", "object"] and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if col not in cat_cols and df[col].dtypes in ["int64", "float64", "uint8"]]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def check_df (dataframe, head=5) :
    print("#################### Shape ##################")
    print(dataframe.shape)
    print("#################### Type ##################")
    print(dataframe.dtypes)
    print("#################### Head ##################")
    print(dataframe.head(head))
    print("#################### Tail ##################")
    print(dataframe.tail(head))
    print("#################### NA ##################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9) :

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1 # InterQuartileRange

    up = quartile3 + iqr * 1.5  # formüle göre üst sınır.
    down = quartile1 - iqr * 1.5  # formüle göre alt sınır.

    return up, down
def check_outlier(df, col) :
    up, down = outlier_thresholds(df, col)
    print(col, " : ", df[(df[col] < down) | (df[col] > up)].any(axis=None))
def replace_with_thresholds(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0 or (dataframe[col] == " ").sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def rare_analyser(dataframe, target, cat_cols) :
    for col in cat_cols :
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                            "Ratio": 100 * (dataframe[col].value_counts() / len(dataframe)),
                            "Target_Mean": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def one_hot_encoder(dataframe, categorial_cols, drop_first=True) :
    dataframe = pd.get_dummies(dataframe, columns=categorial_cols, drop_first=drop_first)
    return dataframe
def hitter_data_prep(df) :
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    for col in num_cols:
        replace_with_thresholds(df, col)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    df[num_cols] = scaler.inverse_transform(df[num_cols])

    df["NEW_CHits_per_CAtBat"] =  df["CHits"] / df["CAtBat"] # Oyuncunun kariyer boyu isabetli vuruş oranı.
    df["NEW_CHmRun_per_CRuns"] = df["CHmRun"] / df["CRuns"] # Oyuncunun kariyeri boyunca her kazandırdığı sayı için ortalama olarak kaç en değerli vuruş yaptığını oranı.
    df["NEW_CRuns_per_CRBI"] = df["CRuns"] / df["CRBI"] # Oyuncunun kariyeri boyunca her bir koşu yaptırdırdığı oyuncu başına kaç sayı kazandırdığı oranı.
    df["NEW_CAtBat_per_CRBI"] = df["CAtBat"] / df["CRBI"] # oyuncunun kariyeri boyunca her bir koşu yaptırdırdığı oyuncu başına kaç kez topa vurduğunu gösterir. Bu, oyuncunun vuruş yeteneklerinin takımın skor üretme yeteneği üzerindeki etkisini değerlendirmek için kullanılabilir.
    df["NEW_Assists_per_RBI"] = df["Assists"] / df["RBI"] # oyuncunun 86/87 sezonunda her bir koşu yaptırdırdığı oyuncu başına asist miktarı.
    df["NEW_Errors_per_Years"] = df["Errors"] / df["Years"] # oyuncunun yıllık ortalama hata yapma sayısı.

    df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
    df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
    df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
    df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"

    df.loc[(df["Years"] <= 2), "NEW_Years_Level"] = "Junior"
    df.loc[(df["Years"] > 2) & (df['Years'] <= 5), "NEW_Years_Level"] = "Mid"
    df.loc[(df["Years"] > 5) & (df['Years'] <= 10), "NEW_Years_Level"] = "Senior"
    df.loc[(df["Years"] > 10), "NEW_Years_Level"] = "Expert"

    df["NEW_C_RUNS_RATIO"] = df["Runs"] / df["CRuns"]
    df["NEW_C_ATBAT_RATIO"] = df["AtBat"] / df["CAtBat"]
    df["NEW_C_HITS_RATIO"] = df["Hits"] / df["CHits"]
    df["NEW_C_HMRUN_RATIO"] = df["HmRun"] / df["CHmRun"]
    df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"] +1
    df["NEW_C_WALKS_RATIO"] = df["Walks"] / df["CWalks"]
    df["NEW_C_HIT_RATE"] = df["CHits"] / df["CAtBat"]
    df["NEW_C_RUNNER"] = df["CRBI"] / df["CHits"]
    df["NEW_C_HIT-AND-RUN"] = df["CRuns"] / df["CHits"]
    df["NEW_C_HMHITS_RATIO"] = df["CHmRun"] / df["CHits"]
    df["NEW_C_HMATBAT_RATIO"] = df["CAtBat"] / df["CHmRun"]

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    binary_cols = [col for col in cat_cols if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    ohe_cols = [col for col in df.columns if df[col].nunique() > 2 and df[col].nunique() <= 20]
    df = one_hot_encoder(df, ohe_cols, True)

    all_possible_values = ["StandN", "StandA", "Descend", "Ascend"]
    for val in all_possible_values:
        if f'NEW_PLAYER_PROGRESS{val}' not in df.columns:
            df[f'NEW_PLAYER_PROGRESS{val}'] = 0

    all_possible_values = ["Junior", "Mid", "Senior", "Expert"]
    for val in all_possible_values:
        if f'NEW_Years_Level{val}' not in df.columns:
            df[f'NEW_Years_Level{val}'] = 0

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    scaler = StandardScaler()
    for col in num_cols:
        replace_with_thresholds(df, col)

    df[num_cols] = imputer.fit_transform(df[num_cols])
    df[num_cols] = scaler.fit_transform(df[num_cols])

    y = df["Salary"]
    X = df.drop("Salary", axis=1)

    return X, y

def base_models(X, y, scoring="neg_mean_squared_error"):
    print("Base Models....")
    regressor = [('LR', LinearRegression()),
                   ("RF", RandomForestRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor(eval_metric='logloss')),
                   ('LightGBM', LGBMRegressor(force_col_wise=True, verbose=-1)),
                   ]

    for name, regressors in regressor:
        cv_RMSE_results = np.mean(np.sqrt(-cross_val_score(regressors, X, y, cv=10, scoring=scoring)))
        print(f"{scoring}: {round(cv_RMSE_results, 4)} ({name}) ")

def hyperparameter_optimization(X, y, cv=3, scoring =["neg_mean_squared_error","mean_absolute_error", "r_squared"]):
    print("Hyperparameter Optimization....")
    best_models = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

    for name, regressors, params in regressor:
        print(f"########## {name} ##########")
        y_pred = regressors.fit(X, y).predict(X_test)

        cv_RMSE_results = np.mean(np.sqrt(-cross_val_score(regressors, X, y, cv=10, scoring=scoring[0])))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = regressors.score(X, y)

        print(f"{scoring[0]} (Before): {round(cv_RMSE_results, 4)}")
        print(f"{scoring[1]} (Before): {round(mae, 4)}")
        print(f"{scoring[2]} (Before): {round(r2, 4)}")

        gs_best = GridSearchCV(regressors, params, cv=cv, n_jobs=-1, verbose=-1).fit(X, y)
        final_model = regressors.set_params(**gs_best.best_params_).fit(X, y)

        print("--------------------------------")

        cv_RMSE_results = np.mean(np.sqrt(-cross_val_score(regressors, X, y, cv=10, scoring=scoring[0])))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = regressors.score(X, y)

        print(f"{scoring[0]} (After): {round(cv_RMSE_results, 4)}")
        print(f"{scoring[1]} (After): {round(mae, 4)}")
        print(f"{scoring[2]} (After): {round(r2, 4)}")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        if r2 <= 0.95:
            best_models[name] = final_model

    return best_models

def val_curve_params(model, X, y, param_name, param_range, scoring="neg_mean_squared_error", cv=10):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, label="Training score", color="darkorange")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
    plt.title("Validation Curve with RandomForestRegressor")
    plt.xlabel(param_name)
    plt.ylabel("Negative Mean Squared Error")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)



missing_values_table(df) # sorunumuz var...

X, y = hitter_data_prep(df)
na_cols = missing_values_table(df, True)

hyperparameter_optimization(X, y)
cat_cols, num_cols, v = grab_col_names(X)
























