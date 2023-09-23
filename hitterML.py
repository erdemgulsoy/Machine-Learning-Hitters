# author : Mustafa Erdem Gülsoy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasetes/hitters.csv")


#######################################################
# Genel Resim ;
#######################################################
df.head()

# DataFrame için ilk olarak kategorik ve numerik değişkenleri ayırıyoruz ;
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
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# DataFrame için ön bilgi alma işlemi ;
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
check_df(df)

def cat_summary2(dataframe, col_name, plot=False) :
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio"  : 100 * dataframe[col_name].value_counts() / len(dataframe) }))
    print("###############################################")

    if plot :
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
for col in cat_cols :
    cat_summary2(df, col, True)

def num_summary2(dataframe, numerical_col, plot=False) :
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols :
    num_summary2(df, col, True)

# Kategorik değişkenlere göre hedef değişkenin ortalaması ;
for col in cat_cols :
    print(df.groupby(col).agg({"Salary": "mean"}))

# Numerik değişkenlere göre hedef değişkenin ortalaması ;
for col in num_cols :
    print(df.groupby(col).agg({"Salary": "mean"}))


#################################################
# Outliers Analizi ;
#################################################


# Aykırı değer kontrolü için up ve down oranlarını belirliyoruz.
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9) :

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    iqr = quartile3 - quartile1 # InterQuartileRange

    up = quartile3 + iqr * 1.5  # formüle göre üst sınır.
    down = quartile1 - iqr * 1.5  # formüle göre alt sınır.

    return up, down
# Değişkenlerin aykırı değerleri varsa True, yoksa False veriyor. (check outlier)
def check_outlier(df, col) :
    up, down = outlier_thresholds(df, col)
    print(col, " : ", df[(df[col] < down) | (df[col] > up)].any(axis=None))

for col in num_cols:
    check_outlier(df, col)

# Aykırı Değerleri up ve down'a eşitledik.
def replace_with_thresholds(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

df.head()
# Aykırı değerleri thresholds'ladık.


#################################################
# Missing Value Analizi ;
#################################################


# Eksik değer var mı ?
def missing_values_table(dataframe, na_name = False) :
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0 or (dataframe[col] == " ").sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name :
        return na_columns

na_cols = missing_values_table(df, True)

# Standart Scaler uygulayalım ;
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# KNN ile Eksik değerleri doldurmayı deneyeceğiz;
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

df.head()

# Eski Haline Getirelim ;
df[num_cols] = scaler.inverse_transform(df[num_cols])

df.head()
# Tüm eksik Değerleri en yakın 5 komşusuna göre doldurmuş bulunuyoruz.

# Tekrar eksik değer var mı diye bakıyoruz ;
missing_values_table(df) # Yok.



#################################################
# KORELASYON ANALİZİ
#################################################


# Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir

df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)



#################################################
# DEĞİŞKEN ÜRETME
#################################################

df["NEW_CHits_per_CAtBat"] =  df["CHits"] / df["CAtBat"] # Oyuncunun kariyer boyu isabetli vuruş oranı.
df["NEW_CHmRun_per_CRuns"] = df["CHmRun"] / df["CRuns"] # Oyuncunun kariyeri boyunca her kazandırdığı sayı için ortalama olarak kaç en değerli vuruş yaptığını oranı.
df["NEW_CRuns_per_CRBI"] = df["CRuns"] / df["CRBI"] # Oyuncunun kariyeri boyunca her bir koşu yaptırdırdığı oyuncu başına kaç sayı kazandırdığı oranı.
df["NEW_CAtBat_per_CRBI"] = df["CAtBat"] / df["CRBI"] # oyuncunun kariyeri boyunca her bir koşu yaptırdırdığı oyuncu başına kaç kez topa vurduğunu gösterir. Bu, oyuncunun vuruş yeteneklerinin takımın skor üretme yeteneği üzerindeki etkisini değerlendirmek için kullanılabilir.
df["NEW_Assists_per_RBI"] = df["Assists"] / df["RBI"] # oyuncunun 86/87 sezonunda her bir koşu yaptırdırdığı oyuncu başına asist miktarı.
df["NEW_Errors_per_Years"] = df["Errors"] / df["Years"] # oyuncunun yıllık ortalama hata yapma sayısı.

# Player Promotion to Next League
df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "StandN"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "StandA"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), "NEW_PLAYER_PROGRESS"] = "Descend"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), "NEW_PLAYER_PROGRESS"] = "Ascend"

# PLAYER LEVEL
df.loc[(df["Years"] <= 2), "NEW_Years_Level"] = "Junior"
df.loc[(df["Years"] > 2) & (df['Years'] <= 5), "NEW_Years_Level"] = "Mid"
df.loc[(df["Years"] > 5) & (df['Years'] <= 10), "NEW_Years_Level"] = "Senior"
df.loc[(df["Years"] > 10), "NEW_Years_Level"] = "Expert"

# CAREER RUNS RATIO
df["NEW_C_RUNS_RATIO"] = df["Runs"] / df["CRuns"]
# CAREER BAT RATIO
df["NEW_C_ATBAT_RATIO"] = df["AtBat"] / df["CAtBat"]
# CAREER HITS RATIO
df["NEW_C_HITS_RATIO"] = df["Hits"] / df["CHits"]
# CAREER HMRUN RATIO
df["NEW_C_HMRUN_RATIO"] = df["HmRun"] / df["CHmRun"]
# CAREER RBI RATIO
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
# CAREER WALKS RATIO
df["NEW_C_WALKS_RATIO"] = df["Walks"] / df["CWalks"]
df["NEW_C_HIT_RATE"] = df["CHits"] / df["CAtBat"]
# PLAYER TYPE : RUNNER
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHits"]
# PLAYER TYPE : HIT AND RUN
df["NEW_C_HIT-AND-RUN"] = df["CRuns"] / df["CHits"]
# MOST VALUABLE HIT RATIO IN HITS
df["NEW_C_HMHITS_RATIO"] = df["CHmRun"] / df["CHits"]
# MOST VALUABLE HIT RATIO IN ALL SHOTS
df["NEW_C_HMATBAT_RATIO"] = df["CAtBat"] / df["CHmRun"]



df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)



#################################################
# Encoding işlemlerini gerçekleştirelim;
#################################################

# Label Encoding  ;
binary_cols = [col for col in cat_cols if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
for col in binary_cols:
    df = label_encoder(df, col)


# Rare Encoding ;
def rare_analyser(dataframe, target, cat_cols) :
    for col in cat_cols :
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                            "Ratio": 100 * (dataframe[col].value_counts() / len(dataframe)),
                            "Target_Mean": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "Salary",cat_cols)
# Rare'lik bir durum yok.


# One-Hot Encoder ;
# Kategori sayısı 2 ile 10 arasında olanları ohe_cols olarak toplayalım ve işlem uygulayalım ;
ohe_cols = [col for col in df.columns if df[col].nunique() > 2 and df[col].nunique() <= 20]
def one_hot_encoder(dataframe, categorial_cols, drop_first=True) :
    dataframe = pd.get_dummies(dataframe, columns=categorial_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, ohe_cols, True)

df.head()


all_possible_values = ["StandN", "StandA", "Descend", "Ascend"]
for val in all_possible_values:
    if f'NEW_PLAYER_PROGRESS{val}' not in df.columns:
        df[f'NEW_PLAYER_PROGRESS{val}'] = 0

all_possible_values = ["Junior", "Mid", "Senior", "Expert"]
for val in all_possible_values:
    if f'NEW_Years_Level{val}' not in df.columns:
        df[f'NEW_Years_Level{val}'] = 0



# Kategorileri güncelliyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


#################################################
# Numerik değişkenleri Standartlaştırıyoruz ;
#################################################


scaler = StandardScaler()
 # Aykırı ve eksik değerler var gibi duruyor. tekrar aykırı değer analizi yapalım.
for col in num_cols:
    replace_with_thresholds(df, col)
df[num_cols] = imputer.fit_transform(df[num_cols])
df[num_cols].head()

df[num_cols] = scaler.fit_transform(df[num_cols]) # şimdi oldu.


#################################################
# Model Oluşturuyoruz ;
#################################################


y = df["Salary"]
X = df.drop("Salary", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)




# LR model tahmini ; (4)
lr_model = LinearRegression().fit(X_train, y_train)
# Tahmin yapma
y_pred_lr = lr_model.predict(X_test)
# Modelin performansını değerlendirme
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_rmse # 0.691
r2 = r2_score(y_test, y_pred_lr) # 0.63
lr_model.score(X, y) # 0.70
np.mean(np.sqrt(-cross_val_score(lr_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.616
mean_absolute_error(y_test, y_pred_lr) # 0.50


# RF model tahmini ; (3)
rf_model = RandomForestRegressor(random_state=17).fit(X_train, y_train)
# Tahmin yapma
y_pred_rf = rf_model.predict(X_test)
# Modelin performansını değerlendirme
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_rmse # 0.661
r2 = r2_score(y_test, y_pred_rf) # 0.66
rf_model.score(X, y) # 0.827
np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.567
mean_absolute_error(y_test, y_pred_rf) # 0.44



# CART model tahmin ; (6)
cart_model = DecisionTreeRegressor(random_state=17).fit(X_train, y_train) # ---> öğrettik
# Tahmin yapma
y_pred_cart = cart_model.predict(X_test)
# Modelin performansını değerlendirme
cart_mse = mean_squared_error(y_test, y_pred_cart)
cart_rmse = np.sqrt(cart_mse)
cart_rmse # 1.16
r2 = r2_score(y_test, y_pred_cart) # -0.03
cart_model.score(X, y) # 0.59
np.mean(np.sqrt(-cross_val_score(cart_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.567
mean_absolute_error(y_test, y_pred_cart) # 0.70


# KNN model tahmin ; (5)
knn_model = KNeighborsRegressor().fit(X_train, y_train)
# Tahmin yapma
y_pred_knn = knn_model.predict(X_test)
# Modelin performansını değerlendirme
knn_mse = mean_squared_error(y_test, y_pred_knn)
knn_rmse = np.sqrt(knn_mse)
knn_rmse # 0.755
r2 = r2_score(y_test, y_pred_knn) # 0.563
knn_model.score(X, y) # 0.63
np.mean(np.sqrt(-cross_val_score(knn_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.66
mean_absolute_error(y_test, y_pred_knn) # 0.48


# XGB model tahmin ; (1)
xgboost_model = XGBRegressor(random_state=17).fit(X_train, y_train)
# Tahmin yapma
y_pred_xgboost = xgboost_model.predict(X_test)
# Modelin performansını değerlendirme
xgboost_mse = mean_squared_error(y_test, y_pred_xgboost)
xgboost_rmse = np.sqrt(xgboost_mse)
xgboost_rmse # 0.5999
r2 = r2_score(y_test, y_pred_xgboost) # 0.724
xgboost_model.score(X, y) # 0.89
np.mean(np.sqrt(-cross_val_score(xgboost_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.62
mean_absolute_error(y_test, y_pred_xgboost) # 0.419


# LightGB model tahmin ; (2)
lgb_model = LGBMRegressor(force_col_wise=True, verbose=-1).fit(X_train, y_train)
# Tahmin yapma
y_pred_lgb = lgb_model.predict(X_test)
# Modelin performansını değerlendirme
lgb_mse = mean_squared_error(y_test, y_pred_lgb)
lgb_rmse = np.sqrt(lgb_mse)
lgb_rmse # 0.635
r2 = r2_score(y_test, y_pred_lgb) # 0.690 -> ne kadar yüksekse o kadar iyi
lgb_model.score(X, y) # 0.84 -> Değişkenlerin birbirini anlama oranı, ne kadar yüksekse o kadar iyi
np.mean(np.sqrt(-cross_val_score(lgb_model, X, y, cv=10, scoring="neg_mean_squared_error"))) # RMSE 0.59
mean_absolute_error(y_test, y_pred_lgb) # 0.44

# Önümüzde kullanabileceğimiz sürekli hedef değişken olan modelleri oluşturduk ve hata oranlarına baktık. Hangi modelleri kullanacağımızı seçeceğiz.



##################################################################
# Yeni türettiğimiz değişkenlerin önem ve işe yarama oranını grafikle inceleyelim ;
##################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(xgboost_model, X_train) #(3)
plot_importance(lgb_model, X_train) #(1)
plot_importance(rf_model, X_train) #(2)

##################################################################
# Modelin Overfit olup olmadığını test edelim ;
##################################################################

train_accuracy = lgb_model.score(X_train, y_train) # 0.93
test_accuracy = lgb_model.score(X_test, y_test) # 0.69

train_accuracy = rf_model.score(X_train, y_train) # 0.93
test_accuracy = rf_model.score(X_test, y_test) # 0.66

train_accuracy = xgboost_model.score(X_train, y_train) # 0.99 ~ 1
test_accuracy = xgboost_model.score(X_test, y_test) # 0.72

# Modellerin hepsi overfit olmuş...
# Bunu düzeltmenin bikaç yolu var. Onlardan biri de modelin ön tanımlı değerleri ile oynamak ;
from sklearn.model_selection import GridSearchCV

# LGBM
lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                   "n_estimators": [250, 300, 350],
                   "colsample_bytree": [0.8, 1]}


lg_best_grid = GridSearchCV(lgb_model, lightgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lg_best_grid.best_params_
lg_final = lgb_model.set_params(**lg_best_grid.best_params_, random_state=17).fit(X, y)

train_accuracy = lg_final.score(X_train, y_train) # 0.99
test_accuracy = lg_final.score(X_test, y_test) # 0.99

# RF
rf_params = {"max_depth": [8, 15, 22],
             "max_features": [5, 7, "sqrt", "auto"],
             "min_samples_split": [20, 29, 39],
             "n_estimators": [100, 200]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

train_accuracy = rf_final.score(X_train, y_train) # 0.825
test_accuracy = rf_final.score(X_test, y_test) # 0.821


# XGBM
xgboost_params = {"learning_rate": [0.01, 0.05, 0.1],
                  "max_depth": range(1, 10),
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.3, 0.5, 1]}

xg_best_grid = GridSearchCV(xgboost_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xg_best_grid.best_params_
xg_final = xgboost_model.set_params(**xg_best_grid.best_params_, random_state=17).fit(X, y)

train_accuracy = xg_final.score(X_train, y_train) # 0.99
test_accuracy = xg_final.score(X_test, y_test) # 0.99

#############################################
# Yukarıdaki modellere göre overfit olayını düzeltmeye devam ediyoruz..

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
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel("Negative Mean Squared Error")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)

rf_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)], ["n_estimators", [100,200,300,400,500]]]
xgb_val_params = [["max_depth", range(1, 11)], ["learning_rate", [0.01, 0.05, 0.1]], ["n_estimators", [100,200,300,400,500]], ["colsample_bytree", [0.3, 0.5, 1]]]
lgb_val_params = [["learning_rate", [0.01, 0.05, 0.1]], ["n_estimators", [100,200,300,400,500]], ["colsample_bytree", [0.3, 0.5, 1]]]


for i in range(len(rf_val_params)) :
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

for i in range(len(xgb_val_params)) :
    val_curve_params(xgboost_model, X, y, xgb_val_params[i][0], xgb_val_params[i][1])

for i in range(len(lgb_val_params)) :
    val_curve_params(lgb_model, X, y, lgb_val_params[i][0], lgb_val_params[i][1])
# gözlemlere göre birkaç değeri tekrar değiştireceğiz;

rf_final.set_params(n_estimators=100, max_depth=3)
xg_final.set_params( max_depth=2)
lg_final.set_params(n_estimators=200, learning_rate=0.05, colsample_bytree=0.5)