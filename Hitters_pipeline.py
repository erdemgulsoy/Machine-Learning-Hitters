import joblib
import pandas as pd
from Hitters_research import hitter_data_prep, hyperparameter_optimization

def main () :
    df = pd.read_csv("datasetes/hitters.csv")
    X, y = hitter_data_prep(df)
    train_columns = X.columns.tolist()
    joblib.dump(train_columns, 'train_col.pkl')
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "telco_voting_clf.pkl")
    return voting_clf


if __name__ == "__main__":
    main()
