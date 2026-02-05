import joblib
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

DATA_URLS = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1617/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1516/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1415/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1314/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1213/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1112/E0.csv",
    "https://www.football-data.co.uk/mmz4281/1011/E0.csv",
    "https://www.football-data.co.uk/mmz4281/0910/E0.csv",
    "https://www.football-data.co.uk/mmz4281/0809/E0.csv",
    "https://www.football-data.co.uk/mmz4281/0708/E0.csv",
    "https://www.football-data.co.uk/mmz4281/0607/E0.csv",
    "https://www.football-data.co.uk/mmz4281/0506/E0.csv"
]

FEATURES = ['Trailing_Team', 'Leading_Team', 'Trailing_Odds', 'Leading_Odds', 'Is_Trailing_Team_Home', 'Deficit', 'Odds_Diff']

class ComebackCalculator():
    def __init__(self, train, valid, test, xgb_model_file=None, calib_model_file=None):
        self.model_file = xgb_model_file
        print("Processing data...")
        self.train_data = self.process_data(train)
        self.valid_data = self.process_data(valid)
        self.test_data = self.process_data(test)
        if xgb_model_file is None:
            print("No Model Given! Starting training process")
            self.xgbmodel, self.calibrated_model = self.train()
        else:
            print("XGB Model found, loading...")
            self.xgbmodel = XGBClassifier()
            self.xgbmodel.load_model(fname=self.model_file)

        if calib_model_file is not None:
            self.calibrated_model = joblib.load(calib_model_file)

    def predict_one(self, x, calibrated=True):
        # x is a row from X
        # We presume x has been sanitised
        # Use the calibrated model by default
        return self.calibrated_model.predict_proba(x)[0][1] if calibrated else self.xgbmodel.predict_proba(x)[0][1] 
    
    def predict_all(self, X, calibrated=True):
        return self.calibrated_model.predict_proba(X)[:, 1] if calibrated else self.xgbmodel.predict_proba(X)[:, 1]
    
    def eval_features(self):
        importances = self.xgbmodel.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
        print(feature_importance_df.sort_values(by='Importance', ascending=False))
    
    def eval(self, valid=False, calibrate_graph=False, calibrated=True, histogram=False):
        if valid:
            dataset = "validation"
            X_test = self.valid_data[FEATURES]
            y_labels = self.valid_data['Comeback_Occurred']
        else:
            dataset = "test"
            X_test = self.test_data[FEATURES]
            y_labels = self.test_data['Comeback_Occurred']

        prob_all = self.predict_all(X_test, calibrated=calibrated)
        y_pred = (prob_all > 0.5).astype(int)
        brier = brier_score_loss(y_labels, prob_all)
        accuracy = accuracy_score(y_labels, y_pred)

        market = 1 / X_test['Trailing_Odds']
        market_brier = brier_score_loss(y_labels, market)

        print(f"EVALUATION")
        print(f"Model Accuracy on {dataset} set: {accuracy}")
        print(f"Model Brier score on {dataset} set: {brier}")
        print(f"Brier score on bookies odds: {market_brier}")
        print(confusion_matrix(y_labels, y_pred))
        print(classification_report(y_labels, y_pred))

        if calibrate_graph:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_labels, prob_all, n_bins=10)
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="XGBoost")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated") # The "Ideal" line
            plt.ylabel("Actual Fraction of Comebacks")
            plt.xlabel("Predicted Probability")
            plt.title("Calibration Curve: Is the Model Lying?")
            plt.legend()
            plt.show()

        if histogram:
            plt.hist(prob_all, bins=10, edgecolor='black')
            plt.title("Distribution of Comeback Predictions")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Number of Games")
            plt.show()

    def train(self):
        X = self.train_data[FEATURES]
        y = self.train_data['Comeback_Occurred']

        xgbmodel = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            learning_rate=0.05,
            max_depth=3,
            reg_lambda=10,
            gamma=1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=30
        )

        xgbmodel.fit(X,y)
        self.model_file = "comeback_calc.json"
        xgbmodel.save_model(self.model_file)

        calibrated = self.calibrate(xgbmodel, X, y)
        joblib.dump(calibrated, 'calibrated_model.pkl')

        return xgbmodel, calibrated
    
    def calibrate(self, model, train, labels):
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated.fit(train,labels)
        return calibrated

    def process_data(self, raw):
        cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTHG', 'HTAG', 'HTR', 'B365H', 'B365D', 'B365A']
        df = raw[cols].dropna().copy()

        home_trailing = df['HTHG'] < df['HTAG']
        away_trailing = df['HTHG'] > df['HTAG']

        conditions = [home_trailing, away_trailing]

        agnostic_df = pd.DataFrame()
        # TEAM MAPPING: Who is trailing vs who is leading?
        agnostic_df['Trailing_Team'] = np.select(conditions, [df['HomeTeam'], df['AwayTeam']], default='None')
        agnostic_df['Leading_Team']  = np.select(conditions, [df['AwayTeam'], df['HomeTeam']], default='None')

        # ODDS MAPPING: (Very important for determining if a favorite is behind)
        agnostic_df['Trailing_Odds'] = np.select(conditions, [df['B365H'], df['B365A']], default=0)
        agnostic_df['Leading_Odds']  = np.select(conditions, [df['B365A'], df['B365H']], default=0)

        # HOME ADVANTAGE: Is the team that's behind currently playing at home?
        agnostic_df['Is_Trailing_Team_Home'] = np.select(conditions, [1, 0], default=0)

        # CONTEXT: What is the deficit?
        agnostic_df['Deficit'] = np.abs(df['HTHG'].values - df['HTAG'].values)
        agnostic_df['Odds_Diff'] = agnostic_df['Trailing_Odds'] - agnostic_df['Leading_Odds']

        # THE TARGET: Did the team that was trailing actually go on to WIN?
        choices_win = [
            (df['FTR'] == 'H').astype(int), # did H end up winning?
            (df['FTR'] == 'A').astype(int)  # did A end up winning?
        ]
        agnostic_df['Comeback_Occurred'] = np.select(conditions, choices_win, default=0)

        # 4. FINAL STEP: Drop the 'None' rows (Draws)
        # This keeps only the games where a comeback was actually possible
        agnostic_df = agnostic_df[agnostic_df['Trailing_Team'] != 'None'].copy()
        for col in agnostic_df.select_dtypes(['object']).columns:
            agnostic_df[col] = agnostic_df[col].astype('category')
        return agnostic_df
    
if __name__ == "__main__":
    train_dfs = [pd.read_csv(url) for url in DATA_URLS[3:]]
    raw_train = pd.concat(train_dfs)

    valid_dfs = [pd.read_csv(url) for url in DATA_URLS[1:3]]
    raw_valid = pd.concat(valid_dfs)

    test_dfs = [pd.read_csv(DATA_URLS[0])]
    raw_test = pd.concat(test_dfs)

    cc = ComebackCalculator(raw_train, raw_valid, raw_test)
    cc.eval(valid=True, calibrate_graph=True, histogram=True)
    cc.eval_features()