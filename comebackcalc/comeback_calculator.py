import joblib
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os


DATA_URLS = [
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
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

FEATURES = ['Trailing_Team', 'Leading_Team', 'Trailing_Odds', 'Leading_Odds', 'Is_Trailing_Team_Home', 'Deficit', 
            'Odds_Diff', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

class ComebackCalculator():
    def __init__(self, train=None, valid=None, test=None, xgb_model_file=None, calib_model_file=None):

        if train is not None and valid is not None and test is not None:
            self.train_data = self.process_data(train)
            self.valid_data = self.process_data(valid)
            self.test_data = self.process_data(test)
        else:
            train_dfs = [pd.read_csv(url) for url in DATA_URLS[3:]]
            raw_train = pd.concat(train_dfs)
            self.train_data = self.process_data(raw_train)

            valid_dfs = [pd.read_csv(url) for url in DATA_URLS[1:3]]
            raw_valid = pd.concat(valid_dfs)
            self.valid_data = self.process_data(raw_valid)

            test_dfs = [pd.read_csv(DATA_URLS[0])]
            raw_test = pd.concat(test_dfs)
            self.test_data = self.process_data(raw_test)

        if xgb_model_file is not None:
            self.model_file = xgb_model_file
            print("XGB Model found, loading...")
            self.xgbmodel = XGBClassifier()
            self.xgbmodel.load_model(fname=self.model_file)
            if calib_model_file is not None:
                self.calibrated_model = joblib.load(calib_model_file)
            else:
                print('Calibrated model not supplied! Training')
                _, self.calibrated_model = self.train(use_smote=True)
        else:
            print("No Model Given! Starting training process")
            self.xgbmodel, self.calibrated_model = self.train(use_smote=True)


    def predict_one(self, x, calibrated=True):
        # x is a row from X
        # We presume x has been sanitised
        # Use the calibrated model by default
        return self.calibrated_model.predict_proba(x)[0][1] if calibrated else self.xgbmodel.predict_proba(x)[0][1] 
    
    def predict_all(self, X, calibrated=True):
        return self.calibrated_model.predict_proba(X)[:, 1] if calibrated else self.xgbmodel.predict_proba(X)[:, 1]
    
    def diagnostics(self, dataset='validation', plot=True):
        """Run diagnostic metrics and return PR-AUC, ROC-AUC and best F1 threshold."""
        if dataset == 'validation':
            X = self.valid_data[FEATURES]
            y = self.valid_data['Comeback_Occurred']
        else:
            X = self.test_data[FEATURES]
            y = self.test_data['Comeback_Occurred']

        probs = self.predict_all(X, calibrated=True)
        print("Positive rate:", y.mean())
        print("Probability quantiles:", np.quantile(probs, [0, .01, .1, .25, .5, .75, .9, .99, 1.0]))

        prec, rec, thresholds = precision_recall_curve(y, probs)
        prauc = auc(rec, prec)
        roc = roc_auc_score(y, probs)

        if thresholds.size > 0:
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            best_idx = np.nanargmax(f1)
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5

        print(f"PR-AUC: {prauc:.4f}, ROC-AUC: {roc:.4f}, Best threshold (F1): {best_threshold:.4f}")

        if plot:
            plt.figure(figsize=(6,4))
            plt.plot(rec, prec, label=f'PR curve (AUC={prauc:.3f})')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.show()
            plt.hist(probs, bins=20, edgecolor='k'); plt.title('Predicted probability histogram'); plt.show()

        return {'prauc': prauc, 'roc_auc': roc, 'best_threshold': best_threshold}

    def tune_threshold(self, valid=True):
        res = self.diagnostics(dataset='validation' if valid else 'test', plot=False)
        return res['best_threshold']

    def eval(self, valid=False, calibrate_graph=False, calibrated=True, histogram=False, threshold=None):
        if valid:
            dataset = "validation"
            X_test = self.valid_data[FEATURES]
            y_labels = self.valid_data['Comeback_Occurred']
        else:
            dataset = "test"
            X_test = self.test_data[FEATURES]
            y_labels = self.test_data['Comeback_Occurred']

        prob_all = self.predict_all(X_test, calibrated=calibrated)

        # choose a better threshold by default
        if threshold is None:
            threshold = self.tune_threshold(valid=valid)
            print(f"Using threshold={threshold:.3f} (tuned from validation set)")

        y_pred = (prob_all > threshold).astype(int)
        brier = brier_score_loss(y_labels, prob_all)
        accuracy = accuracy_score(y_labels, y_pred)

        market = 1 / X_test['Trailing_Odds']
        market_brier = brier_score_loss(y_labels, market)

        from sklearn.metrics import precision_score, recall_score, f1_score
        print(f"EVALUATION")
        print(f"Model Accuracy on {dataset} set: {accuracy:.4f}")
        print(f"Model Brier score on {dataset} set: {brier:.4f}")
        print(f"Brier score on bookies odds (Bet365): {market_brier:.4f}")
        print(f"Precision: {precision_score(y_labels, y_pred):.4f}, Recall: {recall_score(y_labels, y_pred):.4f}, F1: {f1_score(y_labels, y_pred):.4f}")
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

    def train(self, use_smote=False, smote_k=5, n_estimators=200, early_stopping_rounds=50, scale_pos_weight=None):
        """Train XGBoost with improved defaults, optional SMOTE, early stopping and calibration on validation set."""
        X = self.train_data[FEATURES].copy()
        y = self.train_data['Comeback_Occurred']

        # dynamic class weighting
        if scale_pos_weight is None:
            scale_pos_weight = float((len(y) - y.sum()) / (y.sum() + 1e-12))
            print(f"Setting scale_pos_weight={scale_pos_weight:.2f} based on training data")

        xgbmodel = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            learning_rate=0.05,
            max_depth=4,
            reg_lambda=1.0,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            n_estimators=n_estimators,
            use_label_encoder=False,
            random_state=0,
            early_stopping_rounds=early_stopping_rounds
        )

        # Make sure validation set is numeric-coded when training on coded features
        X_valid_num = self.valid_data[FEATURES].copy()
        for col in X_valid_num.select_dtypes(['category']).columns:
            X_valid_num[col] = X_valid_num[col].cat.codes
        eval_set = [(X_valid_num, self.valid_data['Comeback_Occurred'])]

        # same for train
        X_num = X.copy()
        for col in X_num.select_dtypes(['category']).columns:
            X_num[col] = X_num[col].cat.codes
        print(X_num)

        # If using SMOTE, convert categorical features to numeric codes for resampling
        if use_smote:
            sm = SMOTE(k_neighbors=smote_k, random_state=0)
            X_res, y_res = sm.fit_resample(X_num, y)
            print(f"Performed SMOTE: {len(y_res)} samples (pos rate: {y_res.mean():.4f})")
            xgbmodel.fit(X_res, y_res, eval_set=eval_set, verbose=50)
        else:
            xgbmodel.fit(X_num, y, eval_set=eval_set, verbose=50)

        self.model_file = "comeback_calc.json"
        xgbmodel.save_model(self.model_file)

        # Calibrate on validation set using the prefit option
        X_valid = self.valid_data[FEATURES]
        y_valid = self.valid_data['Comeback_Occurred']
        calibrated = CalibratedClassifierCV(xgbmodel, method='sigmoid', cv='prefit')
        calibrated.fit(X_valid, y_valid)
        joblib.dump(calibrated, 'calibrated_model.pkl')

        return xgbmodel, calibrated
    
    def calibrate(self, model, X_valid=None, y_valid=None, cv='prefit'):
        """Calibrate a (prefit) model using a validation set, or perform CV-based calibration if cv != 'prefit'."""
        if cv == 'prefit':
            if X_valid is None or y_valid is None:
                raise ValueError("Provide X_valid and y_valid when using cv='prefit'")
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated.fit(X_valid, y_valid)
        else:
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=cv)
            calibrated.fit(self.train_data[FEATURES], self.train_data['Comeback_Occurred'])
        return calibrated

    def process_data(self, raw):
        df = raw.copy()

        home_trailing = df['HTHG'] < df['HTAG']
        away_trailing = df['HTHG'] > df['HTAG']

        conditions = [home_trailing, away_trailing]

        df['Trailing_Team'] = np.select(conditions, [df['HomeTeam'], df['AwayTeam']], default='None')
        df['Leading_Team']  = np.select(conditions, [df['AwayTeam'], df['HomeTeam']], default='None')
        df['Trailing_Odds'] = np.select(conditions, [df['B365H'], df['B365A']], default=0)
        df['Leading_Odds']  = np.select(conditions, [df['B365A'], df['B365H']], default=0)
        df['Is_Trailing_Team_Home'] = np.select(conditions, [1, 0], default=0)
        df['Deficit'] = np.abs(df['HTHG'].values - df['HTAG'].values)
        df['Odds_Diff'] = df['Trailing_Odds'] - df['Leading_Odds']

        # THE TARGET: Did the team that was trailing actually go on to WIN?
        choices_win = [
            (df['FTR'] == 'H').astype(int), # did H end up winning?
            (df['FTR'] == 'A').astype(int)  # did A end up winning?
        ]
        df['Comeback_Occurred'] = np.select(conditions, choices_win, default=0)

        # Remove draws
        df = df[df['Trailing_Team'] != 'None'].copy()
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].astype('category')
        return df
    
    def select_matches(self, raw):
        """
        Simply just select the matches where comebacks were possible and keep the match info
        """
        df = raw.copy()
        mask = (
            ((df['HTHG'] < df['HTAG'])) |
            ((df['HTHG'] > df['HTAG']))
        )
        return df[mask]


    def predict_season(self, season_csv, display=False):
        """
        Process, predict, and format an entire season's worth of comeback results

        Helpful for the UI
        """
        csv_clean = self.process_data(season_csv)
        X_test = csv_clean[FEATURES]
        y_labels = csv_clean['Comeback_Occurred']
        matches = self.select_matches(season_csv)
        prob_all = self.predict_all(X_test, calibrated=True)
        threshold = self.tune_threshold(valid=True)
        y_pred = (prob_all > threshold).astype(int)

        # Append predictions and actual outcome to match
        matches.insert(len(matches.columns), "Model_Prediction", y_pred)
        matches.insert(len(matches.columns), "Comeback_Probability", prob_all)
        matches.insert(len(matches.columns), "Actual_Outcome", y_labels)

        if display:
            for _, m in matches.iterrows():
                print(f"==== {m['Date']} | {m['Time']} ====")
                print(f"{m['HomeTeam']} vs. {m['AwayTeam']}")
                print(f"HALF-TIME RESULT: {m['HTHG']} - {m['HTAG']}")
                print(f"FULL-TIME RESULT: {m['FTHG']} - {m['FTAG']}")
                format_prediction = "Comeback" if m['Model_Prediction'] else "No Comeback"
                print(f"MODEL PREDICTON: {format_prediction} with probability {m['Comeback_Probability']}")

                format_outcome = "Comeback" if m['Actual_Outcome'] else "No Comeback"
                print(f"ACTUAL OUTCOME: {format_outcome}\n")

        return matches
if __name__ == "__main__":
    #cc = ComebackCalculator(train=raw_train, valid=raw_valid, test=raw_test)
    # Retrain with SMOTE and better defaults (will overwrite model trained in __init__ if present)
    # Diagnostics and evaluation
    #cc.diagnostics(dataset='validation', plot=True)
    #cc.eval(valid=True, calibrate_graph=True)

    xgb_file = os.path.abspath("comeback_calc.json")
    calib_file = os.path.abspath("calibrated_model.pkl")
    cc = ComebackCalculator(xgb_model_file=xgb_file, calib_model_file=calib_file)

    # test: get the 2526 
    tfts = pd.read_csv("https://www.football-data.co.uk/mmz4281/2526/E0.csv")
    matches = cc.predict_season(tfts, display=True)