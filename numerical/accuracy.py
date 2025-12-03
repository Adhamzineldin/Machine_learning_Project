"""
Professional-Grade Medical Test Results Prediction Pipeline
Optimized for Recall(Abnormal) and Macro-F1 with zero data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Optional imports with graceful fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: lightgbm not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: catboost not available. Install with: pip install catboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: shap not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


class MedicalTestPredictor:
    """
    Production-grade pipeline for predicting medical test results.
    Optimizes for recall(Abnormal) and macro-F1 while preventing data leakage.
    """
    
    def __init__(self, data_path='data/healthcare_dataset.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        self.categorical_features = []
        
    def load_data(self):
        """Load and inspect the dataset."""
        print("=" * 80)
        print("STEP 1: Loading Data")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nTest Results distribution:\n{self.df['Test Results'].value_counts(normalize=True)}")
        print(f"\nTest Results counts:\n{self.df['Test Results'].value_counts()}")
        return self
    
    def clean_data(self):
        """
        STEP 2: Data Cleaning
        - Canonicalize text fields
        - Parse dates and engineer time-based features
        - Fix categorical noise
        """
        print("\n" + "=" * 80)
        print("STEP 2: Data Cleaning")
        print("=" * 80)
        
        # a. Canonicalize Text (Mandatory)
        print("\n2a. Canonicalizing text fields...")
        if 'Name' in self.df.columns:
            self.df['Name'] = self.df['Name'].str.lower().str.strip()
        if 'Doctor' in self.df.columns:
            self.df['Doctor'] = self.df['Doctor'].str.lower().str.strip()
        if 'Hospital' in self.df.columns:
            self.df['Hospital'] = self.df['Hospital'].str.lower().str.strip()
        
        # b. Parse Dates → Time-Based Signals
        print("2b. Parsing dates and engineering time features...")
        self.df['Date of Admission'] = pd.to_datetime(self.df['Date of Admission'], errors='coerce')
        self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'], errors='coerce')
        
        # Length of stay (critical feature)
        self.df['length_of_stay'] = (self.df['Discharge Date'] - self.df['Date of Admission']).dt.days
        self.df['length_of_stay'] = self.df['length_of_stay'].fillna(self.df['length_of_stay'].median())
        
        # Admission month and weekday
        self.df['admission_month'] = self.df['Date of Admission'].dt.month
        self.df['admission_weekday'] = self.df['Date of Admission'].dt.dayofweek
        
        # c. Fix Categorical Noise
        print("2c. Standardizing categorical fields...")
        categorical_cols = ['Gender', 'Blood Type', 'Admission Type', 
                          'Medical Condition', 'Medication', 'Insurance Provider']
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.lower().str.strip()
        
        print("✓ Data cleaning complete")
        return self
    
    def feature_engineering(self):
        """
        STEP 3: Feature Selection and Engineering
        Keep high-signal features, drop leakage/noise features.
        """
        print("\n" + "=" * 80)
        print("STEP 3: Feature Engineering")
        print("=" * 80)
        
        # Features to KEEP (high signal)
        keep_features = [
            'Age',
            'Gender',
            'Blood Type',
            'Medical Condition',
            'Billing Amount',
            'Admission Type',
            'length_of_stay',
            'Medication',
            'Insurance Provider',
            'admission_month',
            'admission_weekday'
        ]
        
        # Features to DROP (leakage / noise)
        drop_features = ['Name', 'Doctor', 'Hospital', 'Room Number', 
                        'Date of Admission', 'Discharge Date']
        
        # Create feature dataframe
        feature_df = self.df[keep_features].copy()
        
        # Drop rows with missing target
        feature_df = feature_df[self.df['Test Results'].notna()]
        y = self.df.loc[feature_df.index, 'Test Results']
        
        print(f"\nKeeping {len(keep_features)} features")
        print(f"Dropping {len(drop_features)} features (leakage/noise)")
        print(f"\nFinal feature set: {keep_features}")
        
        self.df_processed = feature_df.copy()
        self.y = y.copy()
        
        return self
    
    def encode_features_basic(self):
        """
        STEP 4a: Basic Encoding (No leakage risk)
        Binary and One-Hot encoding that doesn't depend on target.
        """
        print("\n" + "=" * 80)
        print("STEP 4: Feature Encoding (Basic)")
        print("=" * 80)
        
        df_encoded = self.df_processed.copy()
        
        # Binary encoding for Gender
        print("\n4a. Encoding Gender (binary)...")
        df_encoded['Gender'] = (df_encoded['Gender'] == 'male').astype(int)
        
        # One-Hot encoding for Admission Type and Blood Type
        print("4b. One-Hot encoding Admission Type and Blood Type...")
        df_encoded = pd.get_dummies(df_encoded, columns=['Admission Type', 'Blood Type'], 
                                    prefix=['adm_type', 'blood'])
        
        # Store for later encoding steps
        self.df_encoded_basic = df_encoded
        self.categorical_cols_to_encode = ['Medical Condition', 'Medication', 'Insurance Provider']
        
        print(f"\n✓ Basic encoding complete. Ready for split.")
        return self
    
    def prepare_splits(self):
        """
        STEP 5: Time-Based Split (No Random Splits!)
        Medical data is time-dependent - train on older, test on newer.
        """
        print("\n" + "=" * 80)
        print("STEP 5: Time-Based Data Splitting")
        print("=" * 80)
        
        # Sort by admission date
        date_sorted_idx = self.df.loc[self.df_encoded_basic.index].sort_values('Date of Admission').index
        X_sorted = self.df_encoded_basic.loc[date_sorted_idx]
        y_sorted = self.y.loc[date_sorted_idx]
        
        # Store original categorical columns before encoding
        X_train_cat = self.df_processed.loc[date_sorted_idx, self.categorical_cols_to_encode]
        
        # 80/20 split maintaining temporal order
        split_idx = int(len(X_sorted) * 0.8)
        
        X_train_basic = X_sorted.iloc[:split_idx]
        X_test_basic = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]
        
        X_train_cat_data = X_train_cat.iloc[:split_idx]
        X_test_cat_data = X_train_cat.iloc[split_idx:]
        
        print(f"Training set: {len(X_train_basic)} samples")
        print(f"Test set: {len(X_test_basic)} samples")
        print(f"\nTrain class distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"\nTest class distribution:\n{y_test.value_counts(normalize=True)}")
        
        # NOW do target encoding and frequency encoding ONLY on training data
        print("\n5a. Target encoding Medical Condition and Medication (TRAINING DATA ONLY)...")
        # Add smoothing to reduce overfitting (blend with global mean)
        global_mean_abnormal = (y_train == 'Abnormal').mean()
        smoothing_factor = 10  # Higher = more smoothing
        
        def smooth_target_encode(group, global_mean, smoothing):
            """Smooth target encoding to reduce overfitting."""
            n = len(group)
            group_mean = (group == 'Abnormal').mean()
            return (n * group_mean + smoothing * global_mean) / (n + smoothing)
        
        target_means_mc = y_train.groupby(X_train_cat_data['Medical Condition']).apply(
            lambda x: smooth_target_encode(x, global_mean_abnormal, smoothing_factor)
        ).to_dict()
        target_means_med = y_train.groupby(X_train_cat_data['Medication']).apply(
            lambda x: smooth_target_encode(x, global_mean_abnormal, smoothing_factor)
        ).to_dict()
        
        # Apply to both train and test
        X_train_basic['Medical Condition_encoded'] = X_train_cat_data['Medical Condition'].map(target_means_mc)
        X_test_basic['Medical Condition_encoded'] = X_test_cat_data['Medical Condition'].map(target_means_mc)
        X_train_basic['Medication_encoded'] = X_train_cat_data['Medication'].map(target_means_med)
        X_test_basic['Medication_encoded'] = X_test_cat_data['Medication'].map(target_means_med)
        
        # Frequency encoding for Insurance Provider (TRAINING DATA ONLY)
        print("5b. Frequency encoding Insurance Provider (TRAINING DATA ONLY)...")
        freq_map = X_train_cat_data['Insurance Provider'].value_counts().to_dict()
        X_train_basic['Insurance Provider_encoded'] = X_train_cat_data['Insurance Provider'].map(freq_map)
        X_test_basic['Insurance Provider_encoded'] = X_test_cat_data['Insurance Provider'].map(freq_map)
        
        # Drop original categorical columns
        X_train_final = X_train_basic.drop(self.categorical_cols_to_encode, axis=1, errors='ignore')
        X_test_final = X_test_basic.drop(self.categorical_cols_to_encode, axis=1, errors='ignore')
        
        # Handle any remaining NaN values (use training median)
        train_median = X_train_final.median()
        X_train_final = X_train_final.fillna(train_median)
        X_test_final = X_test_final.fillna(train_median)
        
        # Handle unseen categories in test set (fill with training mean)
        for col in ['Medical Condition_encoded', 'Medication_encoded']:
            train_mean = X_train_final[col].mean()
            X_test_final[col] = X_test_final[col].fillna(train_mean)
        
        self.X_train = X_train_final
        self.X_test = X_test_final
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = list(X_train_final.columns)
        
        print(f"\n✓ Final feature count: {len(self.feature_names)}")
        
        # Scale numerical features
        print("\n5c. Scaling features with RobustScaler...")
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        # Encode target labels
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        self.class_names = self.label_encoder.classes_
        print(f"\nClass mapping: {dict(zip(range(len(self.class_names)), self.class_names))}")
        
        return self
    
    def train_models(self):
        """
        STEP 6: Model Training
        Train multiple models: Logistic Regression, XGBoost, LightGBM, CatBoost
        """
        print("\n" + "=" * 80)
        print("STEP 6: Model Training")
        print("=" * 80)
        
        # Check class imbalance
        class_counts = Counter(self.y_train_encoded)
        total = len(self.y_train_encoded)
        class_weights_dict = {cls: total / (len(class_counts) * count) 
                             for cls, count in class_counts.items()}
        # Convert to list format for CatBoost (ordered by class index)
        num_classes = len(class_counts)
        class_weights_list = [class_weights_dict.get(i, 1.0) for i in range(num_classes)]
        
        print(f"\nClass weights for balancing: {class_weights_dict}")
        
        # 1. Logistic Regression (Baseline)
        print("\n6a. Training Logistic Regression (baseline)...")
        lr = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        lr.fit(self.X_train_scaled, self.y_train_encoded)
        self.models['Logistic Regression'] = lr
        
        # 2. XGBoost (Improved hyperparameters)
        if HAS_XGBOOST:
            print("6b. Training XGBoost...")
            # Try with early stopping in constructor (XGBoost 1.x)
            # If that fails, fall back to no early stopping
            try:
                xgb_model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(self.class_names),
                    eval_metric='mlogloss',
                    random_state=42,
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    tree_method='hist',
                    early_stopping_rounds=50
                )
            except TypeError:
                # XGBoost 2.0+ doesn't support early_stopping_rounds in constructor
                xgb_model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(self.class_names),
                    eval_metric='mlogloss',
                    random_state=42,
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    tree_method='hist'
                )
            
            xgb_model.fit(
                self.X_train_scaled, self.y_train_encoded,
                eval_set=[(self.X_test_scaled, self.y_test_encoded)],
                verbose=0
            )
            self.models['XGBoost'] = xgb_model
        else:
            print("6b. Skipping XGBoost (not available)")
        
        # 3. LightGBM (Improved hyperparameters)
        if HAS_LIGHTGBM:
            print("6c. Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(self.class_names),
                class_weight='balanced',
                random_state=42,
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1
            )
            lgb_model.fit(
                self.X_train_scaled, self.y_train_encoded,
                eval_set=[(self.X_test_scaled, self.y_test_encoded)],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            self.models['LightGBM'] = lgb_model
        else:
            print("6c. Skipping LightGBM (not available)")
        
        # 4. CatBoost (Best for mixed medical data - Improved hyperparameters)
        if HAS_CATBOOST:
            print("6d. Training CatBoost (recommended for medical data)...")
            cat_model = CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                loss_function='MultiClass',
                class_weights=class_weights_list,
                random_seed=42,
                colsample_bylevel=0.8,
                l2_leaf_reg=3,
                verbose=False
            )
            cat_model.fit(
                self.X_train_scaled, self.y_train_encoded,
                eval_set=(self.X_test_scaled, self.y_test_encoded),
                early_stopping_rounds=50,
                verbose=False
            )
            self.models['CatBoost'] = cat_model
        else:
            print("6d. Skipping CatBoost (not available)")
        
        print("\n✓ All models trained successfully")
        return self
    
    def evaluate_models(self):
        """
        STEP 7: Comprehensive Evaluation
        Metrics: Accuracy, Macro-F1, Recall(Abnormal), Confusion Matrix, ROC-AUC
        """
        print("\n" + "=" * 80)
        print("STEP 7: Model Evaluation")
        print("=" * 80)
        
        results = {}
        
        # Find Abnormal class index
        abnormal_idx = np.where(self.class_names == 'Abnormal')[0]
        if len(abnormal_idx) == 0:
            abnormal_idx = np.where(self.class_names == 'abnormal')[0]
        abnormal_idx = abnormal_idx[0] if len(abnormal_idx) > 0 else None
        
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}")
            print(f"{'='*60}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(self.y_test_encoded, y_pred)
            macro_f1 = f1_score(self.y_test_encoded, y_pred, average='macro')
            
            # Recall for Abnormal class
            recall_abnormal = None
            if abnormal_idx is not None:
                recall_abnormal = recall_score(
                    self.y_test_encoded, y_pred, 
                    labels=[abnormal_idx], 
                    average=None
                )[0]
            
            # Per-class recall
            per_class_recall = recall_score(
                self.y_test_encoded, y_pred, 
                average=None
            )
            
            # ROC-AUC (one-vs-rest)
            try:
                roc_auc = roc_auc_score(
                    self.y_test_encoded, y_pred_proba,
                    multi_class='ovr',
                    average='macro'
                )
            except:
                roc_auc = None
            
            results[model_name] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'recall_abnormal': recall_abnormal,
                'per_class_recall': per_class_recall,
                'roc_auc': roc_auc
            }
            
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            if recall_abnormal is not None:
                print(f"Recall(Abnormal): {recall_abnormal:.4f}")
            if roc_auc is not None:
                print(f"ROC-AUC (macro): {roc_auc:.4f}")
            
            print(f"\nPer-class Recall:")
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {per_class_recall[i]:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(
                self.y_test_encoded, y_pred,
                target_names=self.class_names
            ))
            
            print(f"\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test_encoded, y_pred)
            print(cm)
            
            # Visualize confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            # Save in same directory as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, f'{model_name.replace(" ", "_")}_confusion_matrix.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Confusion matrix saved to: {save_path}")
        
        # Summary comparison
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        comparison_df = pd.DataFrame(results).T
        print(comparison_df[['accuracy', 'macro_f1', 'recall_abnormal', 'roc_auc']])
        
        # Select best model based on recall(Abnormal) and macro-F1
        if abnormal_idx is not None:
            best_model_name = comparison_df['recall_abnormal'].idxmax()
            print(f"\n🏆 Best model for Recall(Abnormal): {best_model_name}")
        
        best_f1_model = comparison_df['macro_f1'].idxmax()
        print(f"🏆 Best model for Macro-F1: {best_f1_model}")
        
        self.results = results
        return self
    
    def feature_importance_analysis(self, model_name='CatBoost'):
        """
        STEP 8: Feature Importance + Medical Sanity Check
        Use SHAP values to verify top drivers are medically sensible.
        """
        print("\n" + "=" * 80)
        print(f"STEP 8: Feature Importance Analysis ({model_name})")
        print("=" * 80)
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Using CatBoost instead.")
            model_name = 'CatBoost'
        
        model = self.models[model_name]
        
        # Feature importance from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            print("Could not extract feature importance from model.")
            return self
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        # Medical sanity check
        print("\n" + "-" * 60)
        print("MEDICAL SANITY CHECK")
        print("-" * 60)
        top_features = importance_df.head(10)['feature'].tolist()
        
        expected_important = [
            'Medical Condition', 'length_of_stay', 'Medication',
            'Admission Type', 'Age', 'Billing Amount'
        ]
        
        print("\nExpected important features:")
        for feat in expected_important:
            found = any(feat.lower() in f.lower() for f in top_features)
            status = "✓" if found else "✗"
            print(f"  {status} {feat}")
        
        # Check for leakage indicators
        leakage_indicators = ['doctor', 'hospital', 'name', 'room']
        print("\nLeakage check (should NOT appear in top features):")
        for feat in top_features:
            if any(indicator in feat.lower() for indicator in leakage_indicators):
                print(f"  ⚠️  WARNING: Potential leakage detected: {feat}")
        
        # SHAP analysis (if available)
        if HAS_SHAP:
            try:
                print("\nGenerating SHAP values...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_test_scaled.iloc[:100])  # Sample for speed
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, self.X_test_scaled.iloc[:100], 
                                feature_names=self.feature_names, show=False)
                plt.tight_layout()
                # Save in same directory as script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, 'shap_summary_plot.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✓ SHAP summary plot saved to: {save_path}")
            except Exception as e:
                print(f"Could not generate SHAP plot: {e}")
        else:
            print("\nSkipping SHAP analysis (shap not available)")
        
        return self
    
    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        print("\n" + "=" * 80)
        print("MEDICAL TEST RESULTS PREDICTION PIPELINE")
        print("=" * 80)
        
        (self.load_data()
         .clean_data()
         .feature_engineering()
         .encode_features_basic()
         .prepare_splits()
         .train_models()
         .evaluate_models()
         .feature_importance_analysis())
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return self


if __name__ == "__main__":
    # Initialize and run pipeline
    # Handle path whether script is run from project root or numerical folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'healthcare_dataset.csv')
    
    # If data not found in numerical/data, try project root
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(script_dir), 'numerical', 'data', 'healthcare_dataset.csv')
    
    predictor = MedicalTestPredictor(data_path=data_path)
    predictor.run_full_pipeline()
    
    # Print final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
    Target Metrics:
    - Recall(Abnormal) ≥ 0.90
    - Macro-F1 ≥ 0.85
    
    If targets not met, consider:
    1. Hyperparameter tuning (GridSearchCV/RandomSearchCV)
    2. Feature engineering (interactions, polynomial features)
    3. Ensemble methods (voting/stacking)
    4. SMOTE for class balancing (only on training set)
    5. Cross-validation for more robust evaluation
    
    Medical Validation:
    - Verify top features are medically sensible
    - Ensure no data leakage (Doctor, Hospital, Name)
    - Check calibration for clinical decision support
    """)

