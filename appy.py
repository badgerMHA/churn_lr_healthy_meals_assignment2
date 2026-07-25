{
  "metadata": {
    "kernelspec": {
      "display_name": "Jupyter Notebook",
      "name": "jupyter"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5,
  "cells": [
    {
      "id": "s01",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": true, "collapsed": false },
      "source": "Logistic regression version of Assignment 2 churn prediction notebook — manual OneHotEncoder + separate model/encoder artifacts, matching the instructor's saved-artifact pattern\n*Co-authored with CoCo*"
    },
    {
      "id": "s02",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": true, "collapsed": false },
      "source": "# Assignment 2\n\nPrepared by: Mohammed H. Ahmed\n\nGB 895"
    },
    {
      "id": "s03",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "Churn Prediction Model\n\nPurpose: predict customers who are at risk of churning in 2023 so that intervention can be made.\n        These customers must have had an active subscription as of 1-1-2023\n\nProduct: Healthy Meals\n\nModeling approach:\n\n    - Target variable is renewed column (1 = yes, 0 = no)\n\n    - Algorithm: Logistic Regression\n\n    - Preprocessing: manual SimpleImputer + StandardScaler (numeric) and\n      SimpleImputer + OneHotEncoder (categorical), saved as a preprocessing\n      artifact separate from the model — matching the instructor's two-file\n      (model.pkl + encoder.pkl) save pattern.\n"
    },
    {
      "id": "s04",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "Notebook Structure (same 16 step process as in lecture)\n1. Data prep\n2. Convert to pandas dataframe\n3. Import needed libraries and packages\n4. Feature engineering\n5. Train/test split\n6. Model train\n7. Test and predict/score\n8. Plot ROC\n9. Save model + encoder (two artifacts)\n10. Create stage\n11. Write stage\n12. Prep for scoring/predicting\n13. Load scoring dataset into a pandas dataframe\n14. Encode the scoring (load saved artifacts, transform raw data)\n15. Generate renewal probability scores\n16. Attach scores to the scoring dataframe"
    },
    {
      "id": "s05",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Import libraries" },
      "source": "# ── STEP 1 — Import libraries ─────────────────────────────────────────────────\nimport pandas as pd\nimport numpy as np\nfrom snowflake.snowpark.context import get_active_session\n\nsession = get_active_session()\nprint(\"Session connected.\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s06",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": true, "collapsed": false },
      "source": "# Part 1 of Assignment 2"
    },
    {
      "id": "s07",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "Step 1: Data prep to identify customers fitting our criteria\n\nCriteria:\n    - Must have had a subscription as of 1-1-2023\n\n    - Subscription must be ending/up for renewal in 2023\n\n    - Target variable is renewed column in ol_subscriptions table\n\nFeature sources\n\n    - ol_subscriptions - subscription start and end dates, product, renewal outcome\n    - ol_customer_demog - age, education, income level, device type, tech comfort score\n    - ol_customer_activity - number of sessions, total session length\n"
    },
    {
      "id": "s08",
      "cell_type": "code",
      "metadata": { "language": "sql", "resultVariableName": "training_data", "title": "Load training data" },
      "source": "-- ── STEP 2 — Load training data ───────────────────────────────────────────────\n-- Pull cohort data by using Quiz 4 code as starting point\n-- Add two new features:\n    -- (1) # of active quarters in 2022 (use count distinct on the quarters obtained by applying date_trunc\n    -- to extract quarters from activity date)\n    -- (2) Average sessions per active quarter (total sessions / # of active quarters\n    -- make sure you handle nulls and division by zeros where needed)\n    -- Added days since last activity feature...researched using AI\n-- Left joining ol_customer_demog to two other tables...looks like on customer_id\n\n\nSELECT renewed, s.customer_id,\nSUM(CASE WHEN YEAR(activity_date) = 2022 THEN num_sessions else 0 end) as total_sessions,\nSUM(CASE WHEN YEAR(activity_date) = 2022 then total_session_length else 0 end) as total_session_duration,\nCOUNT(CASE WHEN YEAR(activity_date) = 2022 then 1 else NULL end) as active_days,\nCOUNT(DISTINCT CASE WHEN YEAR(activity_date) = 2022 THEN DATE_TRUNC('QUARTER', activity_date) END) as count_active_quarters,\nDIV0NULL(SUM(CASE WHEN YEAR(activity_date) = 2022 THEN num_sessions ELSE 0 END),\n        NULLIF(COUNT(DISTINCT CASE WHEN YEAR(activity_date) = 2022 THEN DATE_TRUNC('QUARTER', activity_date) END), 0)\n) as avg_sessions_per_quarter,\nDATEDIFF('day',\n    MAX(CASE WHEN YEAR(activity_date) = 2022 THEN activity_date END),\n    '2023-01-01'\n) AS days_since_last_activity\n\nFROM\n\nSUBSCRIPTION_DATA.PROJECT_DATA.OL_SUBSCRIPTIONS s\n\nLEFT JOIN\n\nSUBSCRIPTION_DATA.PROJECT_DATA.OL_CUSTOMER_ACTIVITY a\n\nON a.customer_id = s.customer_id\nAND\na.product = s.product\n\nLEFT JOIN\n\nSUBSCRIPTION_DATA.PROJECT_DATA.ol_customer_demog d\n\nON d.customer_id = a.customer_id\n\nWHERE\n\na.product = 'Healthy Meals'\nAND\n'2023-01-01' >= subscription_start_date\nAND\n'2023-01-01' < subscription_end_date\nAND YEAR(SUBSCRIPTION_END_DATE) = 2023\n\nGROUP BY renewed, s.customer_id;",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s09",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Convert to pandas" },
      "source": "# ── STEP 2 — Load SQL result into pandas ─────────────────────────────────────\n# The SQL cell above returns a Snowpark DataFrame (training_data).\n# We convert it to a pandas DataFrame for use with scikit-learn.\n# Note: Snowflake returns all column names in UPPERCASE.\n\nmodel_df = training_data.to_pandas() if hasattr(training_data, 'to_pandas') else training_data.copy()\n\nprint(f\"Rows: {len(model_df):,}\")\nprint(f\"Columns: {model_df.columns.tolist()}\")\nprint(f\"Renewal rate (% who renewed): {model_df['RENEWED'].mean():.1%}\")\nmodel_df.head()",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s10",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Library imports" },
      "source": "# ── STEP 3 — Library imports ──────────────────────────────────────────────────\n# Manual preprocessing this time (no Pipeline/ColumnTransformer) so that every\n# fitted preprocessing object can be saved as its own artifact, separate from\n# the model — matching the instructor's two-file (model + encoder) pattern.\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import roc_auc_score, roc_curve, classification_report\nimport matplotlib.pyplot as plt\n\nprint(\"Libraries loaded successfully.\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s11",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Feature engineering" },
      "source": "# ── STEP 4 — Feature engineering ──────────────────────────────────────────────\n#\n# Actions taken:\n#   1. Reload model_df from the SQL result to get a fresh copy with CUSTOMER_ID.\n#   2. Join demographic data (OL_CUSTOMER_DEMOG) to bring in categorical features.\n#   3. Drop CUSTOMER_ID — an identifier, not a predictive feature.\n#\n# Final feature set:\n#   Numeric     : TOTAL_SESSIONS, TOTAL_SESSION_DURATION, ACTIVE_DAYS,\n#                 COUNT_ACTIVE_QUARTERS, AVG_SESSIONS_PER_QUARTER,\n#                 DAYS_SINCE_LAST_ACTIVITY, AGE, TECH_COMFORT_SCORE\n#   Categorical : INCOME_LEVEL, EDUCATION, DEVICE_TYPE\n\nfrom snowflake.snowpark.context import get_active_session\n\n# Reload model_df from the SQL cell result to ensure CUSTOMER_ID is present\nmodel_df = training_data.to_pandas() if hasattr(training_data, 'to_pandas') else training_data.copy()\n\n# Join demographic data\nsession = get_active_session()\ndemog_df = session.table('SUBSCRIPTION_DATA.PROJECT_DATA.OL_CUSTOMER_DEMOG').to_pandas()\nmodel_df = model_df.merge(demog_df, on='CUSTOMER_ID', how='left')\n\n# Drop the identifier column\nmodel_df = model_df.drop(columns=['CUSTOMER_ID'])\n\n# Define column types\ncategorical_cols = ['INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE']\nnumeric_cols = [c for c in model_df.columns if c not in categorical_cols and c != 'RENEWED']\n\ndf_encoded = model_df\n\nprint(f\"Feature matrix shape: {df_encoded.shape}\")\nprint(f\"Columns: {df_encoded.columns.tolist()}\")\ndf_encoded.head()",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s12",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Train/Test split" },
      "source": "# ── STEP 5 — Train / test split ───────────────────────────────────────────────\n#\n# We hold out 30% of the data as a test set to evaluate the model on\n# observations it has never seen during training.\n\nX = df_encoded.drop(columns=['RENEWED'])   # feature matrix\ny = df_encoded['RENEWED']                  # target: 1 = renewed, 0 = churned\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.3, random_state=42\n)\n\nprint(f\"Training set : {len(X_train):,} rows  (renewal rate: {y_train.mean():.1%})\")\nprint(f\"Test set     : {len(X_test):,} rows  (renewal rate: {y_test.mean():.1%})\")\nprint(f\"\\nFeatures used in model:\")\nfor col in X.columns:\n    print(f\"  {col}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s13",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Manual preprocessing + train" },
      "source": "# ── STEP 6 — Manual preprocessing + model training ───────────────────────────\n#\n# Unlike a Pipeline/ColumnTransformer, each preprocessing step is fit here\n# explicitly, and each fitted object (imputer, scaler, encoder) is kept as\n# its own variable so it can be saved and reused at inference time —\n# matching the model + encoder two-artifact pattern from the instructor's\n# example.\n#\n# IMPORTANT: every transformer below is FIT on X_train only, then applied\n# with .transform (not .fit_transform) to X_test. Fitting on test data would\n# leak information from the holdout set into preprocessing statistics\n# (medians, means/std, category list) and bias the evaluation.\n\n# --- Numeric features: median-impute missing values, then standardise ---\nnumeric_imputer = SimpleImputer(strategy='median')\nX_train_num = numeric_imputer.fit_transform(X_train[numeric_cols])\nX_test_num = numeric_imputer.transform(X_test[numeric_cols])\n\nscaler = StandardScaler()\nX_train_num = scaler.fit_transform(X_train_num)\nX_test_num = scaler.transform(X_test_num)\n\n# --- Categorical features: mode-impute missing values, then one-hot encode ---\ncategorical_imputer = SimpleImputer(strategy='most_frequent')\nX_train_cat_imputed = categorical_imputer.fit_transform(X_train[categorical_cols])\nX_test_cat_imputed = categorical_imputer.transform(X_test[categorical_cols])\n\nencoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\nX_train_cat = encoder.fit_transform(X_train_cat_imputed)\nX_test_cat = encoder.transform(X_test_cat_imputed)\n\n# --- Combine numeric + categorical into final feature matrices ---\nX_train_final = np.hstack([X_train_num, X_train_cat])\nX_test_final = np.hstack([X_test_num, X_test_cat])\n\n# --- Train Logistic Regression on the fully-encoded numeric array ---\nmodel = LogisticRegression(max_iter=1000, random_state=42)\nmodel.fit(X_train_final, y_train)\n\nprint(\"Logistic Regression model training complete.\")\nprint(f\"Final feature matrix shape: {X_train_final.shape}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s14",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "## Model Evaluation"
    },
    {
      "id": "s15",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "AUC-ROC evaluation" },
      "source": "# ── STEP 7 — Evaluate: AUC-ROC ────────────────────────────────────────────────\ny_probs = model.predict_proba(X_test_final)[:, 1]\nauc_score = roc_auc_score(y_test, y_probs)\n\nfpr, tpr, _ = roc_curve(y_test, y_probs)\nplt.figure(figsize=(8, 6))\nplt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.4f})')\nplt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('ROC Curve — Logistic Regression')\nplt.legend(loc='lower right')\nplt.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()\n\nprint(f\"AUC-ROC: {auc_score:.4f}\")\nif auc_score >= 0.80:\n    print(\"✓ Model meets the target AUC threshold of 0.80\")\nelse:\n    print(\"✗ Model does NOT meet the target AUC threshold of 0.80\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s16",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Classification report" },
      "source": "# ── STEP 7b — Classification report ───────────────────────────────────────────\ny_pred = model.predict(X_test_final)\nprint(classification_report(y_test, y_pred, target_names=['Churned', 'Renewed']))",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s17",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": true, "collapsed": false },
      "source": "# Model 2 — Drop DAYS_SINCE_LAST_ACTIVITY\n\nTest whether dropping DAYS_SINCE_LAST_ACTIVITY improves AUC."
    },
    {
      "id": "s18",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Model 2 — feature ablation" },
      "source": "# Model 2 — drop DAYS_SINCE_LAST_ACTIVITY to test if it improves AUC.\n# Preprocessing is refit from scratch on the reduced numeric column list —\n# reusing Model 1's imputer/scaler would be fit on the wrong column set.\n\nnumeric_cols_2 = [c for c in numeric_cols if c != 'DAYS_SINCE_LAST_ACTIVITY']\n\nnumeric_imputer_2 = SimpleImputer(strategy='median')\nX_train_num_2 = numeric_imputer_2.fit_transform(X_train[numeric_cols_2])\nX_test_num_2 = numeric_imputer_2.transform(X_test[numeric_cols_2])\n\nscaler_2 = StandardScaler()\nX_train_num_2 = scaler_2.fit_transform(X_train_num_2)\nX_test_num_2 = scaler_2.transform(X_test_num_2)\n\n# Categorical preprocessing is unchanged in concept, but refit independently\n# so Model 2 is fully self-contained if it ever needs to be saved on its own.\ncategorical_imputer_2 = SimpleImputer(strategy='most_frequent')\nX_train_cat_imputed_2 = categorical_imputer_2.fit_transform(X_train[categorical_cols])\nX_test_cat_imputed_2 = categorical_imputer_2.transform(X_test[categorical_cols])\n\nencoder_2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\nX_train_cat_2 = encoder_2.fit_transform(X_train_cat_imputed_2)\nX_test_cat_2 = encoder_2.transform(X_test_cat_imputed_2)\n\nX_train_final_2 = np.hstack([X_train_num_2, X_train_cat_2])\nX_test_final_2 = np.hstack([X_test_num_2, X_test_cat_2])\n\nmodel_2 = LogisticRegression(max_iter=1000, random_state=42)\nmodel_2.fit(X_train_final_2, y_train)\n\ny_probs_2 = model_2.predict_proba(X_test_final_2)[:, 1]\nauc_score_2 = roc_auc_score(y_test, y_probs_2)\n\nprint(f\"Model 1 AUC (all features):                    {auc_score:.4f}\")\nprint(f\"Model 2 AUC (without DAYS_SINCE_LAST_ACTIVITY): {auc_score_2:.4f}\")\nprint(f\"Difference: {auc_score_2 - auc_score:+.4f}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s19",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "**Decision:** keeping Model 1 (all features, including DAYS_SINCE_LAST_ACTIVITY) since it produced the higher AUC. Model 1 — and its associated preprocessing objects (numeric_imputer, scaler, categorical_imputer, encoder) — is what gets saved and deployed below."
    },
    {
      "id": "s20",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Save model + encoder" },
      "source": "# ── STEP 9 — Save the trained model and preprocessing artifacts ─────────────\n#\n# Two files are saved, matching the instructor's model + encoder pattern:\n#\n#   1. churn_lr_healthy_meals.pkl        — the fitted LogisticRegression model\n#   2. churn_encoder_healthy_meals.pkl   — every fitted preprocessing object\n#      needed to transform new/raw data identically to training. Bundled as\n#      a single dict so there is still just one 'encoder' file to track,\n#      even though it internally holds four fitted transformers plus the\n#      column-name lists needed to apply them in the right order.\n#\n# Both files must be uploaded to the stage together — the model is unusable\n# without the preprocessing artifact to prepare new data for it, and vice versa.\n\nimport pickle\n\nmodel_path = '/tmp/churn_lr_healthy_meals.pkl'\nencoder_path = '/tmp/churn_encoder_healthy_meals.pkl'\n\nwith open(model_path, 'wb') as f:\n    pickle.dump(model, f)\n\npreprocessing_artifacts = {\n    'numeric_imputer': numeric_imputer,\n    'scaler': scaler,\n    'categorical_imputer': categorical_imputer,\n    'encoder': encoder,\n    'numeric_cols': numeric_cols,\n    'categorical_cols': categorical_cols\n}\n\nwith open(encoder_path, 'wb') as f:\n    pickle.dump(preprocessing_artifacts, f)\n\nprint(f\"Model saved   : {model_path}\")\nprint(f\"Encoder saved : {encoder_path}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s21",
      "cell_type": "code",
      "metadata": { "language": "sql", "resultVariableName": "stage_result", "title": "Create stage" },
      "source": "%%sql -r stage_result\n-- ── STEP 10 — Create a Snowflake internal stage ───────────────────────────────\nCREATE STAGE IF NOT EXISTS subscription_data.project_data.churn_model_stage\n    COMMENT = 'Storage for trained churn prediction model artefacts';",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s22",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Upload artifacts to stage" },
      "source": "# ── STEP 11 — Upload model and encoder to Snowflake stage ───────────────────\nfor local_path in [model_path, encoder_path]:\n    result = session.file.put(\n        local_path,\n        '@subscription_data.project_data.churn_model_stage',\n        overwrite=True,\n        auto_compress=False\n    )\n    print(f\"Uploaded: {local_path}  →  {result[0].target}\")\n\nprint(\"\\nBoth artefacts uploaded. The model is now available in:\")\nprint(\"  @subscription_data.project_data.churn_model_stage\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s23",
      "cell_type": "markdown",
      "metadata": { "codeCollapsed": false },
      "source": "# Part 2 — Batch Scoring (2023 Cohort)"
    },
    {
      "id": "s24",
      "cell_type": "code",
      "metadata": { "language": "sql", "resultVariableName": "scoring_cohort", "title": "Load 2023 scoring cohort" },
      "source": "%%sql -r scoring_cohort\n-- ── STEP 12 — Pull 2023 scoring cohort ────────────────────────────────────────\nSELECT\n    s.customer_id,\n    SUM(CASE WHEN YEAR(a.activity_date) = 2022 THEN a.num_sessions ELSE 0 END) AS total_sessions,\n    SUM(CASE WHEN YEAR(a.activity_date) = 2022 THEN a.total_session_length ELSE 0 END) AS total_session_duration,\n    COUNT(CASE WHEN YEAR(a.activity_date) = 2022 THEN 1 ELSE NULL END) AS active_days,\n    COUNT(DISTINCT CASE WHEN YEAR(a.activity_date) = 2022 THEN DATE_TRUNC('QUARTER', a.activity_date) END) AS count_active_quarters,\n    DIV0NULL(\n        SUM(CASE WHEN YEAR(a.activity_date) = 2022 THEN a.num_sessions ELSE 0 END),\n        NULLIF(COUNT(DISTINCT CASE WHEN YEAR(a.activity_date) = 2022 THEN DATE_TRUNC('QUARTER', a.activity_date) END), 0)\n    ) AS avg_sessions_per_quarter,\n    DATEDIFF('day',\n        MAX(CASE WHEN YEAR(a.activity_date) = 2022 THEN a.activity_date END),\n        '2023-01-01'\n    ) AS days_since_last_activity,\n    d.age,\n    d.income_level,\n    d.education,\n    d.device_type,\n    d.tech_comfort_score\nFROM subscription_data.project_data.ol_subscriptions s\nLEFT JOIN subscription_data.project_data.ol_customer_activity a\n    ON a.customer_id = s.customer_id\n    AND a.product = s.product\nLEFT JOIN subscription_data.project_data.ol_customer_demog d\n    ON d.customer_id = s.customer_id\nWHERE s.product = 'Healthy Meals'\n  AND '2023-01-01' >= s.subscription_start_date\n  AND '2023-01-01' < s.subscription_end_date\n  AND YEAR(s.subscription_end_date) = 2023\nGROUP BY s.customer_id, d.age, d.income_level, d.education, d.device_type, d.tech_comfort_score;",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s25",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Scoring cohort to pandas" },
      "source": "# ── STEP 13 — Load scoring cohort into pandas ─────────────────────────────────\nscore_df = scoring_cohort.to_pandas() if hasattr(scoring_cohort, 'to_pandas') else scoring_cohort.copy()\nprint(f\"Scoring cohort: {len(score_df):,} customers\")\nscore_df.head()",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s26",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Encode the scoring data" },
      "source": "# ── STEP 14 — Prepare scoring features using the saved preprocessing artifacts ─\n#\n# Reload from the pickled files rather than reusing the in-memory objects from\n# training. This mirrors how a real deployment/scoring job would run — as a\n# separate process or session with access only to the saved .pkl files, not\n# the training notebook's variables — and confirms the saved artifacts\n# round-trip correctly.\n\nwith open(model_path, 'rb') as f:\n    scoring_model = pickle.load(f)\n\nwith open(encoder_path, 'rb') as f:\n    prep = pickle.load(f)\n\nnum_cols = prep['numeric_cols']\ncat_cols = prep['categorical_cols']\n\nscore_num = prep['numeric_imputer'].transform(score_df[num_cols])\nscore_num = prep['scaler'].transform(score_num)\n\nscore_cat = prep['categorical_imputer'].transform(score_df[cat_cols])\nscore_cat = prep['encoder'].transform(score_cat)\n\nscore_features_final = np.hstack([score_num, score_cat])\n\nprint(f\"Scoring feature matrix shape: {score_features_final.shape}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s27",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Generate scores" },
      "source": "# ── STEP 15 — Generate renewal probability scores ────────────────────────────\nscores = scoring_model.predict_proba(score_features_final)[:, 1]\n\nprint(f\"Scores generated for {len(scores):,} customers.\")\nprint(f\"Score range: {scores.min():.3f} – {scores.max():.3f}\")",
      "outputs": [],
      "execution_count": null
    },
    {
      "id": "s28",
      "cell_type": "code",
      "metadata": { "language": "python", "title": "Results summary" },
      "source": "# ── STEP 16 — Attach scores and display results ──────────────────────────────\nscore_df['RENEWAL_PROBABILITY'] = scores\n\nprint(f\"Scored {len(score_df):,} customers.\")\nprint(f\"High churn risk (score < 0.5): {(score_df['RENEWAL_PROBABILITY'] < 0.5).sum():,}\")\nscore_df[['CUSTOMER_ID', 'RENEWAL_PROBABILITY']].sort_values('RENEWAL_PROBABILITY').head(10)",
      "outputs": [],
      "execution_count": null
    }
  ]
}
