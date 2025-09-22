<h1>ML-Model-for-Stroke-Prediction</h1>

<p>
This repository contains end-to-end machine learning work across three strands:
  
(1) <strong>Unsupervised Learning</strong> with K-Means on a synthetic dataset;

(2) <strong>Dimensionality Reduction</strong> with PCA on housing data; and

(3) a full <strong>Binary Classification</strong> workflow for stroke-risk prediction, including an
operational <strong>Healthcare AI Agent</strong> that preprocesses new patient records and outputs risk categories.
</p>

<hr>

<h2>Repository Contents</h2>
<ul>
  <li><code>Unsupervised Learning.ipynb</code> — K-Means, elbow and silhouette analysis on <code>cluster1.csv</code>.</li>
  <li><code>Dimensionality Reduction.ipynb</code> — PCA analysis on <code>kc_house_data.csv</code>.</li>
  <li><code>Stroke Prediction.ipynb</code> — model training, CV and test evaluation for stroke prediction.</li>
  <li><code>Stroke Risk Agent.ipynb</code> — agent that automates preprocessing and risk scoring.</li>
  <li><code>Dataset_raw.csv</code> — raw clinical dataset (620×10, cleaned to 596×7 after preprocessing).</li>
  <li><code>cluster1.csv</code> — 600×2 synthetic clustering dataset.</li>
  <li><code>kc_house_data.csv</code> — 21,613×19 housing dataset for PCA experiments.</li>
</ul>

<hr>

<h2>Environment & Setup</h2>
<pre><code>python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
</code></pre>

<p><strong>Run notebooks</strong> in Jupyter or VS Code:</p>
<pre><code># If needed:
pip install jupyter
jupyter notebook
</code></pre>

<hr>

<h2>1) Sub-task 1 — Unsupervised Learning (K-Means on <code>cluster1.csv</code>)</h2>

<h3>Approach</h3>
<ul>
  <li>K-Means clustering with <code>k = 3</code> and <code>k = 10</code>.</li>
  <li>Evaluation via within-cluster sum of squares (SSD/inertia), Elbow Method (<code>k = 1..10</code>), and Silhouette Analysis (with <code>StandardScaler</code>).</li>
</ul>

<h3>Key Results</h3>
<ul>
  <li>SSD: <code>k=3</code> → 279.2723; <code>k=10</code> → 78.4356 (lower SSD at larger <em>k</em> is expected and not sufficient on its own).</li>
  <li>Silhouette (scaled): <code>k=3</code> → 0.5957; <code>k=10</code> → 0.6723.</li>
  <li>Elbow Method indicates a clear elbow at <strong>k=3</strong>.</li>
</ul>

<h3>Conclusion</h3>
<p>
Although <code>k=10</code> reduces SSD and has a higher average silhouette value, qualitative inspection and the elbow curve
show over-fragmentation at high <em>k</em>. <strong>k=3</strong> provides a better trade-off between simplicity, interpretability and generalisability.
A 240-word written justification is included in the notebook.
</p>

<hr>

<h2>2) Sub-task 2 — Dimensionality Reduction (PCA on <code>kc_house_data.csv</code>)</h2>

<h3>Preprocessing</h3>
<ul>
  <li>Target-encode <code>zipcode</code> to a numeric <code>zipcode_encoded</code> using mean price per ZIP.</li>
  <li>Drop <code>price</code> and <code>zipcode</code> to run unsupervised PCA on input features only.</li>
  <li>Standardise features with <code>StandardScaler</code>.</li>
</ul>

<h3>Findings</h3>
<ul>
  <li><strong>3 components</strong>: variance ≈ 0.512; reconstruction MSE ≈ 0.4882; compression ≈ 83.33%.</li>
  <li><strong>14 components</strong> (≈95% variance): variance ≈ 0.9634; reconstruction MSE ≈ 0.0366; compression ≈ 22.22%.</li>
</ul>

<h3>Conclusion</h3>
<p>
Three components are helpful for visualisation but retain only ~51% of variance and incur higher reconstruction error.
Using <strong>14 principal components</strong> retains ~95% variance and substantially lowers reconstruction error, giving a better balance
between compression and information preservation. A 255-word justification is included in the notebook.
</p>

<hr>

<h2>3) Stroke Risk Prediction — Binary Classification</h2>

<h3>Data & Preprocessing</h3>
<ul>
  <li>Clean: drop rows with NA (620×10 → 596×10), then select <code>TOTCHOL, AGE, SYSBP, DIABP, CIGPDAY, BMI, STROKE</code>.</li>
  <li>Map <code>STROKE</code> to binary (1→1, 2→0). Balanced classes (~50/50).</li>
  <li>Clinical binning rules:
    <ul>
      <li><code>TOTCHOL</code>: Normal / Borderline High / High</li>
      <li><code>SYSBP</code>, <code>DIABP</code>: Normal / Elevated / Hypertension</li>
      <li><code>BMI</code>: Normal Weight / Overweight / Obese</li>
    </ul>
    Encoded with <code>LabelEncoder</code>.
  </li>
  <li>Hold-out test split: 20% with stratification. Cross-validation: Stratified 5-fold.</li>
</ul>

<h3>Models Evaluated</h3>
<ul>
  <li>Logistic Regression</li>
  <li>Support Vector Machine</li>
  <li>K-Nearest Neighbours</li>
  <li>Random Forest</li>
  <li><strong>Gradient Boosting</strong> (chosen)</li>
</ul>

<h3>Headline Test Metrics</h3>
<ul>
  <li>Logistic Regression — AUC 0.8207; Acc 0.7917; Prec 0.8889; Rec 0.6667; F1 0.7619.</li>
  <li>SVM — AUC 0.7385; Acc 0.6083; Prec 0.9333; Rec 0.2333; F1 0.3733.</li>
  <li>KNN — AUC 0.8121; Acc 0.7583; Prec 0.7541; Rec 0.7667; F1 0.7603.</li>
  <li>Random Forest — AUC 0.9178; Acc 0.8167; Prec 0.8065; Rec 0.8333; F1 0.8197.</li>
  <li><strong>Gradient Boosting</strong> — AUC 0.9378; Acc 0.8750; Prec 0.8947; Rec 0.8500; F1 0.8718.</li>
</ul>

<h3>Chosen Model & Tuning</h3>
<ul>
  <li><strong>Gradient Boosting</strong> selected for best overall trade-off (AUC, accuracy, precision, recall, F1).</li>
  <li>Best hyperparameters (GridSearchCV):
    <code>{learning_rate: 0.05, max_depth: 3, min_samples_leaf: 1, min_samples_split: 5, n_estimators: 50, subsample: 0.6}</code>.
  </li>
  <li>With best params — Test: AUC 0.9328; Acc 0.85; Prec 0.8889; Rec 0.80; F1 0.8421.</li>
  <li>Feature importance (example run): DIABP, CIGPDAY, SYSBP, TOTCHOL, AGE, BMI.</li>
</ul>

<hr>

<h2>Healthcare AI Agent</h2>

<p>
An operational agent that:
(1) accepts a CSV of new patients;
(2) applies the same binning and encodings as training; and
(3) outputs a probability, binary prediction and a risk category (Low / Medium / High).
</p>

<h3>Model & Thresholds</h3>
<ul>
  <li>Classifier: <code>GradientBoostingClassifier(learning_rate=0.05, max_depth=3, min_samples_leaf=1, min_samples_split=5, n_estimators=50, subsample=0.6, random_state=42)</code>.</li>
  <li>Risk categories from predicted probability:
    <ul>
      <li>&lt; 0.30 → Low</li>
      <li>0.30–0.70 → Medium</li>
      <li>&ge; 0.70 → High</li>
    </ul>
  </li>
  <li>Test performance: AUC 0.9328; Acc 0.85; Prec 0.8889; Rec 0.80; F1 0.8421.</li>
</ul>

<h3>Programmatic Use</h3>

<pre><code class="language-python">
# After training the GB model and fitting label encoders:
agent = StrokeRiskAgent(gb_model, label_encoders)

# Predict on a new CSV of patients (with TOTCHOL, AGE, SYSBP, DIABP, CIGPDAY, BMI):
new_patients = pd.read_csv("new_patients.csv")

predictions = agent.predict(new_patients)

agent.export_results(predictions, filename="risk_report.csv")
</code></pre>

<hr>

<h2>Reproducibility Notes</h2>
<ul>
  <li>Random seeds (e.g., <code>random_state=42</code>) are set where applicable, but PCA and some algorithms may still show slight variation.</li>
  <li>Cross-validation uses stratification to preserve class balance across folds.</li>
</ul>

<hr>

<h2>References</h2>
<ul>
  <li>Hutter, F. et al. Automated Machine Learning. Springer, 2019.</li>
  <li>Dong, X. et al. A Survey on Ensemble Learning. Frontiers of Computer Science, 2019.</li>
  <li>He, H., Garcia, E. Learning from Imbalanced Data. IEEE TKDE, 2009.</li>
  <li>Seto, H. et al. Scientific Reports, 2022.</li>
  <li>Molnar, C. Interpretable Machine Learning, 2022.</li>
  <li>NannyML Documentation — Data Reconstruction with PCA.</li>
  <li>IBM SPSS — Categorical PCA notes.</li>
  <li>Built In — Step-by-step PCA explanation.</li>
  <li>Tan, P.-N., Steinbach, M., Pearson, V. Introduction to Data Mining.</li>
</ul>
