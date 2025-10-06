#Project 1: AER850
#Name: Yanni Z. Martinez S. 

#Libs import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score
from sklearn.ensemble import StackingClassifier

# ---------------- Step 1: Read file and convert into DataFrame ----------------

df = pd.read_csv("Project 1 Data.csv") #Reading the csv file and printing info for dataframes
print(df.info())
print(df.columns) 

# ---------------- Step 2: Statistical analysis with graphs ----------------

# Calculate mean of each set by Step
step_means_X = df.groupby("Step")["X"].mean()
step_means_Y = df.groupby("Step")["Y"].mean()
step_means_Z = df.groupby("Step")["Z"].mean()

# *** Create subplots ***
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) #Figure with 1 row, 3 columns

# ** Bar plots **

# For X values: index = x axis position and values is the height
axes[0].bar(step_means_X.index, step_means_X.values, color='orange', edgecolor='black')
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Average X")
axes[0].set_title("X by Step")

# For Y values:
axes[1].bar(step_means_Y.index, step_means_Y.values, color='salmon', edgecolor='black')
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Average Y")
axes[1].set_title("Y by Step")
         
# For Z values:
axes[2].bar(step_means_Z.index, step_means_Z.values, color='brown', edgecolor='black')
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Average Z")
axes[2].set_title("Z by Step")

plt.tight_layout() # To adjust spacing
plt.show()

# ** Boxplots **

steps = df["Step"].unique() # Returns a Numpy array

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(["X", "Y", "Z"]): #Gives an index (i) and col name
    data = [df[df["Step"] == step][col] for step in steps]
    axes[i].boxplot(data, labels=steps)
    axes[i].set_title(f"{col} by Step")
    axes[i].set_xlabel("Step")
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()

# ** Scatter plot **

# Get steps and colors palette
steps = df["Step"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(steps)))

# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each step with a different color
for i, step in enumerate(steps):
    df_step = df[df["Step"] == step] #Focuses on each individual step
    ax.scatter(df_step["X"], df_step["Y"], df_step["Z"], 
    color=colors[i], label=f"Step {step}", alpha=0.7, s=30) #alpha for transparency and s for size dots

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot of X, Y, Z by Step")

#For legend of steps and representative colors
ax.legend(title="Step", bbox_to_anchor=(1.05, 1), loc='upper left') 

plt.tight_layout()
plt.show()

# ---------------- Step 3: Correlation matrix ----------------

corr_matrix = df.corr(method="pearson")
print("Correlation matrix:\n", corr_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Pearson Correlation")
plt.show()

# ---------------- Step 4: Model development ---------------- 

# ***** Prepare features and target *****
X = df[["X", "Y", "Z"]]
y = df["Step"]

# Now, split dataset into TRAINING and TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------ Grid search cross-validation Models ------------

# *** 1. Logistic Regression ***
log_params = {
    "C": [0.01, 0.1, 1, 10], # C=small -> strong regularization and vice versa (Test all)
    "solver": ["liblinear", "lbfgs"], #liblinear for small and lbfgs for large datasets
    "penalty": ["l2"] #To prevent overfitting (Called Ridge: coefficients small BUT not 0)
}
log_grid = GridSearchCV(LogisticRegression(max_iter=1000), log_params, cv=3) #3 cross validation
log_grid.fit(X_train_scaled, y_train)
print("\n====== Best Logistic Regression parameters: ======")
print(classification_report(y_test, log_grid.predict(X_test_scaled)))

# *** 2. Random Forest ***
rf_params = {
    "n_estimators": [50, 100, 150], #More trees, more accurate but slower
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
print("\n====== Best Random Forest parameteres: ======")
print(classification_report(y_test, rf_grid.predict(X_test)))

# *** 3. Support Vector Machine ***
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
svm_grid = GridSearchCV(SVC(), svm_params, cv=3)
svm_grid.fit(X_train_scaled, y_train)
print("\n====== Best SVM parameteres: ======")
print(classification_report(y_test, svm_grid.predict(X_test_scaled)))

# *** 4. RandomizedSearchCV Model ***
gb_params = {
    "n_estimators": randint(50, 200),
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(2, 10)
}

gb_random = RandomizedSearchCV(GradientBoostingClassifier(), gb_params, n_iter=20,
    cv=5,random_state=42)

gb_random.fit(X_train, y_train)
print("\n ====== Best RandomizedSearchCV parameteres: ======")
print(classification_report(y_test, gb_random.predict(X_test)))

# ------------ Step 5: Model performance analysis ------------
# *** NOTE: PLEASE GIVE 1-2 MINUTES FOR THE CONFUSION MATRICES, PROCESSING THE BEST MODEL ***

# ***** Collect models and names *****
models = {
    "Logistic Regression": (log_grid, X_test_scaled),
    "Random Forest": (rf_grid, X_test),
    "SVM": (svm_grid, X_test_scaled),
    "Gradient Boosting (Randomized)": (gb_random, X_test)
}

# Define dictionary of models and test data
models = {
    "Logistic Regression": (log_grid, X_test_scaled),
    "Random Forest": (rf_grid, X_test),
    "SVM": (svm_grid, X_test_scaled),
    "Gradient Boosting (Randomized)": (gb_random, X_test)
}

# Evaluate models and plot confusion matrices
for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# ------------ Step 6: Stacked Model Performance Analysis ------------

# Define models to stack
estimators = [
    ('rf', rf_grid.best_estimator_),      # Random Forest
    ('logit', log_grid.best_estimator_)  # Logistic Regression
]

# Final estimator
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

# Train stacked model and prediction on test
stack_model.fit(X_train_scaled, y_train)
y_pred_stack = stack_model.predict(X_test_scaled)

# Evaluate performance
acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack, average='weighted', zero_division=0)
f1_stack = f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)

print("\n====== STACKED MODEL PERFORMANCE ======")
print(f"Accuracy:  {acc_stack:.3f}")
print(f"Precision: {prec_stack:.3f}")
print(f"F1 Score:  {f1_stack:.3f}")

# Confusion matrix
cm_stack = confusion_matrix(y_test, y_pred_stack)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_stack)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix — Stacked Model (RF + Logistic Regression)")
plt.show()

# ------------ Step 7: model evaluation ------------

from joblib import dump, load

#Save the trained stacking model
dump(stack_model, "stacked_model.joblib")

#Load the model
loaded_model = load("stacked_model.joblib")

#Random coordinate sets to predict maintenance step
new_data = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

#Apply same scaling used for training
new_data_scaled = scaler.transform(new_data)

#Predict the maintenance step
predicted_steps = loaded_model.predict(new_data_scaled)

#Display predictions
print("\n ====== Predicted maintenance steps for randomn coordinates: ======")
for coords, step in zip(new_data, predicted_steps):
    print(f"Coordinates {coords} → Predicted Step: {step}")
