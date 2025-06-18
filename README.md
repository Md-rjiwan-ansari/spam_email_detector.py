#  Step 1: Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Step 2: Load the Dataset
# Dataset ka naam: spam.csv | Columns: v1 (label), v2 (message)
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Columns rename for clarity

#  Step 3: Label Encoding (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Optional: Data Visualization
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Spam vs Ham Distribution")
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'], rotation=0)
plt.show()

#  Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Step 5: Text Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#  Step 6: Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print(" Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

#  Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix ")
plt.show()

#  Step 9: Predict on New Message
sample_message = ["Congratulations! You've won a free ticket to Bahamas. Reply WIN now!"]
sample_vec = vectorizer.transform(sample_message)
result = model.predict(sample_vec)
print("\n Sample Message Prediction:", "Spam " if result[0] == 1 else "Ham")
