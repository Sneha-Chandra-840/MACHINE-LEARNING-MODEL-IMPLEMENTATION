# MACHINE-LEARNING-MODEL-IMPLEMENTATION

"COMPANY" : CODTECH IT SOLUTIONS

"NAME" : SNEHA CHANDRA

"INTERN ID" : CT1MTDF345

"DOMAIN" : PYTHON PROGRAMING

"DURATION" : 4 WEEKS

"MENTOR" : NEELA SANTOSH KUMAR

"PROJECT DISCRIPTION" : Project Overview : In today’s digital age, mobile phones and messaging services are widely used to communicate. Along with useful messages, many people receive unwanted promotional messages or fraud links—these are known as spam messages. To prevent this, machine learning can help build a system that automatically detects whether a message is spam or not. This project does exactly that.

We use Python and the scikit-learn library to create a simple machine learning model that can classify SMS messages as either spam or ham (not spam). This is a text classification problem.

🔧 Tools and Technologies Used:
Python: Programming language

pandas: For reading and handling the dataset

scikit-learn:

CountVectorizer: For converting text to numbers

MultinomialNB: A Naive Bayes algorithm

train_test_split: To split data into training and testing sets

accuracy_score, classification_report, confusion_matrix: To check the model’s performance

matplotlib and seaborn: For data visualization (confusion matrix heatmap)

🧾 Dataset:
The dataset is taken from an online link and contains two columns:

label: Indicates if the message is "ham" (normal) or "spam"

message: The content of the SMS text message

This data is loaded using pandas from a .tsv (tab-separated values) file.

🔄 Step-by-Step Workflow:
✅ Step 1: Load the Dataset
The dataset is loaded from a URL using pandas.read_csv(). It contains SMS messages and their respective labels.

✅ Step 2: Preprocess the Data
Since machine learning models understand only numbers, we convert text labels:

"ham" is converted to 0

"spam" is converted to 1

This makes it easy for the model to understand.

✅ Step 3: Split the Data
Using train_test_split, we divide our dataset into:

Training set (80%) – Used to train the model

Testing set (20%) – Used to evaluate the model's performance

✅ Step 4: Convert Text into Numbers
Text cannot be directly used by machine learning models, so we use CountVectorizer to convert words into a bag-of-words model—a matrix of numbers that represent word counts.

✅ Step 5: Train the Model
We use Multinomial Naive Bayes (MultinomialNB), which works well with text data. The model is trained using the training data.

✅ Step 6: Make Predictions
The trained model predicts whether the test messages are spam or not. This is done using .predict().

✅ Step 7: Evaluate the Model
We calculate:

Accuracy – How many predictions were correct

Classification Report – Precision, recall, F1-score

Confusion Matrix – A 2x2 matrix that shows:

True positives (correctly predicted spam)

True negatives (correctly predicted ham)

False positives (ham wrongly predicted as spam)

False negatives (spam wrongly predicted as ham)

✅ Step 8: Visualize with Heatmap
We use seaborn and matplotlib to plot a heatmap of the confusion matrix, which gives a visual idea of the model's performance.

📈 Output:
You will see the accuracy score (e.g., 0.98 means 98% accurate).

A classification report showing details like precision and recall.

A confusion matrix heatmap as a graph showing prediction results.

🎯 Conclusion:
This project demonstrates how machine learning can be applied to solve real-world problems like spam detection. It uses simple text classification methods and a clean workflow to achieve high accuracy in spam detection. Even beginners can understand and implement this using Python.

The project not only introduces you to key ML concepts like preprocessing, vectorization, model training, and evaluation but also shows how to use visualization tools for performance reporting.

