import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image


# Change Page Icon and Page Title
img = Image.open('titanic.jpg')
st.set_page_config(page_title='Titanic Survival Prediction', page_icon = img)

# Hide Setting Menu and Footer
hide_menu_style = """
		<style>
		#MainMenu {visibility: hidden; }
		footer {visibility: hidden; }
		<style>
		"""

st.markdown(hide_menu_style, unsafe_allow_html = True)



data = pd.read_csv('train.csv')

data['Sex'] = data['Sex'].map(lambda x:0 if x =='male' else 1)
data = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']]
data = data.dropna()

X = data.drop(['Survived'], axis=1)
y= data['Survived']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()
train_features = sc.fit_transform(X_train)
test_features = sc.transform(X_test)


model = LogisticRegression()
model.fit(train_features, y_train)

train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_pred = model.predict(test_features)

confusion = metrics.confusion_matrix(y_test, y_pred)
FN = confusion[1][0]
TN = confusion[0][0]
FP = confusion[0][1]
TP = confusion[1][1]
metrics.classification_report(y_test, y_pred)

#ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

#calculate AUC

auc = metrics.roc_auc_score(y_test, y_pred)


st.title("Data Science")
st.write("## Titanic Survival Prediction Project ")

menu = ["Overview", "Build Project", "New Prediction"]

choise = st.sidebar.selectbox('Menu', menu)


if choise == "Overview":
	st.subheader("Overview")
	st.write("""
	#### The data has been split into two groups:
	- training set (train.csv): 
	The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
	- test set (test.csv):
	The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
	- gender_submission.csv: a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
	""")

elif choise == "Build Project":
	st.subheader("Build Project")
	st.write("#### Data Preprocessing")
	st.write("##### Show Data")
	st.table(data.head(5))

	st.write("#### Build model and evaluaton: ")
	st.write("Train Set Score: {}".format(round(train_score, 2)))
	st.write("Test Set Score: {}".format(round(test_score, 2)))
	st.write("Confusion Matrix:")
	st.table(confusion)
	st.write(metrics.classification_report(y_test, y_pred))
	st.write("##### AUC: %.3f" % auc)

	st.write("#### Visualization")
	fig, ax = plt.subplots()
	ax.bar(['False Negative', 'True Negative', 'False Positive', 'True Positive'],
		[FN, TN, FP, TP])
	st.pyplot(fig)

	st.write("ROC curve")
	fig1, ax1 = plt.subplots()
	ax1.plot([0, 1], [0, 1], linestyle='--')
	ax1.plot(fpr, tpr, marker='.')
	ax1.set_title("ROC Curve")
	ax1.set_xlabel("False Positive Rate")
	ax1.set_ylabel("True Possitive Rate")
	st.pyplot(fig1)

elif choise == 'New Prediction':
	st.subheader("Make new Prediction")
	st.write("##### Input/Select data")
	name = st.text_input("Name of Passenger")
	sex = st.selectbox("Sex", options=["Male", "Female"])
	age = st.slider("Age", 1, 100, 1)
	Pclass = np.sort(data['Pclass'].unique())

	pclass = st.selectbox("PClass", options = Pclass)
	max_sibsp = max(data["SibSp"])
	sibsp = st.slider("Siblings", 0, max_sibsp, 1)
	max_parch = max(data["Parch"])
	parch = st.slider("Parch", 0, max_parch, 1)	
	max_fare = round(max(data["Fare"]) +10,2)
	fare = st.slider("Fare", 0.0, max_fare, 0.1)

	sex = 0 if sex == 'Male' else 1

	if st.button('Submit'):
		new_data = sc.transform([[sex, age, pclass, sibsp, parch, fare]])
		prediction = model.predict(new_data)
		predict_probability = model.predict_proba(new_data)

		if prediction[0] == 1:
			st.subheader("Passenger {} would have survived with a probability of {}%".format(name, round(predict_probability[0][1]*100, 2)))
		else:
			st.subheader("Passenger {} would have survived with a probability of {}%".format(name, round(predict_probability[0][0]*100, 2)))


