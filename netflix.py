
import pandas as pd
import missingno as lost                                   # ek Python library hai jo missing values (NaN) ko visualize karti hai.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data= pd.read_csv(r"G:\\netflix-ml-project-main\\netflix_titles.csv")

# EXPLORATORY DATA ANALAYSIS

print(data.head(10))
print(f"\n****Tail****\n{data.tail(10)}")
print(f"\n*****8*\n{data.isnull()}")
print(f"\n***Shape***\n{data.shape}")
print(f"\n***Info***\n{data.info}")

print(f"\n***Numerical Summary***\n{data.describe()}")
print(f"\n***Most Occured value***\n{data.rating.value_counts() }")   #displaying the most occurred rating
top10_country =data['country'].value_counts().head(10)


data['rating'] = data['rating'].fillna("TV-MA")
print("\n***Filled missing ratings***")
print(data['rating'].head(10))

# **Checking missing values**
data.isnull().sum()
data.columns

# Duplicate columns
data.duplicated().sum()
np.int64(0)
data["listed_in"]


fig=px.histogram(data,
                 x= "title",
                 color="rating",#Change 'Location' to 'locations'
                 hover_data=data.columns,
                 title="Show's Title And Their Rating",
                 barmode="group",
                 )
fig.show()

director_film = data[data["type"] == "Movie"]["director"].value_counts()[:10]

plt.figure(figsize = (10,8))
sns.barplot(x = director_film.index, y = director_film.values)
plt.xticks(rotation = 45, fontsize = 9)
plt.title("Top 10 Directors with the Most Films",fontsize = 18)
plt.show()
plt.figure(figsize = (9,8))
sns.barplot(x = top10_country.index, y = top10_country.values)
plt.title("Top 10 Content Producing Countries", fontsize = 22)
plt.xticks(rotation = 45, fontsize = 14)
plt.show()

x = data['type'].value_counts()
labels = x.index  # automatically ['Movie', 'TV Show']
plt.figure(figsize=(8,8))
plt.pie(x.values, labels=labels, autopct="%1.1f%%", colors=['skyblue','lightgreen'], startangle=90)
plt.title("Distribution of Movies vs TV Shows on Netflix", fontsize=16)
plt.show()


# Data Visualization
plt.figure(figsize=(5,5))
lost.bar(data, color='Pink')
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(data.isnull(), cmap=sns.color_palette(["green","red"]), cbar=False)
plt.title("Missing Values in Dataset (Red = Missing, Green = Present)")
plt.show()

print("🚥 "*20)

# Machine Learning Algorithm

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

y = data['release_year'] # target predication
X = data.drop(['show_id',], axis=1) #traning
print(x,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# K-nearest neighbour

# Import necessary libraries

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('netflix_titles.csv')

# Prepare the data for modeling
y = data['release_year']  # Target variable
X = data.drop(['show_id'], axis=1)  # Features

# Identify categorical columns for encoding
categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']

# Create a LabelEncoder object
le = LabelEncoder()

# Apply Label Encoding to categorical columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Now perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Initialize the K-Nearest Neighbors model
modelKNN = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
modelKNN.fit(X_train, y_train)

# Make predictions on the test data
y_pred = modelKNN.predict(X_test)

# Calculate accuracy
accuracyKNN = accuracy_score(y_test, y_pred)

# Print the accuracy
print(accuracyKNN)
# Decision tree  
print("🌲"*20)
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('netflix_titles.csv')

# Prepare the data for modeling
y = data['release_year']  # Target variable
X = data.drop(['show_id'], axis=1)  # Features

# Identify categorical columns for encoding
categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']

# Create a LabelEncoder object
le = LabelEncoder()

# Apply Label Encoding to categorical columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Now perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Initialize the Decision Tree model
modelDT = DecisionTreeClassifier(random_state=7)

# Fit the model to the training data
modelDT.fit(X_train, y_train)

# Make predictions on the test data
y_pred = modelDT.predict(X_test)

# Calculate accuracy
accuracyDT = accuracy_score(y_test, y_pred)

# Print the accuracy
print(accuracyDT)


# naive based
print("🎇🎇"*19)
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('netflix_titles.csv')

# Prepare the data for modeling
y = data['release_year']  # Target variable
X = data.drop(['show_id'], axis=1)  # Features

# Identify categorical columns for encoding
categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']

# Create a LabelEncoder object
le = LabelEncoder()

# Apply Label Encoding to categorical columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Now perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Initialize the Naive Bayes model (Multinomial Naive Bayes)
modelNB = MultinomialNB()

# Fit the model to the training data
modelNB.fit(X_train, y_train)

# Make predictions on the test data
y_pred = modelNB.predict(X_test)

# Calculate accuracy
accuracyNB = accuracy_score(y_test, y_pred)

# Print the accuracy
print(accuracyNB)

# Random forest
print("🍀🌿🌿🌾🌾"*18)
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv('netflix_titles.csv')  # Replace with your dataset path

# Prepare the data for modeling
y = data['release_year']  # Target variable
X = data.drop(['show_id', 'release_year'], axis=1)  # Features: Removing 'show_id' and 'release_year'

# Identify categorical columns for encoding
categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']

# Creating a LabelEncoder object to convert categorical columns into numeric format
le = LabelEncoder()

# Applying Label Encoding to categorical columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))  # Convert categorical values to numeric
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

# Initialize the Random Forest Regressor
modelRF = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
modelRF.fit(X_train, y_train)

# Making predictions using the trained model on the test data
y_pred = modelRF.predict(X_test)


# Visualizing the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Release Year')
plt.ylabel('Predicted Release Year')
plt.title('Actual vs Predicted Release Year')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line

plt.show()




