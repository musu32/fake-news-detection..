import pandas as pd

df = pd.read_csv("titanic.csv")
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

df.head()

inputs = df.drop('Survived',axis='columns')
target = df.Survived

inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})

inputs.Age[:10]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

inputs.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

len(X_train)

len(X_test)

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train)

#model.score(X_test,y_test)

accuracy = model.score(X_test,y_test)
print("Accuracy:", accuracy)

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(model, out_file=None, 
                           feature_names=inputs.columns.values.tolist(),
                           class_names=['Died', 'Survived'], 
                           filled=True, rounded=True, 
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("titanic_decision_tree") # Save tree as PDF file
