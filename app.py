import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import base64
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))


st.markdown("""
<style>
.big-font {
    text-align: center;
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

def main():
	"""Fake News Classifier"""
	st.markdown('<div class="big-font"><b>Fake News Classifier</b></div>', unsafe_allow_html=True)
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Machine Learning Web App </h1>
	</div>

	"""
	st.markdown(html_temp,unsafe_allow_html=True)

main()
st.info("Prediction with Machine Learning")

all_ml_models = ["Logistic Regression","Random Forest","Support Vector Machine(SVM)","Decision Tree Classifier"]
model_choice = st.selectbox("Select Your Model",all_ml_models)

if model_choice == 'Logistic Regression':
    load_model = pickle.load(open('lr.pkl', 'rb'))

elif model_choice == 'Random Forest':
    load_model = pickle.load(open('rfc.pkl', 'rb'))
        
elif model_choice == 'Support Vector Machine(SVM)':
	load_model = pickle.load(open('svm.pkl', 'rb'))

elif model_choice == 'Decision Tree Classifier':
	load_model = pickle.load(open('dtc.pkl', 'rb'))
                        
#Making The Function For Detecting News From Input
def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

#Stemming The Given Text Data
def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con



if __name__ == '__main__':
    # st.title('Fake News Classification App ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here","Please Type Here",height=200)
    predict_btt = st.button("Predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Real News')
        if prediction_class == [1]:
            st.warning('Fake News')