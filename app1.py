
from flask import Flask,render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def ValuePredictor(news): 
	m=pickle.load(open('model.pkl','rb'))
	vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2),vocabulary = pickle.load(open('tf1.pkl','rb')))
	news = vectorizer.fit_transform(news)
	p=m.predict(news)
	return p

app=Flask(__name__) 

@app.route('/',methods=['GET'])
def index():
	return render_template('a.html')
@app.route('/result', methods = ['GET','POST']) 
def result(): 
	if request.method == 'POST': 
		to_predict_list = request.form.to_dict() 
		news = list(to_predict_list.values()) 
		result = ValuePredictor(news)         
		if result ==0:
			pred='Business'
		if result==1:
			pred='Sport'
		if result==2:
			pred='Technology'            
	return render_template('result.html', prediction = pred) 

