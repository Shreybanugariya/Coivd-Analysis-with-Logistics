from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

#open file
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
  if request.method == "POST":
    myDict = request.form
    Age = int(myDict['age'])
    Fever = int(myDict['Fever'])
    Pain = int(myDict['bodyPain'])
    Cold = int(myDict['Cold'])
    #code for interference
    inputFeatures = [Age, Fever, Pain, Cold]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    print(infProb)
    #return render_template('show.html', inf=infProb)
  return render_template('index.html', inf=infProb)
  #return 'Hello, World!' +str(infProb)


if __name__ == "__main__":   
    app.run(debug=True)