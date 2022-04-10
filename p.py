from flask import *
app = Flask(__name__)  
 
@app.route('/')  
def home():  
    return render_template("fis.html");
@app.route('/')  
def home():  
      return render_template("output.html",name="Raja");
app.run()