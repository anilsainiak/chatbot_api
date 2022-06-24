from flask import Flask,request,jsonify
from main import chatbot_res

app=Flask(__name__)

@app.route('/chat',methods=['GET','POST'])

def chatbot():
    chatInput=request.form['chatInput']
    return jsonify(chatBotReply=chatbot_res(chatInput))

if __name__=='__main__':
    app.run( debug=True)