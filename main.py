import json
import pickle
import random
import nltk
import self
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
import numpy as np
import datetime
from PyDictionary import PyDictionary
import spacy
import requests


lemmatizer=WordNetLemmatizer
intents=json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot_model.h5')

api_key = "cd1a705e3a5f59b120ec6189e11f810e"

def get_weather(city_name):
    global temp
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)

    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]
    temp=response_dict["main"]["temp"]
    temp="{:.2f}".format(temp-273.15)

    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None

nlp=spacy.load("en_core_web_md")
#python -m spacy download en_core_web_md
#nlp=spacy.load("en_core_web_sm")
def chatbot(statement):
    weather=nlp("Current weather in city")
    statement=nlp(statement)
    min_similarity=0.4

    if weather.similarity(statement)>=min_similarity:
        for ent in statement.ents:
            if ent.label_ =="GPE":
                city=ent.text
                break
            else:
                res="You need to tell me a city to check"
                return res

        city_weather=get_weather(city)
        if city_weather is not None:
            res="In "+city+", the current weather is "+city_weather+ " and the temperature is "+str(temp)+" °C"
            return res
        else:
            res="something went wrong"
            return res
    else:
        res="sorry i don't understand that"
        return res

def yt(text):
    pass

def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(self,word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence,words,show_details=True):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1

    return(np.array(bag))

def predict_class(sentence,model):
    bow=bag_of_words(sentence,words,show_details=False)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r]for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result


dict=PyDictionary()

def meaning(text):
    return dict.meaning(str(text).split()[2])['Noun'][0]

todo=[]
def create_note():
    pass

def add_todo():
    pass

def show_todo():
    pass

def chatbot_res(text):
    intents_list = predict_class(text, model)
    if intents_list[0]['intent']=='weather':
        res=chatbot(text)
    elif 'search' in text:
        text=text.lower()
        indx = text.split().index("search")
        ind = text.split()[indx + 1:]
        url="https://www.google.com/search?client=firefox-b-d&q="+'+'.join(ind)
        res="searched "+text
    elif intents_list[0]['intent']=='meaning':
        res=meaning(text)
    elif 'time' in text:
        res=datetime.datetime.now().strftime("%H:%M:%S")
        return res
    elif intents_list[0]['intent']=='song':
        res=yt(text)
    elif intents_list[0]['intent']=='createnote':
        res=create_note()
    elif intents_list[0]['intent']=='addtodo':
        res=add_todo()
    elif intents_list[0]['intent']=='showtodo':
        res=show_todo()
    else:
        res = get_response(intents_list, intents)
    return res

def chat():
    print('Starting the Chatbot')
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            res = chatbot_res(inp)
            print(f"BOT: {res}")
            break
        else:
            res = chatbot_res(inp)
            print(f"BOT: {res}")

#chat()




