from __future__ import print_function, unicode_literals
import random
import os
import webbrowser
from flask import Flask, render_template, request, jsonify
import sys
from filt import FILTER_WORDS
from textblob import TextBlob
from predict import predict as p

app = Flask(__name__)

@app.route("/")

def hello():
    return render_template('chattemp.html')

@app.route("/ask", methods=['POST'])
def start_here():
        from firebase import firebase
        
        firebase = firebase.FirebaseApplication('https://cdata-2da9b.firebaseio.com', None)
        f = open('counter1.txt', 'r')
        data=f.read()
        f.close()
        f = open('counter2.txt', 'r')
        s_data=f.read()
        f.close()
        while True:
            
                message = str(request.form['messageText'])
                filt_resp=filter_response(message)
            
                if (filt_resp=='Unspeakable'):
                    resp="Unspeakable. Please type 'documentation' or 'docs'"
                    return jsonify({'status':'OK','answer':resp})
            
                elif check_for_exit(message):
                    return jsonify({'status':'OK','answer':'Goodbye sir, I hope you have a wonderful day.'})

                elif check_for_thanks(message):
                    return jsonify({'status':'OK','answer':'Most welcome. Always at your service, sir.'})

                elif check_for_greeting(message):
                    greeting=check_for_greeting(message)
                    return jsonify({'status':'OK','answer':greeting})

                elif check_for_documentation(message):
                    documentation=check_for_documentation(message)
                    return jsonify({'status':'OK','answer':documentation})

                elif check_for_sir(message):
                    sir_presence=check_for_sir(message)
                    return jsonify({'status':'OK','answer':sir_presence})
                
                elif check_for_comment_about_bot(message):
                    reply=check_for_comment_about_bot(message)
                    return jsonify({'status':'OK','answer':reply})

                elif 'schedule' in message or s_data=='1':    

                    if s_data=='0':
                        f.close()
                        f=open('counter2.txt','w')
                        f.write('1')
                        f.close()

                        for word in message:
                            print (word)

                            if integer_check(word):
                                f=open('sched_time.txt','w')
                                f.write(word)
                                f.close()        
                        return jsonify({'status':'OK','answer':'Please type your name:'})
        
                    if s_data=='1':
                        f=open('counter2.txt','w')
                        f.write('0')
                        f.close()
                        
                        f = open('sched_time.txt', 'r')
                        time_data=f.read()
                        f.close() 

                        from firebase import firebase

                        firebase = firebase.FirebaseApplication('https://cdata-2da9b.firebaseio.com', None)
                        
                        url='/'+message
                        result =firebase.get(url, None)

                        if(result): 
                            url='/'+message+'/schedule/'

                            data = {
                                time_data: "Meeting"
                            }
                            
                            result = firebase.post(url, data)
                        
                            return jsonify({'status':'OK','answer':'Schedule Updated!!!!!'})

                        else:
                            return jsonify({'status':'OK','answer':'Authentication Error!!! Please try again.'})  

    
                elif check_for_jokes(message):
                    joke=check_for_jokes(message)
                    return jsonify({'status':'OK','answer':joke})
    
                elif('project'in message) or (data=='1'):
                    if data=='0':
                        f.close()
                        f=open('counter1.txt','w')
                        f.write('1')
                        f.close()
                        return jsonify({'status':'OK','answer':'Please type your name:'})
    
                    if data=='1':
                        f=open('counter1.txt','w')
                        f.write('0')
                        f.close()
                        url='/'+message
                        result =firebase.get(url, None)
    
                        if(result):
                            url1=url
                            url1=url1+'/Project'
                            result_proj =firebase.get(url1, None)
                            a="Your Project is "+str(result_proj)
                            url2=url
                            url2=url+'/Tech'
                            result_tech =firebase.get(url2, None)
                            a=a+' and Assosiated Technologies are:    '
                            for key,value in result_tech.items():
                                a=a+"    Learn "+key+" from "+value+"         "   
                            return jsonify({'status':'OK','answer':a})

                        else:
                            f=open('counter1.txt','w')
                            f.write('0')
                            f.close()
                            return jsonify({'status':'OK','answer':'Invalid Login ID, Please try again or Contact the System administrator'})
                                             
                else:
                    resp= p(message)
                    filt_resp=filter_response(resp)
                    if (filt_resp=='Unspeakable'):
                        resp="Sorry, it appears that your query has resulted in an incoherent response. Please type 'documentation' or 'docs' to check my functionality or try different keywords."
                    return jsonify({'status':'OK','answer':resp})

@app.route("/ask/c")

def check_for_greeting(sentence):
    greeting_keywords = ["hello", "hi", "greetings","Are u alive?"]
    greeting_responses = ["Hello sir!","Hi sir!","Same to you sir!", "Welcome sir."]    
    if sentence in greeting_keywords and sentence!=greeting_keywords[3]:
        
        return (random.choice(greeting_responses)+' My name is V, your helpful bot. How can I help you today?')
    
    elif sentence==greeting_keywords[3]: return 'For you sir, always. My name is V, your helpful bot.'

def check_for_exit(sentence):
    if sentence in ['exit','bye','die','goodbye','Exit','Bye','Die','Good bye','Leave']:
        
        return 'Goodbye sir'

def check_for_thanks(sentence):
    if sentence in ['thanks','Thanks', 'appreciate it', 'Thank you','thank u']:
        
        return True


def check_for_documentation(sentence):
    if sentence in ['docs','Docs','Documentation','documentation']:
        documentation="Hello Sir; My functionality is.... Please type the next command"
        
        return documentation

def check_for_sir(sentence):
    if 'sir' in sentence:

        import pyrebase

        config = {
        "apiKey": "AIzaSyBbxqnz3BZ5j5pfD7c4hOVBSrAfr3ziT78",
        "authDomain": "nvidiahack-d0e6d.firebaseapp.com",
        "databaseURL": "https://nvidiahack-d0e6d.firebaseio.com",
        "storageBucket": "nvidiahack-d0e6d.appspot.com",
        "serviceAccount": "/home/swap/s2s/nvidiaHack-52580fff8e27.json"
        }

        firebase = pyrebase.initialize_app(config)
        auth = firebase.auth()

        db=firebase.database()
        user = auth.sign_in_with_email_and_password("nvidia@gmail.com", "nvidia")
        
        sir_data = db.child("sir_presence").get(user['idToken']).val()
        return(sir_data)

def check_for_comment_about_bot(sentence): 
    os.environ['NLTK_DATA'] = os.getcwd() + '/nltk_data'   
    from textblob import TextBlob
    parsed = TextBlob(sentence)
    for sent in parsed.sentences:
        adj = adjective(sent)

    if 'you' in sentence and (adj):
        print('a')
        resp="Thank you sir for the feedback! I will record that into my database"
        return resp

def adjective(sent):

    adj = None
    for w, p in sent.pos_tags:
        if p == 'JJ': 
            adj = w
            break
    return adj


def integer_check(s):

    try: 
        int(s)
        return True

    except ValueError:
        return False

def filter_response(resp):
    tokenized = resp.split(' ')

    for word in tokenized:

        for s in FILTER_WORDS:

            if word.lower().startswith(s):
                a="Unspeakable"
                return a

def check_for_jokes(sentence):
    import re

    if 'joke' in sentence:
        f = open('jokescount.txt', 'r')
        data=f.read()
        f.close()
        data=int(data)
        print (data)
    
        if data<2: 
            jokes_file_name = '/home/swap/s2s/jokes.txt'
            jokes=[]
            with open(jokes_file_name, 'r') as jokes_f:

                for line in jokes_f:
                    ln = line.strip()

                    if not ln or ln.startswith('#'):
                        continue
                    jokes.append(ln)
            content = random.choice(jokes)
            content = re.sub(r'_np_', '', content)
            f=open('jokescount.txt','w')
            
            data=int(data)
            data+=1
            data=str(data)
            f.write(data)
            f.close()
            
            return content
    
        else:
            content="Well, you know what, here is the best one yet: You know about the one time a manager fired his employee for too asking for too many jokes to the chatbot? I leave it up to you"
            f=open('jokescount.txt','w')
            f.write('0')
            f.close()
            return content

if __name__ == '__main__':
    
    app.run(debug=True)
    start_here()
