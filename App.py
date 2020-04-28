"""
This is the final project for Data Science. This is crime classification
app with kivy. The program uses a dataset, in our case of crime and
a model. The app contains a text field in which the user enters a "narrative"
or crime description. The description goes through the model and classifies it
as a specific crime. This app also takes voice input. The user can speak into the
app and have the crime be predicted. The app is not pigeonholed to just crime
classification. By inputting another dataset such as a movie dataset, the app
could predict something like whether a movie is good or not. Or maybe find the
author of a book based on some descriptions.
"""


import pandas as pd
df = pd.read_csv('train_comp2.csv')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import speech_recognition as sr
r =sr.Recognizer()

X_train, X_test, y_train, y_test = train_test_split(df['NARRATIVE'], df['CRIMETYPE'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
import pyaudio
class textClassification(App):
    def build(self):
        from kivy.core.window import Window
        #Window.clearcolor = (1,0,0,1)
        im = Image(source='img.JPG')
        im.reload()                        #try to add image       
        layout = BoxLayout(padding=10, orientation='vertical')
        label1=Label(text ="NARRATIVE",bold=True,color=(1,0,0,1))
        label1.font_size='50dp'
        label2 = Label(text="")
        layout.add_widget(label1)
        self.txt1 = TextInput(text='', multiline=False)
        layout.add_widget(self.txt1)
        btn1 = Button(text="CLASSIFY" ,font_size=34,size=(300, 100), size_hint=(None, None),
                      pos_hint={'center_x': 0.5, 'center_y': 0.5},
                      background_color= (1.0, 0.0, 0.0, 1.0), background_normal= 'kivy.png',border=(30,30,30,30))

        btn12 = Button(text="", font_size=34, size=(100, 100), size_hint=(None, None),
                      pos_hint={'center_x': 0.5, 'center_y': 0.0},
                       background_normal='mic2.png')
        btn1.bind(on_press=self.buttonClicked)
        btn12.bind(on_press=self.buttonClicked2)
        layout.add_widget(btn1)
        layout.add_widget(label2)
        layout.add_widget(btn12)
        self.lbl1 = Label(text="Crime Type",bold=True,color=(6,80,204,1))
        self.lbl1.font_size='50dp'
        layout.add_widget(self.lbl1)
        return layout

    def buttonClicked(self,btn):
        name =self.txt1.text
        if name !="":
            pred= clf.predict(count_vect.transform([name]))
            #self.txt1="ffffffff"
            self.lbl1.text = pred[0]
        else:
            self.lbl1.text="Crime Type"

    def buttonClicked2(self,btn):

        with sr.Microphone() as source:
            print("say something")
            audio = r.listen(source)
            print("time is over")

        try:
            output = r.recognize_google(audio);
            print("Text:",output);
            self.txt1.text=output;
            #name = ourput;
            #pred = clf.predict(count_vect.transform([name]));
            #self.lbl1.text = pred[0];

            #print(output);
        except:
            pass;

if __name__=="__main__":
    textClassification().run()


