import os
from spyre import server
import liveprediction

class SimpleApp(server.App):
    title = "Classifying Twitter users in Pakistan"
    
    inputs = [{ "type" : "text",
                "key" : "words",
                "label" : "Enter Twitter handle",
                "value" : ""}
              ]
    
    outputs = [{"type" : "html",
                "id" : "some_html",
                "control_id" : "button1"}]
    
    controls = [{"type" : "button",
                 "label" : "classify",
                 "id" : "button1"}]

    def getHTML(self, params):
        words = params['words']
        if words == '':
            return 'Enter Twitter handle and press classify to run'
        else:
            Result=liveprediction.get_prediction(words)
            Result=" <br> ".join(str(x) for x in Result)
            return "Twitter user classified as: "+words+"<br>"+Result

app = SimpleApp()
app.launch(host='127.0.0.1', port=int(os.environ.get('PORT', '8080')))
