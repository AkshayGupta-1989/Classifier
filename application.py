import random
from flask import Flask, jsonify, request, render_template
from flasgger import Swagger
from api.classify import classify

application = app = Flask(__name__)
Swagger(application)

@app.route('/classify', methods=['GET'])
def index():
    """
    This is the Document Classification API
    Call this api passing a document and get back its class
    ---
    tags:
      - Document Classification API
    parameters:
     - name: words
       in: query
       type: string
       required: true
       description: Document words
    produces: [
        "application/json",
        "text/html"
    ]

    responses:
      200:
        description: A class with its confidence
        schema:
          id: classification
          properties:
            class:
              type: string
              description: Document class
              default: POLICY CHANGE
            confidence:
              type: string
              description: Class confidence
              default: 74%

    """
    query = request.args['words']
    if len(query) == 0:
        return "An error occurred, enter words", 500
    else:
        classification, confidence = classify(query)
        result = 'class: '+str(classification)+', classification: '+str(confidence)

        accept = (request.headers.get('accept'))

        if (accept == "text/html"):
            return result
        else:
            return jsonify({
                'class': classification,
                'confidence': confidence
            })



if __name__ == '__main__':
    application.run()
