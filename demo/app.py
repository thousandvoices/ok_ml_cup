import os
from flask import Flask, request, jsonify, send_from_directory, abort
from toxic_text_classifier.inference.classifier import Classifier


def create_app(model_path):
    os.environ['OMP_NUM_THREADS'] = '1'

    classifier = Classifier.load(model_path)
    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')

    @app.route('/toxic')
    def is_toxic():
        text = request.args.get('text')
        if text is None:
            abort(400, 'Text argument is required')

        result = {
            key: float(value.mean())
            for key, value
            in classifier.predict(text).items()
        }

        return jsonify(result)

    return app
