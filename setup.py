import setuptools


setuptools.setup(
    name='toxic_text_classifier',
    version='0.0.1',
    description='Simple toxicity detection tool for russian',
    packages=[
        'toxic_text_classifier',
        'toxic_text_classifier.common',
        'toxic_text_classifier.inference'
    ],
    license='MIT',
    extras_require={
        'bert': ['numpy>=1.19.2', 'transformers>=4.3.3', 'onnxruntime>=1.6.0', 'onnx>=1.8.0'],
        'logistic': ['numpy>=1.19.2', 'scikit-learn>=0.23.2', 'textvec>=2.0'],
        'dev': [
            'numpy>=1.19.2', 'transformers>=4.3.3', 'onnxruntime>=1.6.0', 'onnx>=1.8.0',
            'scikit-learn>=0.23.2', 'torch>=1.6.0', 'pytorch-lightning>=1.2.1',
            'pandas>=0.24.2', 'textvec>=2.0'
        ]
    }
)
