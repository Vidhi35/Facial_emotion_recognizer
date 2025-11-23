# Facial Emotion Recognizer

![Demo](https://huggingface.co/spaces/Vidhi35/Face-Expression-Recognition/resolve/main/demo.gif)

A deep learning-based web app to detect human facial emotions from images using a Convolutional Neural Network (CNN). Instantly recognize emotions like happy, sad, angry, surprise, and more!

## 🚀 Live Demo

Try it now: [Facial Emotion Recognizer on Hugging Face Spaces](https://huggingface.co/spaces/Vidhi35/Face-Expression-Recognition)

## 📦 GitHub Repository

Source code: [github.com/Vidhi35/Facial_emotion_recognizer](https://github.com/Vidhi35/Facial_emotion_recognizer)

---

## Features

- Upload any face image and get instant emotion prediction
- Seven emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Interactive Gradio web interface
- Trained on the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- Model built with TensorFlow/Keras

---

## How It Works

1. **Image Preprocessing:** Uploaded images are converted to grayscale and resized to 48x48 pixels.
2. **Model Prediction:** A CNN model predicts the emotion class probabilities.
3. **Result Display:** The top emotion and probability scores are shown in the web interface.

---

## Model Architecture

- 4 Convolutional layers (128, 256, 512, 512 filters)
- MaxPooling and Dropout for regularization
- 2 Dense layers (512, 256 units)
- Output: Softmax layer for 7 emotion classes

Model files:
- `emotiondetector.h5` (weights)
- `emotiondetector.json` (architecture)

---

## Usage

### Online

Just visit the [Hugging Face Space](https://huggingface.co/spaces/Vidhi35/Face-Expression-Recognition) and upload your image!

### Local Setup

1. Clone the repo:
	```sh
	git clone https://github.com/Vidhi35/Facial_emotion_recognizer.git
	cd Facial_emotion_recognizer
	```
2. Install dependencies:
	```sh
	pip install tensorflow keras gradio numpy pillow pandas scikit-learn tqdm
	```
3. Run the app:
	```sh
	python app.py
	```
4. Open the Gradio link in your browser.

---

## Dataset

- [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- Contains labeled images for 7 emotions.

---

## Screenshots

![Interface Screenshot](https://huggingface.co/spaces/Vidhi35/Face-Expression-Recognition/resolve/main/screenshot.png)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- [Gradio](https://gradio.app/)
- [TensorFlow](https://www.tensorflow.org/)

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first.