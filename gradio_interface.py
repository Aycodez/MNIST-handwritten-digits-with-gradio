import keras
import gradio as gr

#Loading saved model
model = keras.models.load_model('Digitclassifier.keras')

# Lets create a function to recognize the images
def recognize_image_digits(image):
    if image is not None:
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        prediction = model.predict(image)
        return {str(i): prediction[0][i] * 10 for i in range(10)}
    else:
        return ''

#Creating a GUI interface with Gradio
iface = gr.Interface(
    fn=recognize_image_digits,
    inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, source='canvas'),
    outputs=gr.Label(num_top_classes=5),
    live=True
)
# It will return a url, open the url in your browser to make use of the interface
iface.launch()
