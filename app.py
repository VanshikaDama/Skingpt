from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from lang_model import perform_object_detection, generate_response

app = Flask(__name__)

current_disease = None  # Define current_disease variable globally

@app.route('/')
def imageinput():
    return render_template('firstpage.html')


@app.route('/process-data', methods=['POST'])
def process_data():
    global current_disease
    if 'image' in request.files:
        image_file = request.files['image']
        current_disease = perform_object_detection(image_file)  # Update current_disease
        return redirect(url_for('chatbot'))


@app.route('/chatbot')
def chatbot():
    global current_disease
    return render_template('chatbot.html', disease=current_disease)


@app.route('/queries', methods=['POST'])
def query():
    global current_disease
    text_input = request.form['text']
    answer = generate_response(current_disease, text_input)
    print(answer)
    return answer


if __name__ == "__main__":
    app.run(debug=True)


    #  if 'image' in request.files:
    #     image_file = request.files['image']
    #     image_filename = image_file.filename
    # else:
    #     image_filename = None

    # text_input = request.form['text']

    # if image_filename and text_input:
    #     ans  = perform_object_detection(image_file)
    #     answer = generate_response(ans,text_input)
    #     response_message = f"The given disease is {ans}.{answer}.Ask more questions or type 'exit' to exit"