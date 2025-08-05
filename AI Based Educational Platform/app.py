from flask import Flask, json, render_template, request, redirect, url_for, jsonify
from career_model import recommend
from answer_model import extract_text_from_pdf, load_cached_text, save_extracted_text, query_gemini_api
import os

app = Flask(__name__)

# Home page
@app.route('/')
def homepage():
    return render_template('homepage.html')



# Dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/career_roadmap', methods=['POST', 'GET'])
def career_roadmap():
    if request.method == 'POST':
        # Fetch data from the form
        user_present_domain = request.form.get('present_domain')
        user_area_of_interest = request.form.get('area_of_interest')
        user_career_aspiration = request.form.get('career_aspiration')
        user_excelled_subjects = request.form.getlist('excelled_subjects')  # Get as a list
        user_long_term_vision = request.form.get('long_term_vision')

        # Convert list to a single string (if needed for further processing)
        user_excelled_subjects = ', '.join(user_excelled_subjects)

        # Call the recommend function with the form data
        recommendations = recommend((
            user_present_domain,
            user_area_of_interest,
            user_career_aspiration,
            user_excelled_subjects,
            user_long_term_vision,
        ))

        # Redirect to the recommendations page with data
        return redirect(url_for('recommendation', recommendations=json.dumps(recommendations)))

    return render_template('career_roadmap.html')

@app.route('/career_roadmap/recommendation', methods=['GET'])
def recommendation():
    recommendations = request.args.get('recommendations')
    if recommendations:
        recommendations = json.loads(recommendations)  # Deserialize JSON string into a Python object
    return render_template('recommendation.html', recommendations=recommendations)


if not os.path.exists("temp_dir"):
    os.makedirs("temp_dir")

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_message = request.form.get('message')
        pdf = request.files.get('pdf')

        if pdf:
            pdf_name = pdf.filename
            pdf_path = os.path.join("temp_dir", pdf_name)
            pdf.save(pdf_path)

            # Check if the extracted text is cached
            extracted_text = load_cached_text(pdf_name)

            if extracted_text is None:
                # Extract text from PDF and cache it
                try:
                    extracted_text = extract_text_from_pdf(pdf_path)
                    save_extracted_text(pdf_name, extracted_text)
                except Exception as e:
                    app.logger.error(f"Error extracting text from PDF: {str(e)}")
                    return jsonify({"response": "There was an error processing the PDF file. Please try again."})

            # Send a confirmation message
            return jsonify({"response": "File uploaded successfully. You may now start asking questions!"})

        if user_message:
            pdf_name = request.form.get('pdf_name')
            extracted_text = load_cached_text(pdf_name)

            if not extracted_text:
                return jsonify({"response": "Please upload a PDF file first."})

            try:
                answer = query_gemini_api(user_message, extracted_text)
                return jsonify({"response": answer})
            except Exception as e:
                app.logger.error(f"Error while querying Gemini API: {str(e)}")
                return jsonify({"response": "There was an error querying the Gemini API. Please try again."})

    # Render the HTML interface on a GET request
    return render_template('chat_bot.html')




@app.route('/chatback')
def chatback():
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)
