import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import google.generativeai as gemini

# Step 1: Configure Gemini API Key
GEMINI_API_KEY = ""  # Replace with your actual API key
gemini.configure(api_key=GEMINI_API_KEY)

# Step 2: Load the dataset
df = pd.read_excel('data_set.xlsx')
df.fillna('', inplace=True)  # Fill missing values with an empty string

# Helper function to process dataset values with multiple entries
def process_multiple_values(column):
    return column.apply(lambda x: ' '.join(x.split(', ')))

# Preprocess dataset columns with multiple values
df['Area of Interest'] = process_multiple_values(df['Area of Interest'])
df['Career Aspiration'] = process_multiple_values(df['Career Aspiration'])
df['Excelled Subject'] = process_multiple_values(df['Excelled Subject'])
df['Long Term Vision'] = process_multiple_values(df['Long Term Vision'])

# Step 3: Define a function to compute similarity
def get_similarity(user_input, dataset_column, weight_cosine=0.7, weight_fuzzy=0.3, split_input=False):
    if split_input:
        # Split user input into multiple entries and process individually
        inputs = user_input.split(', ')
        combined_sim = pd.Series([0.0] * len(dataset_column))
        for inp in inputs:
            vectorizer = TfidfVectorizer()
            all_data = dataset_column.tolist() + [inp]
            tfidf_matrix = vectorizer.fit_transform(all_data)

            # Compute cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

            # Compute fuzzy similarity
            fuzzy_sim = dataset_column.apply(lambda x: fuzz.partial_ratio(x, inp) / 100.0)

            # Combine similarities
            combined_sim += (weight_cosine * cosine_sim + weight_fuzzy * fuzzy_sim) / len(inputs)
        return combined_sim
    else:
        # Original functionality for single input
        vectorizer = TfidfVectorizer()
        all_data = dataset_column.tolist() + [user_input]
        tfidf_matrix = vectorizer.fit_transform(all_data)

        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Compute fuzzy similarity
        fuzzy_sim = dataset_column.apply(lambda x: fuzz.partial_ratio(x, user_input) / 100.0)

        # Combine similarities
        combined_sim = weight_cosine * cosine_sim + weight_fuzzy * fuzzy_sim
        return combined_sim


# Step 4: Define the recommendation function
def recommend(user_inputs):
    # Unpack user inputs
    user_present_domain, user_area_of_interest, user_career_aspiration, user_excelled_subjects, user_long_term_vision = user_inputs

    # Compute similarity scores
    similarity_present_domain = get_similarity(user_present_domain, df['Present Domain'])
    similarity_area_of_interest = get_similarity(user_area_of_interest, df['Area of Interest'])
    similarity_career_aspiration = get_similarity(user_career_aspiration, df['Career Aspiration'])
    similarity_excelled_subjects = get_similarity(user_excelled_subjects, df['Excelled Subject'], split_input=True)
    similarity_long_term_vision = get_similarity(user_long_term_vision, df['Long Term Vision'])

    # Weighted similarity scoring
    total_similarity_all = (
        0.15 * similarity_present_domain +
        0.3 * similarity_area_of_interest +
        0.2 * similarity_career_aspiration +
        0.15 * similarity_excelled_subjects +
        0.2 * similarity_long_term_vision
    )

    # Dynamic threshold for recommendation
    best_match_idx = total_similarity_all.argmax()

    # Extract initial recommendations
    recommended_courses = df.loc[best_match_idx, 'Recommended Courses and Certificates']
    recommended_textbooks = df.loc[best_match_idx, 'Recommended Textbooks']
    career_roadmap = df.loc[best_match_idx, 'Career Roadmap']

    # Now, generate content with the Gemini API based on the user's inputs and initial recommendations
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2000,
        "response_mime_type": "text/plain",
    }

    model = gemini.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    contents = f"""
    You are an expert in career planning and academic advising. Based on the following user inputs and initial recommendations, provide the most accurate, enriched, and detailed recommendations for textbooks, courses, certificates, and career roadmaps. Feel free to add additional insights or suggestions to better align with the user's goals and aspirations.

    ### User Inputs:
    - **Present Domain**: {user_present_domain}
    - **Area of Interest**: {user_area_of_interest}
    - **Career Aspiration**: {user_career_aspiration}
    - **Excelled Subjects**: {user_excelled_subjects}
    - **Long Term Vision**: {user_long_term_vision}

    ### Initial Recommendations:
    - **Courses and Certificates**: {recommended_courses}
    - **Textbooks**: {recommended_textbooks}
    - **Career Roadmap**: {career_roadmap}

    ### Task:
    1. Refine and enhance the initial recommendations to ensure they are precise and actionable.
    2. Suggest additional courses, certificates, textbooks, or learning resources that may be beneficial.
    3. Provide a step-by-step career roadmap tailored to the user's inputs, ensuring it aligns with their long-term vision and career aspirations.

    ### Output Structure:
    Provide the output in the following structured format:
    - **Final Recommended Courses and Certificates**:  
      - [Course/Certificate Name]: A 1-2 sentence explanation of why this course/certificate is relevant and how it will benefit the user.  
      - (Add only more Course/Certificate Name with explanation .)
    
    - **Final Recommended Textbooks**:  
      - [Textbook Title]: A 2-3 sentence explanation of why this textbook is useful and what knowledge or skills it imparts.  
      - (Add only more Textbook Title with explanation.)
    
    - **Final Career Roadmap**:  
      - [Step 1]: A detailed description (2-3 sentences) explaining this step, why it is important, and how it helps the user achieve their goals.  
      - [Step 2]: (Continue with additional steps, providing explanations for each. Add more steps as needed.)

    ### Additional Guidelines:
    - Be specific and ensure each explanation is actionable and tailored to the user's unique aspirations.
    - Highlight how each course, textbook, or roadmap point addresses the user's current skills, interests, and long-term vision.
    - Include practical and realistic advice that can be immediately implemented.
    """

    # Generate content using the Gemini model
    response = model.generate_content(contents=contents)
    response_text = response.text

    # Extract the individual sections from the generated content
    recommended_courses = extract_courses_from_response(response_text)
    recommended_textbooks = extract_textbooks_from_response(response_text)
    career_roadmap = extract_career_roadmap_from_response(response_text)

    user_inputs_summary = f"""
Summarize the user's inputs for display on a web interface. Ensure clarity, relevance, and conciseness.

### Instructions:
1. Start with a concise, 50-word summary of the user's inputs.
2. Follow this with a smooth transition to the recommendations.
3. Structure the content clearly for the user to review and understand.


### User Inputs:
- **Present Domain**: {user_present_domain}
- **Area of Interest**: {user_area_of_interest}
- **Career Aspiration**: {user_career_aspiration}
- **Excelled Subjects**: {user_excelled_subjects}
- **Long Term Vision**: {user_long_term_vision}

### Output Structure:
   Clearly summarize the user's key inputs. and  end with Based on your inputs, here are the recommendations that best suit your profile and aspirations.")  
"""




    response2 = model.generate_content(contents=user_inputs_summary)
    response_text1 = response2.text

    # Return a dictionary with separate parts
    return {
        "user_inputs_summary": response_text1,
        "recommended_courses": recommended_courses,
        "recommended_textbooks": recommended_textbooks,
        "career_roadmap": career_roadmap
    }


    

def replace_bold_text_with_html(text):
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    return text


def convert_to_pointwise(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n- '.join(lines)


def extract_courses_from_response(response_text):
    start = response_text.find("Final Recommended Courses and Certificates") + len("Final Recommended Courses and Certificates")
    end = response_text.find("Final Recommended Textbooks")
    course_text = response_text[start:end].strip()
    
    course_text = replace_bold_text_with_html(course_text)
    return convert_to_pointwise(course_text)


def extract_textbooks_from_response(response_text):
    start = response_text.find("Final Recommended Textbooks") + len("Final Recommended Textbooks")
    end = response_text.find("Final Career Roadmap")
    textbook_text = response_text[start:end].strip()
    
    textbook_text = replace_bold_text_with_html(textbook_text)
    return convert_to_pointwise(textbook_text)

def extract_career_roadmap_from_response(response_text):
    start = response_text.find("Final Career Roadmap") + len("Final Career Roadmap")
    roadmap_text = response_text[start:].strip()
    
    roadmap_text = replace_bold_text_with_html(roadmap_text)
    return convert_to_pointwise(roadmap_text)



'''if __name__ == '__main__':
    # Define example user inputs
    user_present_domain = "Mechanical Engineering"
    user_area_of_interest = "Software Development"
    user_career_aspiration = "Software Engineer"
    user_excelled_subjects = "Software, Programming"
    user_long_term_vision = "Software Development Leader"

    # Combine the user inputs into a tuple
    user_inputs = (user_present_domain, user_area_of_interest, user_career_aspiration, user_excelled_subjects, user_long_term_vision)

    # Call the recommend function with the user inputs
    recommendations = recommend(user_inputs)

    # Display the returned recommendations (formatted outputs)
    print("User Input Summary:")
    print(recommendations["user_inputs_summary"])
    print("\nRecommended Courses and Certificates:")
    print(recommendations["recommended_courses"])
    print("\nRecommended Textbooks:")
    print(recommendations["recommended_textbooks"])
    print("\nCareer Roadmap:")
    print(recommendations["career_roadmap"])'''
  