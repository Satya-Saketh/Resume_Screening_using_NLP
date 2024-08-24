
import streamlit as st
import pickle
import re
import nltk
from time import sleep

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    # Convert to lowercase
    txt = txt.lower()
    # Remove URLs
    txt = re.sub(r'http\S+', '', txt)
    # Remove RT and cc
    txt = re.sub(r'\b(rt|cc)\b', '', txt)
    # Remove hashtags
    txt = re.sub(r'#\S+', '', txt)
    # Remove mentions
    txt = re.sub(r'@\S+', '', txt)
    # Remove special characters and punctuations
    txt = re.sub(r'[^\w\s]', ' ', txt)
    # Remove non-ASCII characters
    txt = re.sub(r'[^\x00-\x7f]', '', txt)
    # Tokenize the text
    tokens = word_tokenize(txt)
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    # Lemmatize and remove stopwords
    lemmatized_tokens = []
    for word, tag in tagged_tokens:
        if word not in stop_words:
            if tag.startswith('V'):  # Verb
                lemmatized_word = lemmatizer.lemmatize(word, pos='v')
            elif tag.startswith('J'):  # Adjective
                lemmatized_word = lemmatizer.lemmatize(word, pos='a')
            elif tag.startswith('R'):  # Adverb
                lemmatized_word = lemmatizer.lemmatize(word, pos='r')
            else:  # Noun (default)
                lemmatized_word = lemmatizer.lemmatize(word)
            lemmatized_tokens.append(lemmatized_word)
    # Join tokens back into a single string
    cleanText = ' '.join(lemmatized_tokens)
    return cleanText

def main():
    # Custom title with styling
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Resume Screening App</h1>", 
        unsafe_allow_html=True
    )

    # Sidebar for uploading resume
    st.sidebar.header("Upload Your Resume")
    uploaded_file = st.sidebar.file_uploader('Choose a file', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        
        # Display progress bar
        with st.spinner('Processing your resume...'):
            sleep(2)  # Simulate a delay
        
        cleaned_resume = clean_resume(resume_text)
        
        # Display a summary of the cleaned resume
        st.subheader("Resume Summary")
        st.write(cleaned_resume[:500] + "...")  # Show only the first 500 characters
        
        # Transform and predict
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            # Add other mappings here
        }

        # Display the prediction result
        st.subheader("Predicted Job Category")
        st.write(category_mapping.get(prediction_id, "Unknown Category"))

if __name__ == '__main__':
    main()
