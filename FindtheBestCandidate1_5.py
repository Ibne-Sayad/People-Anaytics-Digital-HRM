import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from collections import Counter
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the path to resumes and skills CSV file
resume_path = r'D:\UNIVERSITY\PEOPLE ANALYTICS\PA_Code\Resume'
resume_files = [os.path.join(resume_path, f) for f in os.listdir(resume_path) if os.path.isfile(os.path.join(resume_path, f))]
skills_csv_file = r'D:\UNIVERSITY\PEOPLE ANALYTICS\PA_Code\data\skills.csv'

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text_list = []
    for page in document:
        page_text = page.get_text()
        text_list.append(page_text)
    return " ".join(text_list)

# Function to build a skill profile from a resume
def build_skill_profile(file_path):
    resume_text = extract_text_from_pdf(file_path).lower().replace("\\n", " ")

    # Load skills from CSV file
    skills_df = pd.read_csv(skills_csv_file, encoding='ansi')
    
    # Define skill categories and their abbreviations
    skill_categories = {
        'Statistics': 'Stats',
        'Mathematics': 'Math',
        'Artificial Intelligence': 'AI',
        'Programming': 'Prog',
        'Cloud Computing': 'CloudComp',
        'Digital Transformation Manager': 'DTManager'
    }
    
    # Initialize the PhraseMatcher
    skill_matcher = PhraseMatcher(nlp.vocab)
    
    # Add skills to the matcher for each category
    for category, abbreviation in skill_categories.items():
        skill_phrases = [nlp(phrase) for phrase in skills_df[category].dropna()]
        skill_matcher.add(abbreviation, None, *skill_phrases)
    
    # Match skills in the resume text
    resume_doc = nlp(resume_text)
    skill_matches = skill_matcher(resume_doc)
    
    # Collect matched skills and their categories
    matched_skills = [(nlp.vocab.strings[match_id], span.text) for match_id, start, end in skill_matches for span in [resume_doc[start:end]]]
    
    # Count matched skills
    skill_counter = Counter(matched_skills)
    
    if not skill_counter:
        return pd.DataFrame(columns=['Employee Name', 'Category', 'Keyword', 'Count'])
    
    # Create a DataFrame from matched skills
    skill_df = pd.DataFrame(skill_counter.items(), columns=['Category_Keyword', 'Count'])
    
    # Split the combined category and keyword into separate columns
    skill_df[['Category', 'Keyword']] = skill_df['Category_Keyword'].apply(lambda x: pd.Series(x))
    skill_df.drop(columns=['Category_Keyword'], inplace=True)
    
    # Extract employee name from the filename
    employee_name = os.path.basename(file_path).split('_')[0].lower()
    skill_df['Employee Name'] = employee_name
    
    return skill_df[['Employee Name', 'Category', 'Keyword', 'Count']]

# Process each resume to create skill profiles
all_profiles = pd.DataFrame()

for resume in resume_files:
    profile = build_skill_profile(resume)
    all_profiles = pd.concat([all_profiles, profile], ignore_index=True)

# Aggregate skills by employee and category
profile_summary = all_profiles.groupby(['Employee Name', 'Category'])['Keyword'].count().unstack(fill_value=0).reset_index()
profile_data = profile_summary.set_index('Employee Name')

# Plot the skills per category using Matplotlib with a different color palette
plt.rcParams.update({'font.size': 10})
ax = profile_data.plot.barh(title="Skills per category", stacked=True, figsize=(25, 7), colormap='tab20')

# Create labels for the bars
bar_labels = []
for col in profile_data.columns:
    for val in profile_data[col]:
        bar_labels.append(f"{col}: {val}")

# Annotate bars with labels
for label, rect in zip(bar_labels, ax.patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x() + width / 2
        y = rect.get_y() + rect.get_height() / 2
        ax.text(x, y, label, ha='center', va='center')

# Display the plot
plt.show()
