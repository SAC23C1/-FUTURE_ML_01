from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


df = pd.read_csv(r"C:\Users\Sai Discovery\OneDrive\Desktop\Resume_ screnning_project\resumes.csv")

with open("job.txt", "r") as f:
    job_description = f.read()
df['clean_resume'] = df['resume_text'].apply(clean_text)
job_clean = clean_text(job_description)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(
    df['clean_resume'].tolist() + [job_clean])

job_vector = tfidf_matrix[-1]
resume_vectors = tfidf_matrix[:-1]

scores = cosine_similarity(resume_vectors, job_vector)
df['score'] = scores.flatten()

job_skills = set(job_clean.split())


def get_skill_gap(resume):
    words = set(resume.split())
    return ", ".join(job_skills - words)


df['missing_skills'] = df['clean_resume'].apply(get_skill_gap)

df = df.sort_values(by='score', ascending=False)

print("\n=== Ranked Candidates ===\n")
print(df[['name', 'score', 'missing_skills']])
