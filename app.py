import gradio as gr
import chromadb
import openai  # Use DeepSeek API if preferred

# Set up OpenAI API key
openai.api_key = "your_openai_api_key"

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("interview_questions")

# Sample Interview Questions (Add more as needed)
sample_data = [
    {"role": "Software Engineer", "question": "Explain the concept of OOP with examples."},
    {"role": "Software Engineer", "question": "How does memory management work in Python?"},
    {"role": "Data Scientist", "question": "What is the difference between supervised and unsupervised learning?"},
    {"role": "Data Scientist", "question": "How do you handle missing data in a dataset?"},
]

# Insert sample data into ChromaDB
for i, data in enumerate(sample_data):
    collection.add(ids=[str(i)], metadatas=[data], documents=[data["question"]])

# Function to generate interview questions
def generate_questions(job_title):
    # Retrieve related questions from ChromaDB
    results = collection.query(
        query_texts=[job_title],
        n_results=3  # Fetch top 3 relevant questions
    )

    retrieved_questions = [q for q in results["documents"][0]] if results["documents"] else []

    # Generate additional questions using ChatGPT
    prompt = f"Generate 3 unique interview questions for a {job_title} position."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    generated_questions = [msg["message"]["content"] for msg in response["choices"]]

    return retrieved_questions + generated_questions

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## AI-Powered Interview Question Generator 🎤🤖")
    job_title_input = gr.Textbox(label="Enter Job Title")
    generate_button = gr.Button("Generate Questions")
    output = gr.Textbox(label="Generated Questions", lines=5)
    
    generate_button.click(generate_questions, inputs=job_title_input, outputs=output)

# Run the Gradio app
demo.launch()
