import requests
from scholarly import scholarly
from transformers import pipeline
import gradio as gr
from scihub import SciHub
import os

# Initialize the Hugging Face Summarization model (BART or T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to search for papers using Google Scholar (via scholarly library)
def search_google_scholar(query):
    search_query = scholarly.search_pubs(query)
    papers = []
    
    # Retrieve top 10 results
    for i in range(5):
        paper = next(search_query)
        # Constructing a Google Scholar URL manually using the title or DOI
        title = paper['bib']['title']
        authors = paper['bib']['author']
        abstract = paper['bib'].get('abstract', 'No abstract available')
        
        # Attempt to build the Google Scholar link from the title (search link)
        search_link = f"https://scholar.google.com/scholar?q={title.replace(' ', '+')}"
        
        # Alternatively, if DOI is available, construct a DOI-based link
        doi = paper['bib'].get('doi', '')
        if doi:
            link = f"https://doi.org/{doi}"
        else:
            link = search_link  # Fallback to search link if DOI is not available
        
        papers.append({
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'link': link,
            'doi': doi  # Save the DOI for downloading
        })
    
    return papers

# Function to download paper from SciHub if it's not open access (using SciHub API)
def download_paper(doi):
    try:
        # Ensure DOI is not empty
        if not doi:
            return "Error: No DOI provided."

        # Using requests to download the paper from SciHub
        url = f"https://sci-hub.se/{doi}"
        response = requests.get(url, stream=True)

        # Ensure the response is successful
        if response.status_code == 200:
            # Create a valid filename (by sanitizing the DOI)
            filename = f"{doi.replace('/', '_')}.pdf"
            filepath = os.path.join(os.getcwd(), filename)  # Save to the current working directory
            
            # Save the file locally
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            return filepath  # Return the local file path
        else:
            return "Error: Unable to download the paper."
    except Exception as e:
        return f"Error downloading paper: {str(e)}"

# Function to summarize the abstract using Hugging Face's BART (or T5)
def summarize_paper(text):
    try:
        summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing paper: {e}"

# Function to handle paper DOI input and return PDF download link
def handle_doi(doi):
    if doi:
        try:
            # Try downloading the paper from SciHub using the provided DOI
            filename = download_paper(doi)
            if filename.startswith("Error"):
                return filename  # Return error message if download fails
            return filename  # Return the file path to be served by Gradio
        except Exception as e:
            return f"Error: {str(e)}"
    return "No DOI provided. Please enter a valid DOI."

# Main function to search, download, and summarize papers
def research_tool(query):
    papers = search_google_scholar(query)
    
    if not papers:  # If no papers are found, return a message
        return "No papers found."  # Return a string, not a tuple
    
    results = []
    
    for paper in papers:
        title, authors, abstract, link, doi = paper['title'], paper['authors'], paper['abstract'], paper['link'], paper['doi']
        
        # Summarize the paper's abstract
        summary = summarize_paper(abstract)
        
        # Provide the link for download (using Google Scholar link or DOI link)
        download_link = link  # Using constructed link (Google Scholar or DOI-based)
        
        # Format each paper's information with Markdown, including line breaks
        formatted_result = f"**Title**: {title}\n**Authors**: {authors}\n**Summary**: {summary}\n**Link**: {download_link}"
        results.append(formatted_result)
    
    # Join the results with a line break between each paper's details
    return "\n\n".join(results)  # Return a single string, not a tuple

# Create the Gradio interface for the user to interact with the search and DOI input
with gr.Blocks() as interface:
    # Add the title to the interface
    gr.Markdown("<h1 style='text-align: center;'>Lunar: The AI Powered Research Tool</h1>")
    gr.Markdown("<h2 style='text-align: center;'>Search for Papers and Download by DOI</h2>")
    gr.Markdown("<p style='text-align: center;'>This tool allows you to search for research papers and download them using their DOI.</p>")
    

    with gr.Column():
        # Top part of the UI for searching papers
        search_input = gr.Textbox(label="Enter Research Topic or Query", placeholder="e.g., Machine Learning for Health")
        search_output = gr.Markdown(label="Search Results")  # Removed 'interactive=False'
        
        # Button to trigger the search manually
        send_button = gr.Button("Send")
        send_button.click(research_tool, inputs=search_input, outputs=search_output)
    
    with gr.Column():
        # Bottom part of the UI for entering DOI to download paper
        doi_input = gr.Textbox(label="Enter DOI", placeholder="e.g., 10.1234/abcd5678")
        download_output = gr.File(label="Download Paper")
        

        
        doi_input.submit(handle_doi, inputs=doi_input, outputs=download_output)

        
# Launch the interface
interface.launch(share=True)
