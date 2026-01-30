from papertrail.ingestion.arxiv_client import ArxivClient
from papertrail.ingestion.pdf_loader import download_pdf, save_processed_text
from papertrail.processing.splitters import split_and_save

def main():
    print("Hello from papertrail!")

    client = ArxivClient(
        categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"], 
        max_results=5)
    
    papers = client.fetch_papers()
    for i, p in enumerate(papers):
        print(f"{i+1}/{len(papers)} :Downloading paper {p.title}")
        file_path = download_pdf(p)
        file_path = save_processed_text(file_path, clean=True)
        splitted_text = split_and_save(file_path)
        print("splitted: ", type(splitted_text))
        # break


if __name__ == "__main__":
    main()
