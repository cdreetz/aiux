import chromadb
from config.clients import ChromaDBClient


def query(query_texts, n_results=2, where={"ResourceType":"Content"}):
    '''
    Query from the collection and return the documents.

    Parameters:
    - query_texts (list): List of query texts.
    - n_results (int): Number of results to retrieve. Default is 2.

    Returns:
    - A single string containing all the documents concatenated, or a list of documents.
    '''

    collection = ChromaDBClient.setup_chroma_collection('dev')

    try:
        # Ensure that query_texts is a list
        if not isinstance(query_texts, list):
            query_texts = [query_texts]

        results = collection.query(query_texts=query_texts, n_results=n_results, where=where)
        documents = []
        for i, document_list in enumerate(results['documents']):
            for j, document in enumerate(document_list):
                if results['distances'][i][j] < 0.5:
                    metadata = results['metadatas'][i][j]
                    documents.append({'document': document, 'metadatas': metadata})

        if not documents:
            print("No relevant documents")
            return "No relevant documents"

        return documents

    except Exception as e:
        print(f"Error during query: {e}")
        return []