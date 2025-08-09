import os
from groq import Groq
from NLP import embedding
from dotenv import load_dotenv

def get_response(_q, query):
    load_dotenv()

    # Get the embedding model, vector index, and flattened document chunks
    model, index, flattened_chunks = embedding(_q)

    # Encode the query into a vector
    query_vector = model.encode([query]).astype("float32")

    # Perform vector search to retrieve the top 5 most similar chunks
    D, I = index.search(query_vector, k = 5)

    relevant_news = []

    for score, idx in zip(D[0], I[0]):
        relevant_news.append(flattened_chunks[idx])

    client = Groq(api_key = os.getenv("GROQ_API_KEY"))

    if "What are the latest news and outlook for" in query:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a professional financial analyst.

                        Here is the user's question:
                        {query}

                        Below is a collection of news articles related to the company "{_q}":
                        {relevant_news}

                        Please answer the question based on the information above in a clear, professional, and well-reasoned manner.
                        And give a score between -1 and 1, where -1 means "strongly negative" and 1 means "strongly positive", in the format "Score: X", where X is the score.
                        """
            }
            ],
            model = "llama-3.1-8b-instant",
            # https://console.groq.com/docs/model/llama-3.1-8b-instant
        )

    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
            }
            ],
            model = "llama-3.1-8b-instant",
            # https://console.groq.com/docs/model/llama-3.1-8b-instant
        )

    # Batch processing required later

    return chat_completion.choices[0].message.content

