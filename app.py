import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)


model = ChatGroq(
    api_key = "CHATGROQ_API_KEY",
    temperature=0.2,
    model="llama-3.3-70b-versatile"
)


prompt = PromptTemplate(
    template = """
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


st.title("YouTube RAG Q&A System")

video_id = st.text_input("Enter YouTube Video ID (e.g., Gfr50f6ZBvo):")
query = st.text_input("Ask a question based on the video transcript:")

if st.button("Submit"):
    if not video_id or not query:
        st.warning("Please provide both a video ID and a question.")
    else:
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=["en"])
            raw_transcript = fetched.to_raw_data()
            transcript = " ".join(chunk["text"] for chunk in raw_transcript)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            
            vector_store = FAISS.from_documents(chunks, embed_model)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

          
            parallel = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            parser = StrOutputParser()
            chain = parallel | prompt | model | parser

            answer = chain.invoke(query)
            st.success("Answer:")
            st.write(answer)

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except NoTranscriptFound:
            st.error("No transcript found in the requested language.")
        except VideoUnavailable:
            st.error("The video is unavailable.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
