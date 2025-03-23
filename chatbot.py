import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain.globals import set_llm_cache
from upstash_semantic_cache import SemanticCache
import gradio as gr


load_dotenv()

cache = SemanticCache(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"), token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"), min_proximity=0.7
)
set_llm_cache(cache)



embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv("API_KEY"))

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            temperature=0.5,
                            repetition_penalty=1.03,
                            huggingfacehub_api_token= os.getenv("HF_API"))


database_path = r"chromadb"

vector_store = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory=database_path, 
)

num_results = 5
retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={'k': num_results})

def stream_response(message, history):

    docs = retriever.invoke(message)

    knowledge = "".join(d.page_content+ "\n\n" for d in docs)
    
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you use your internal knowledge only to reply messeages that are out of context of the provided knowledge. Other than
        this, solely use the information in the "The knowledge" section.
        

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        response = llm.invoke(rag_prompt)
        return response 

  
def yes(message, history):
    return "yes"

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])



with gr.Blocks() as test:
    chatbot = gr.Chatbot(placeholder="<strong>Your Personal Bot</strong><br>",
    avatar_images=(None,"https://media-hosting.imagekit.io//af943f5d2df64d65/geek-avatar-1632962.jpg?Expires=1835706136&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=J3nliAd1SFngONboPUMI58m92rYsT4JRbYx4TNhe630ppSuQbcXXAiKRtcQHLYYsl1nPJPwuFKgF747gByKUFUHCr9b7pMYEeXaFxCY6w9lZ3VHZv6wtham~EarzjXHO-nCYAxx3TF818Rjb~9EErCG9fpt3nPr9L-ZxmHpfZfynzjaHAkn-5VcBQLiSv9by-yMRGWANkxWHb2oLixumIGqpW4u1D1mv4GoYNK8bKDZ3EaKuI1ilzKKflQ1Zzp6Fna9PAWIQTCJ08DnZTM4BUEDqbqrv~Mdgt6zWJng0Xx1URFtkakg5JHH6UzetTXPVFRFVXlslGaSdXAKHPIAkwg__"))
    chatbot.like(vote, None, None)
    gr.ChatInterface(stream_response, type="messages", textbox=gr.Textbox(placeholder="Ask anything..",
    container=True,
    autoscroll=True,
    scale=7),chatbot = chatbot)

test.launch(share=True)


