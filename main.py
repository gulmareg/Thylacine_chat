from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
import logging
import os


from google.cloud import texttospeech


load_dotenv()  


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
print("DG Key:", DEEPGRAM_API_KEY)  

def google_tts_synthesize(text: str) -> bytes:
    """
    Synthesizes the given text into speech (LINEAR16 / WAV) and returns raw audio bytes.
    Make sure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set
    to your GCP service account JSON key file.
    """
    client = texttospeech.TextToSpeechClient()

    # Provide the text to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)


    voice = texttospeech.VoiceSelectionParams(
    language_code="en-AU",           
    name="en-AU-Standard-A",       
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)


    # Configure audio output (LINEAR16 = WAV format)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.2,
        pitch=0.0
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        request={
            "input": synthesis_input,
            "voice": voice,
            "audio_config": audio_config,
        }
    )

    return response.audio_content

# AnimalAssistant class manages the interaction between the user and the LLM.
class AnimalAssistant:
    def __init__(self, system_prompt, llm, message_history=None, vector_store=None, user_age=None):
        self.system_prompt = system_prompt  # The system's behavior and context description.
        self.llm = llm  # The language model used for generating responses.
        self.messages = message_history if message_history else []  # Stores the conversation history.
        self.vector_store = vector_store  # The vector database for retrieving relevant knowledge.
        self.user_age = user_age  # The user's age to tailor responses.

        self.chain = self._get_conversation_chain()  # Initializes the conversation chain.

    def _get_conversation_chain(self):
        # Define the prompt structure and processing chain for the conversation.
        prompt = ChatPromptTemplate([
            ("system", self.system_prompt.format(user_age=self.user_age)),
            MessagesPlaceholder("conversation_history"),
            ("human", "{user_input}"),
        ])

        chain = (
            {
                "retrieved_animal_info": self.vector_store.as_retriever(),  # Retrieve relevant info from vector store.
                "user_age": lambda x: self.user_age,  # Pass the user's age.
                "user_input": RunnablePassthrough(),  # Directly pass the user input.
                "conversation_history": lambda x: self.messages,  # Pass the conversation history.
            }
            | prompt
            | self.llm  # Generate the response using the LLM.
            | StrOutputParser()  # Parse the output into a string format.
        )
        return chain

    def get_response(self, user_input):
        # Stream the LLM's response to the user input.
        return self.chain.stream(user_input)


# AnimalAssistantGUI class manages the user interface for interacting with the assistant.
class AnimalAssistantGUI:
    def __init__(self, assistant, image_path):
        self.assistant = assistant  # The assistant instance for handling conversations.
        self.messages = assistant.messages  # Access to the conversation history.
        self.image_path = image_path  # Path to the image displayed in the UI.

    def get_response(self, user_input):
        # Fetch the assistant's response for the user input.
        return self.assistant.get_response(user_input)

    def render_messages(self):
        # Display the conversation history in the chat UI.
        for message in self.messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            elif message["role"] == "ai":
                st.chat_message("ai").markdown(message["content"])

    def render_user_input(self):
        # 1) Prompt for user input
        user_input = st.chat_input("Ask Platty something...")

        if user_input:
            if user_input.lower() == "goodbye":
                st.warning("You have exited the chat. Please close this tab manually.")
                st.stop()

            # 2) Display the user’s message
            st.chat_message("user").markdown(user_input)

            # 3) Get a streaming response from the LLM
            response_generator = self.get_response(user_input)

            # 4) Stream the final text
            with st.chat_message("assistant"):
                response_text = st.write_stream(response_generator)

            # 5) ALWAYS generate and play audio (no checkbox)
            try:
                audio_data = google_tts_synthesize(response_text)
                st.audio(audio_data, format="audio/wav")
            except Exception as e:
                st.error(f"Failed to generate TTS audio: {e}")

            # 6) Update conversation history
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": response_text})
            st.session_state.messages = self.messages

    def render(self):
        # Render the chat interface including the animal image and messages.
        st.image(self.image_path, caption="Thylacine Chat", use_column_width=True)
        # Display the introductory message
        st.markdown("### Hi, I'm Thyla the Thylacine! What do you want to know about Thylacines?")
        self.render_messages()
        self.render_user_input()


#@st.cache_data(ttl=3600, show_spinner="Loading Vector Store...")
@st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
def init_vector_store(file_path):
    try:
        # Load the .txt file containing animal-related knowledge.
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()

        # Split the document into manageable chunks for efficient processing.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(document)

        # Embed the split documents into the vector database using JinaEmbeddings.
        embedding_function = JinaEmbeddings()
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory="./data/thylacine_vectorstore",
        )

        return vector_store
    except Exception as e:
        # Handle errors during vector store initialization.
        logging.error(f"Error initializing vector store: {str(e)}")
        st.error(f"Failed to initialize vector store: {str(e)}")
        return None


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from a .env file.

    st.set_page_config(page_title="Thylacine Chat", page_icon="🐾")  # Configure the Streamlit app.

    logging.basicConfig(level=logging.INFO)  # Set up logging for debugging.

    # Prompt for the user's age before granting access to the chat.
    user_age = st.number_input("Enter your age to proceed", min_value=1, max_value=120, value=None, step=1, key="user_age_input")

    if "age_verified" not in st.session_state:
        st.session_state.age_verified = False

    if st.button("Submit Age"):
        if user_age:
            st.session_state.age_verified = True
            st.success("Age verified! You may now proceed.")

    if not st.session_state.age_verified:
        st.stop()  # Stop rendering the rest of the app until age is verified.

    # Specify the path to your .txt file containing animal knowledge.
    file_path = r"C:\Users\Precision\Desktop\animal_chat\Thylacine_chat\data\thylacine.txt"

    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initialize the session state for messages.

    # Initialize the vector store with the specified file path.
    vector_store = init_vector_store(file_path)

    # Set up the LLM model for generating responses.
    llm = ChatGroq(model="llama-3.1-8b-instant")

    # Create an instance of the AnimalAssistant with tailored system prompt.
    assistant = AnimalAssistant(
        system_prompt="""
        You're name is Thyla and you are a living Thylacine. 
        You are knowledgeable about Thalycines, their habitats, behaviors, and unique characteristics. 
        Respond to all questions as a Thylacine might. Thylacine's are extinct and any answer you give regarding Thylacine's will be in the past tense.
        Tailor your responses based on the user's age. The user is ({user_age} years old):
        - For younger users (e.g., under 12 years old), keep the language simple.
        - For teenagers, use an educational and engaging tone.
        - For adults, adopt a more sophisticated and detailed conversational style.
        Always adjust your tone and content to ensure the user feels the interaction is age-appropriate.
        """,
        llm=llm,
        message_history=st.session_state.messages,
        vector_store=vector_store,
        user_age=user_age,
    )

    # Render the GUI for the Animal Chat Assistant.
    gui = AnimalAssistantGUI(assistant, r"C:\Users\Precision\Desktop\animal_chat\Thylacine_chat\img\img4.jpeg")
    gui.render()
