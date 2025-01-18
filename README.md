
# Thylacine Chat

Thylacine Chat is an interactive AI application that brings the extinct Thylacine (also known as the Tasmanian Tiger) to life. 
Using advanced language models and a vector database, it provides engaging, age-appropriate conversations about Thylacines, their behaviors, and habitats.

This app was developed to deliver the following:

Education and Awareness: It helps users learn about the Thylacine, its behavior, habitat, and extinction in an engaging way while promoting environmental awareness.

Interactive Learning: By simulating a conversation with a Thylacine, it creates a memorable and interactive learning experience tailored to different age groups.

Preservation of Heritage: It serves as a medium to keep the memory of extinct species alive, fostering appreciation for biodiversity and its fragility.

Technological Showcase: Demonstrates the integration of AI technologies like natural language processing, vector databases, and text-to-speech in an innovative application.

Entertainment and Curiosity: Appeals to those curious about extinct animals and history, providing a unique experience of "talking" to an extinct creature with realistic audio responses.

---

## Features

- **Interactive Chat**: Simulates a conversation with "Thyla," a Thylacine persona.
- **Age-Adapted Responses**: Tailors content based on the user's age.
- **Vector Store Integration**: Retrieves relevant knowledge about Thylacines from a vector database.
- **Text-to-Speech**: Converts responses into audio with an Australian accent.
- **Streamlit Frontend**: Provides an intuitive, user-friendly interface.

---

## Prerequisites

- Python 3.10+
- Google Cloud Platform credentials for Text-to-Speech API
- Environment variables file (`.env`) containing API keys:
  ```
  GROQ_API_KEY=your_groq_api_key
  JINA_API_KEY=your_jina_api_key
  LAGNCHAIN_API_KEY=your_langchain_api_key
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=animal_chat 
  DEEPGRAM_API_KEY=your_deepgram_api_key
  GOOGLE_APPLICATION_CREDENTIALS=path_to_gcp_credentials.json
  ```
- Required Python packages (see below).

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd thylacine_chat
   ```

2. **Set Up Environment**:
   Create a `.env` file with the required API keys and credentials.

3. **Install Dependencies**:
   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Knowledge Base**:
   Place your knowledge base file (e.g., `thylacine.txt`) in the `data` directory.

---

## Usage

1. **Run the Application**:

   For speech audio response
   ```bash
   streamlit run main.py 

   ```
   
   For no speech audio response
   ```bash
   streamlit run main_no_audio.py 

   ```

2. **Provide Age Verification**:
   Enter your age to ensure age-appropriate content.

3. **Chat with Thyla**:

   - Type your questions in the chat interface.
   - Receive responses tailored to your age.
   - Listen to Thyla's responses through the built-in Text-to-Speech feature.

---

## Project Structure

```
Thylacine_Chat
├── data/
│   └── thylacine.txt         # Knowledge base file
├── img/
│   └── img4.jpeg             # Image for Streamlit UI
├── main.py                   # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .env                      # Environment variables file
```

---

## Key Modules

### `AnimalAssistant`

- Handles interaction with the language model.
- Retrieves relevant knowledge from the vector database.
- Generates responses tailored to the user's age.

### `AnimalAssistantGUI`

- Manages the Streamlit-based user interface.
- Streams responses and plays synthesized audio.

### Google Text-to-Speech Integration

- Synthesizes responses into speech using GCP's Text-to-Speech API.
- Configured to use an Australian accent.

---

## Configuration

### Environment Variables

Ensure the following environment variables are set:

- `DEEPGRAM_API_KEY`: Your Deepgram API key.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your GCP service account JSON key file.

### Vector Store

To initialize the vector store, place the knowledge base file (`thylacine.txt`) in the `data` directory. The application will split the content and embed it using JinaEmbeddings.

---

## Dependencies

- `langchain`
- `streamlit`
- `google-cloud-texttospeech`
- `dotenv`
- `jina`
- `chroma`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Add voice-to-voice interaction.
- Expand to support other extinct or endangered species.
- Improve personalization with more advanced user profiling.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- Built using LangChain, Streamlit, and Google Cloud Text-to-Speech.
- Inspired by the unique history of the Thylacine.

---

## Contributing

Contributions are welcome! If you want to improve the project, submit a pull request or open an issue.

---


s.
