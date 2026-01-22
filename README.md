# VerifyChatBot

**VerifyChatBot** is a high-performance AI assistant built with FastAPI and Google Gemini 2.5. It serves as the intelligent support interface for the Verify Platform—a secure ecosystem designed to eliminate fake degrees and certificates using Blockchain, AI, and OCR technologies.

This bot is engineered to guide students, university administrators, and verifiers through complex validation processes, including Ethereum blockchain integration, MetaMask wallet connections, and certificate analysis via YOLOv11 and Tesseract OCR.

This repository contains the source code and dependencies required to deploy the chatbot API locally or in a cloud environment.

## Repository Structure

* **`app.py`**: The main entry point for the application. Contains the core logic for the chatbot.
* **`requirements.txt`**: A list of Python dependencies and libraries required to run the project.
* **`.gitignore`**: Specifies files and directories that should be ignored by Git (e.g., local environments, cache files).

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* **Python 3.x** installed on your system.
* **pip** (Python package installer).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Kashika221/VerifyChatBot.git](https://github.com/Kashika221/VerifyChatBot.git)
    cd VerifyChatBot
    ```

2.  **Create a virtual environment (Recommended):**
    It is good practice to use a virtual environment to manage dependencies.
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install the required Python libraries using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, execute the `app.py` script.

```bash
python app.py

```

> **Note:** If this application is built with a **FastAPI** framework, the command might differ:
> `python app.py`
> 
> 

## Features

* **AI-Powered Assistance**: Built with **LangChain** and **Google Gemini 2.5 Flash**, offering high-speed and context-aware responses.
* **Domain-Specific Persona**: The bot is pre-configured with a specialized system prompt to act as **VerifyBot**, an expert in blockchain verification, OCR (Tesseract/YOLOv11), and certificate authenticity.
* **Contextual Conversations**: Maintains chat history per `session_id`, allowing for multi-turn conversations where the bot remembers previous context.
* **Session Management**: Includes dedicated endpoints to manage user sessions, including a `clear-history` function to reset conversations.
* **FastAPI Architecture**: High-performance, asynchronous REST API structure with automatic interactive documentation (Swagger UI).
* **CORS Enabled**: Configured with Cross-Origin Resource Sharing (CORS) middleware to seamlessly integrate with frontend applications (React/TypeScript).
* **Secure Configuration**: Uses environment variables (`.env`) for secure API key management.

### API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/chat` | Sends a message to the bot and receives a response (requires `session_id`). |
| `POST` | `/clear-history` | Resets the conversation memory for a specific session. |
| `GET` | `/health` | Simple health check to verify the API is running. |
| `GET` | `/` | Root endpoint confirming API status. |

## Configuration

[If your app requires API keys (like OpenAI, Discord, etc.), mention how to set them up here. Example below:]

1. Create a `.env` file in the root directory.
2. Add your API keys:
```env
API_KEY=your_api_key_here

```



## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

[State the license here, e.g., MIT, Apache 2.0, or "This project is unlicensed".]

## Contact

* **Author:** Kashika221
* **GitHub:** [github.com/Kashika221](https://www.google.com/search?q=https://github.com/Kashika221)

```

```