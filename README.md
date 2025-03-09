# LifeLine AI - Chat for Health Advice

## Overview
LifeLine AI is an AI-powered chatbot designed to provide reliable health advice based on credible sources. This chatbot retrieves relevant medical knowledge using advanced AI models while ensuring that users are encouraged to consult healthcare professionals for medical concerns.

## Features
- **AI-driven health guidance**
- **Source-referenced responses**
- **Real-time question answering**
- **User-friendly chat interface**
- **Secure and efficient retrieval system**

## Prerequisites
Make sure you have **Pipenv** installed on your system. If not, follow the official guide here:
👉 [Install Pipenv](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Clone the Repository
```bash
git clone https://github.com/basir2353/chatbot.git
cd chatbot
```

### Install Required Packages
Run the following commands to install dependencies:
```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub streamlit
```

### Set Up Environment Variables
Create a `.env` file in the root directory and add your Hugging Face API token:
```
HF_TOKEN=your_huggingface_api_token_here
```

### Run the Application
```bash
pipenv shell  # Activate the virtual environment
streamlit run app.py  # Start the chatbot
```

---

## Contact & Social Links
Developed by **Abdul Basit**. Feel free to reach out!

🔗 [LinkedIn](https://www.linkedin.com/in/abdul-basit-1a56b3275/)  
🐙 [GitHub](https://github.com/basir2353)  
📷 [Instagram](https://www.instagram.com/dogar_basit08/)  
📘 [Facebook](https://www.facebook.com/mabdulbasit.dogar.1)  
📞 Contact: [+92 346 9517653](tel:+923469517653)

---

## Notes
- This chatbot is for informational purposes only and **does not provide medical diagnoses**.
- Always consult a healthcare professional for medical advice.
- Contributions are welcome! Feel free to fork and improve the project.

Happy coding! 🚀