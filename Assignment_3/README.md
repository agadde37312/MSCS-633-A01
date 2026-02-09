# Assignment 3 – Terminal-Based Chatbot

## Course
MSCS-633-A01

## Assignment
Terminal Client Chatbot Using Django and ChatterBot

---

## Project Overview
This project implements a **terminal-based chatbot** using **Django** and **ChatterBot**.  
The chatbot allows users to interact with a machine-learning conversational agent directly from the terminal using a **custom Django management command**.

ChatterBot generates responses based on a pre-trained corpus of conversations and continuously improves with more input.

---

## Technologies Used
- Python 3.
- Django
- ChatterBot
- ChatterBot Corpus
- SQLite (default Django database)
- spaCy (for NLP processing)

---

## Setup Instructions

### 1. Create and Activate Virtual Environment

python -m venv venv

# Windows PowerShell
venv\Scripts\activate

### 2. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 3. Apply Migrations
python manage.py migrate


### 4. Running Chatbot
python manage.py chat



### 2. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 3. Apply Migrations
python manage.py migrate

###4. Running Chatbot
python manage.py chat
