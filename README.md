# 🏠 Real Estate Voice Assistant – "Agent Alex"

A **voice-enabled real estate assistant made for Real Estate Agents in Dubai** that answers property queries, fills out inquiry forms via speech, and matches users with the most relevant brokers — all powered by **RAG (Retrieval-Augmented Generation)** and speech technologies.

---

## 📌 Project Overview

**Purpose:**  
"Agent Alex" is an interactive voice-based system that:

- Answers user property-related queries using **RAG**.
- Searches a **local CSV-based property & broker database**.
- Assigns the most relevant broker based on location.
- Fills out a property inquiry form via **voice interaction**.

**Mode:** Conversational, powered by:
- **Speech-to-Text (STT)**
- **Text-to-Speech (TTS)**
- **LLM** (OpenAI GPT)

---

## 🛠 Tech Stack

**Core Language:** Python  

**Libraries:**
- `pandas` → Data handling for CSV/Excel.
- `speech_recognition` → STT (Google Speech Recognition API).
- `pyttsx3` → TTS (offline speech synthesis).
- `openpyxl` → Excel read/write.
- `json` → Parsing GPT output into structured search parameters.
- `openai` / `OpenAI` → GPT API calls.
- `os` → File operations.

**Data Storage:**
- `propdata.csv` → Property listings.
- `brokdata.csv` → Broker details.
- Saves leads to Excel (`real_estate_voice_leads.xlsx`) with CSV fallback.

---

## 🔍 RAG Flow

1. **Query Understanding**  
   Checks if query is real-estate-related via keyword + location match.

2. **Parameter Extraction (Retrieval Step)**  
   GPT parses query into JSON:
   ```json
   {
     "location": "...",
     "property_type": "...",
     "max_budget": ...,
     "has_parking": ...
   } 
   
## 🔍 Explanation:

User Speaks Query → Captured via microphone.

STT → Converts speech to text using Google Speech Recognition API.

LLM Query Understanding → GPT extracts structured parameters from the query.

RAG Retrieval → Filters local CSV property database based on extracted parameters.

LLM Generation → GPT forms a conversational reply only from matched data.

TTS → Converts text response into voice.

Broker Matching → Finds best broker based on location.
Lead Storage → Saves the interaction for sales follow-up.

