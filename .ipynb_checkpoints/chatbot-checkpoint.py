import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import transformers

# Suppress BERT warnings
transformers.logging.set_verbosity_error()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


# Function to load hotel Q&A data from hotels.txt
def load_hotel_data():
    hotel_data = {}
    current_hotel = None
    current_qna = {}

    with open("hotels.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            if not line.startswith("Q:") and not line.startswith("A:"):  
                # It's a hotel name (e.g., "Hotel A:")
                current_hotel = line[:-1]  # Remove trailing colon
                hotel_data[current_hotel] = {}  # Initialize dictionary
            elif line.startswith("Q:"):
                question = line[2:].strip().lower().replace("wi-fi", "wifi").replace("cancel", "cancellation")  # Normalize "Wi-Fi" and "Cancel"
                hotel_data[current_hotel][question] = ""  # Initialize answer slot
                last_question = question
            elif line.startswith("A:") and current_hotel and last_question:
                hotel_data[current_hotel][last_question] = line[2:].strip()  # Store answer

    return hotel_data


# Function to find best-matching question (improved logic)
def find_best_match(user_input, hotel_qna):
    user_input = user_input.lower().strip().replace("wi-fi", "wifi").replace("cancel", "cancellation")  # Normalize

    # First, try **exact** match
    if user_input in hotel_qna:
        return hotel_qna[user_input]

    # Second, try finding a question that **contains** the input (less strict matching)
    for question in hotel_qna:
        if user_input in question:
            return hotel_qna[question]

    return None  # No match found


# Function to handle chatbot responses
common_questions = {
    "wifi": "do you have free wifi?",
    "wi-fi": "do you have free wifi?",  # Fix hyphen issue
    "check-in": "what time is check-in?",
    "check-out": "what time is check-out?",
    "cancellation": "what is the cancellation policy?",
    "cancel": "what is the cancellation policy?"  # Handle "cancel" input
}

def chatbot_response(hotel_name, user_input):
    hotels = load_hotel_data()  # Load hotel data from hotels.txt

    # Convert short hotel names (A → Hotel A, B → Hotel B)
    hotel_mapping = {"A": "Hotel A", "B": "Hotel B"}
    hotel_name = hotel_mapping.get(hotel_name.strip().title(), hotel_name.strip().title())

    if hotel_name not in hotels:
        return "Sorry, I don't have information about that hotel."

    # Get hotel's Q&A context from hotels.txt
    hotel_info = hotels[hotel_name]

    # Standardize input question
    cleaned_input = user_input.lower().strip().replace("wi-fi", "wifi").replace("cancel", "cancellation")  # Normalize
    for key in common_questions:
        if key in cleaned_input:
            cleaned_input = common_questions[key]

    # Find best matching Q&A
    answer = find_best_match(cleaned_input, hotel_info)
    if answer:
        return answer

    return "Sorry, I couldn't find an answer."


# Run chatbot in a loop for testing
if __name__ == "__main__":
    print("Hotel Chatbot: Hello! Which hotel do you need information about? (Hotel A / Hotel B)")
    hotel = input("Hotel: ")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        response = chatbot_response(hotel, user_input)
        print("Chatbot:", response)
