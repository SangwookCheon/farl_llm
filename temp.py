import os
from openai import OpenAI
from dotenv import load_dotenv

def load_input_text(file_path: str) -> str:
    """Read the entire contents of a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    # Load .env file so OPENAI_API_KEY becomes available
    load_dotenv()

    # 1. Set your file path here
    input_file = "input/sample1.txt"

    # 2. Load the text that will go into the prompt
    patient_text = load_input_text(input_file)

    # 3. Initialize OpenAI client (uses OPENAI_API_KEY from env)
    client = OpenAI()

    # 4. Build a simple prompt
    system_prompt = (
        "You are a careful diabetes lifestyle coach. "
        "You ONLY give lifestyle and behaviour recommendations, "
        "not medication changes. "
        "Be realistic, concrete, and supportive."
    )

    user_prompt = (
        "Below is the information about a patient and an action plan suggested "
        "by a reinforcement learning model. "
        "Using this, write a short, feasible recommendation for the patient.\n\n"
        f"{patient_text}\n\n"
        "Now write your recommendation:"
    )

    # 5. Call the Chat Completions API
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    # 6. Extract and print the model's reply
    reply = response.choices[0].message.content
    print("=== MODEL RECOMMENDATION ===")
    print(reply)

if __name__ == "__main__":
    main()