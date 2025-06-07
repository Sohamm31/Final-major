from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
import speech_recognition as sr
import threading
import pyttsx3
import time
import os
from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv('OPENROUTESERVICE_API_KEY')
recognizer = sr.Recognizer()
tts = pyttsx3.init()

tts.setProperty('rate', 180)  
voices = tts.getProperty('voices')
tts.setProperty('volume', 1.0)
tts.setProperty('voice', voices[117].id)  


loader = PyPDFLoader("Final-major/Soham Resume TnP Updated May.pdf")
pdf = loader.load()
all_content = [doc.page_content for doc in pdf]

schema = [
    ResponseSchema(name="question-1",description="Question-1 on resume"),
    ResponseSchema(name="question-2",description="Question-2 on resume"),
    ResponseSchema(name="question-3",description="Question-3 on resume"),
    ResponseSchema(name="question-4",description="Question-4 on resume"),
    ResponseSchema(name="question-5",description="Question-5 on resume")
]
intro_schema = [
    ResponseSchema(name="question-1",description="Question-1 on Introduction"),
    ResponseSchema(name="question-2",description="Question-2 on Introduction"),
]
feedback_schema = [
    ResponseSchema(name="Suggestion-1",description="Suggestion-1 on interview"),
    ResponseSchema(name="Suggestion-2",description="Suggestion-2 on interview"),
    ResponseSchema(name="Suggestion-3",description="Suggestion-3 on interview"),
    ResponseSchema(name="Suggestion-4",description="Suggestion-4 on interview"),
    ResponseSchema(name="Suggestion-5",description="Suggestion-5 on interview")
]
introduction_parser = StructuredOutputParser.from_response_schemas(intro_schema)

parser = StructuredOutputParser.from_response_schemas(schema)

feedback_parser =  StructuredOutputParser.from_response_schemas(feedback_schema)

categorized_feedback_schema = [
    ResponseSchema(name="Communication", description="Rate the candidate's communication skills on a scale of 1 to 5."),
    ResponseSchema(name="Confidence", description="Rate the candidate's confidence and body language on a scale of 1 to 5."),
    ResponseSchema(name="Technical", description="Rate the candidate's technical knowledge and responses on a scale of 1 to 5."),
    ResponseSchema(name="Resume Fit", description="Rate how well the candidate fits the resume on a scale of 1 to 5."),
    ResponseSchema(name="Improvement Areas", description="Rate the candidate's overall areas for improvement on a scale of 1 to 5.")
]
categorized_feedback_parser = StructuredOutputParser.from_response_schemas(categorized_feedback_schema)

# prompt = PromptTemplate(
#     template= "Attached here is a resume.Based on resume frame 5 interview questions.Only questions are to be formed without answers in a proper format.{Resume}",
#     input_variables=["Resume"]
# )

prompt = PromptTemplate(
    template="Give 5 interview questions based on the attached resume following is the attached resume./n{Resume}/n{format_instruction}",
    input_variables=["Resume"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)
intro_prompt = PromptTemplate(
    template="Based on the Introduction form 2 questions for the interview./n{introduction}/n{format_instruction}",
    input_variables=["introduction"],
    partial_variables={"format_instruction": introduction_parser.get_format_instructions()}
)
feedback_prompt = PromptTemplate(
    template="You are an HR assistant. Based on the following interview responses, provide personalized feedback in 5 points./n{interview_summary}/n{format_instruction}",
    input_variables=["interview_summary"],
    partial_variables={"format_instruction": feedback_parser.get_format_instructions()}
)
categorized_feedback_prompt = PromptTemplate(
    template="You are an HR assistant. Based on the following interview responses, provide categorized feedback in the following sections: Communication, Confidence, Technical, Resume Fit, Improvement Areas.\n{interview_summary}\n{format_instruction}",
    input_variables=["interview_summary"],
    partial_variables={"format_instruction": categorized_feedback_parser.get_format_instructions()}
)



llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1-0528-qwen3-8b:free",  
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1", 
)



chain  = prompt | llm | parser
questions = chain.invoke({"Resume":all_content})
responses = {}

def speak(text, pause=2):
    tts.say(text)
    tts.runAndWait()
    time.sleep(pause)

def listen():
    print("Listening... Press Enter when you're done speaking.")
    audio_data = sr.AudioData(b'', 16000, 2)
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        collected_audio = []
        stop_listening = False
        while not stop_listening:
            try:
                audio_chunk = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                collected_audio.append(audio_chunk)
            except sr.WaitTimeoutError:
                continue
            if input() == '':
                stop_listening = True

        print("Transcribing...")
        try:
            full_audio = sr.AudioData(b''.join(chunk.get_raw_data() for chunk in collected_audio),
                                      collected_audio[0].sample_rate,
                                      collected_audio[0].sample_width)
            return recognizer.recognize_google(full_audio)
        except Exception as e:
            return f"Error in transcription: {e}"

def generate_summary(responses_dict, questions_dict):
    summary_lines = []
    for key in questions_dict:
        question = questions_dict[key]
        answer = responses_dict.get(key, "No response.")
        summary_lines.append(f"Q: {question}\nA: {answer}\n")
    return "\n".join(summary_lines)

speak("Introduce Yourself Please")
introduction = listen()

introduction_chain = intro_prompt | llm | introduction_parser
intro_questions = introduction_chain.invoke({"introduction":introduction})

for key, question in intro_questions.items():
    print(f"\nQuestion: {question}")
    speak(question, pause=3)

    print("Your response:")
    intro_answer = listen()
    print(f"You said: {intro_answer}")
    responses[key] = intro_answer



for key, question in questions.items():
    print(f"\nQuestion: {question}")
    speak(question, pause=3)

    print("Your response:")
    answer = listen()
    print(f"You said: {answer}")
    responses[key] = answer

summary_text = generate_summary(responses, questions)
print("\nInterview Summary:\n")
print(summary_text)
print("\nInterview Summary:\n")
print(summary_text)



summary_chain = feedback_prompt | llm | feedback_parser

interview_review = summary_chain.invoke({"interview_summary":summary_text})




categorized_feedback_chain = categorized_feedback_prompt | llm | categorized_feedback_parser
categorized_feedback = categorized_feedback_chain.invoke({"interview_summary": summary_text})

speak(str(interview_review))

print("\nCategorized Feedback:\n")
for category, feedback in categorized_feedback.items():
    print(f"{category}: {feedback}\n")
