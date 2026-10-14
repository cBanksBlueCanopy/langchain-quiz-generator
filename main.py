import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

print('loading ***')
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=1,
    convert_system_message_to_human=True
)
print ('loading ...')

grade_level = input("What grade level are you creating a course for? ")
course = input("What course is this for? ")
number_of_questions = input("How many questions? ")


## Questions
question_template = PromptTemplate.from_template(f"For a {grade_level} grade level {course} course, create a {number_of_questions} question multiple choice exam.")
questions_chain = question_template | model | StrOutputParser()
generated_questions = questions_chain.invoke({
    "grade_level": grade_level,
    "course": course,
    "number_of_questions": number_of_questions
})


## Detailed answers
answer_template = PromptTemplate.from_template(f"Create explanataions for the following questions {generated_questions}, all explanations should be formatted in a manner that a student that is currently in the grade {grade_level} would be able to understand.  Also include the original question above the explanation.")
answer_chain = answer_template | model | StrOutputParser()
generated_answers = answer_template.invoke({
    "generated_questions": generated_questions,
    "grade_level": grade_level
})

print("\nQuiz questions:\n")
print(generated_questions)

print("\n")
print("\nExplanations:\n")
print(generated_answers)

