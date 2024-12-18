#Importing the libraries
import dspy
from dspy.retrieve.faiss_rm import FaissRM

#Data to feed
document_chunks = [
    "The superbowl this year was played between the San Francisco 49ers and the Kanasas City Chiefs",
    "Pop corn is often served in a bowl",
    "My name is Shruti",
    "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
    "Mars is the fourth planet in the Solar System",
    "An aquarium is a place where children can learn about marine life",
    "The capital of the United States is Washington, D.C",
    "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
    "Music albums were published on Long Play Records in the 70s and 80s",
    "Sichuan cuisine is a spicy cuisine from central China",
    "The creator of this model is Krishan walia",
    "The interest rates for mortgages is considered to be very high in 2024",
]

#Creating RM
frm = FaissRM(document_chunks)

#Configuring The LM and RM
# gemini = dspy.Google(model='gemini-1.5-flash', api_key="<YOUR_API_KEY>", temperature=0.3)
gemini = dspy.OllamaLocal(model='llama3')
dspy.settings.configure(lm=gemini, rm=frm)

#Creating the Signature
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers.
    And if the context is not related to the question being asked, reply that you dont have relevant information."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Creating the RAG Module
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#Querying the RAG agent
# r = RAG()
# r("What is the capital of the United States?")