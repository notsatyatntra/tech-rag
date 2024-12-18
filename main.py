# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from rag_1 import RAG
from rag_2 import search_wikipedia,rag
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    r = RAG()
    print(r("Washington DC is the capital of which country?"))

    # rag = dspy.ChainOfThought('context, question -> response')
    # question = "Who is the CEO of Tntra software company located in gujarat india?"
    # print(rag(context=search_wikipedia(question), question=question))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
