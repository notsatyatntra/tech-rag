import dspy


def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]
gemini = dspy.OllamaLocal(model='llama3')
dspy.settings.configure(lm=gemini)
rag = dspy.ChainOfThought('context, question -> response')

# question = "What's the name of the castle that David Gregory inherited?"
# print(rag(context=search_wikipedia(question), question=question))