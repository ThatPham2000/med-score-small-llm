from decompose.decomposer import Decomposer
from llm.llm import LLM

if __name__ == '__main__':
    test_llm = LLM(model_name='gpt-oss:20b')
    # print(test_llm.model_name)

    test_decomposer = Decomposer(llm=test_llm)
    print(test_decomposer.llm.model_name)
