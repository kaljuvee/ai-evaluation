import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](question=question).query
            passages = self.retrieve(query).passages
            context.extend(passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)

def run_multihop_evaluation():
    """Runs a multi-hop QA evaluation example using DSPy."""

    # Setup LLM and RM
    llm = dspy.OpenAI(model=\"gpt-3.5-turbo\")
    rm = dspy.ColBERTv2(url=\"http://20.102.90.50:2017/wiki17_abstracts\")
    dspy.settings.configure(lm=llm, rm=rm)

    # Load the dataset
    dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
    trainset = [x.with_inputs(\"question\") for x in dataset.train]
    devset = [x.with_inputs(\"question\") for x in dataset.dev]

    # Define the validation logic
    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM

    # Set up the optimizer
    config = dict(max_bootstrapped_demos=2, max_labeled_demos=2)
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer, **config)

    # Compile the program
    optimized_baleen = teleprompter.compile(SimplifiedBaleen(), trainset=trainset)

    # Set up the evaluator
    evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

    # Evaluate the optimized program
    evaluate_on_hotpotqa(optimized_baleen, metric=validate_context_and_answer)

if __name__ == "__main__":
    run_multihop_evaluation()

