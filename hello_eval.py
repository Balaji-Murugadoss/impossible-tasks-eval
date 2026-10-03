from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

@task
def hello():
    return Task(
        dataset=[
            Sample(input="What is 17 + 25? Answer with just the number.", target="42"),
            Sample(input="What is the capital of Texas? Answer in one word.", target="Austin"),
        ],
        solver=generate(),
        scorer=includes(),
    )
