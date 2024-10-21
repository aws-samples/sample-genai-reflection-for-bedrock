trigger = """
Using the solutions and reasoning from the other agents as additional information, can you solve this task?
The original task description is {task}.
"""

debate = """
These are the recent/updated opinions from other agents:
"""

reflect = """
Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response.
"""

careful = """
Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.
"""

aggregate = """
You will see proposed answers to the following question from multiple agents, please provide an aggregated final answer.
"""
