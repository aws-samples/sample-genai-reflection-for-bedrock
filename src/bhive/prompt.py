trigger = """
Using the solutions and reasoning from the other agents as additional information, can you solve this task?
The original task description is {task}.
"""

debate = """
These are the recent answers to the same question from other agents:
"""

reflect = """
Can you verify that your answer is correct. Please reiterate your answer by thinking step by step, making sure to state your answer at the end of the response.
"""

careful = """
Use these opinions carefully as additional advice and examining each solution step by step, can you provide an updated answer? Make sure to state your answer at the end of the response.
"""

aggregate = """
You will see proposed answers to the following question from multiple agents, please provide an aggregated final answer.
"""
