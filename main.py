import dspy
from dspy.functional import TypedPredictor
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


class Task(BaseModel):
    "Class for keeping track of a task"
    name: str = Field(..., description="name of the task")
    done: bool = Field(default=False, description="True if the task was done, and False if not")
    result: str = Field(default="", description="results from the execution of the task")


class TaskList(BaseModel):
    list: List[Task] = Field(default=[], description="list of tasks")


class InitiatorAgentSignature(dspy.Signature):
    """Given an objective, returns a list of tasks to fulfill that objective
    """
    objective: str = dspy.InputField(desc="the overall objective to accomplish")
    tasks_list: TaskList = dspy.OutputField(desc="list of current tasks and their status")


class TaskAgentSignature(dspy.Signature):
    """Given an objective and a list of tasks and their status
       Create a new task if necessary or keep current tasks as is.
       Also decide if there are enough elements to provide a final answer to
       user objective
    """
    objective = dspy.InputField(desc="the overall objective to accomplish")
    tasks_list: TaskList = dspy.InputField(desc="list of current tasks and their status")
    add: bool = dspy.OutputField(desc="Whether or not to add a new task is necessary to satisfy objective.")
    new_task: Task = dspy.OutputField(desc="the new task to add to the task list")


class ExecutionAgentSignature(dspy.Signature):
    """Given a task, executes it and return the result
    """
    objective = dspy.InputField(desc="the overall objective to accomplish")
    task: Task = dspy.InputField(desc="a task to execute")
    result: str = dspy.OutputField(desc="a textual report of the results of the task execution. just the result, no mention of the task.")
    stop: bool = dspy.OutputField(desc="True if there is a response to the user objective and we can stop the process")


if __name__ == "__main__":

    load_dotenv()

    lm = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.settings.configure(lm=lm)

    initiator_agent = TypedPredictor(InitiatorAgentSignature)

    task_agent = TypedPredictor(TaskAgentSignature)

    execution_agent = TypedPredictor(ExecutionAgentSignature)

    # pdb.set_trace()

    OBJECTIVE = input("\033[96m\033[1m"+"\n*****Enter the objective of your Baby AGI:*****\n"+"\033[0m\033[0m")

    # Initialize task_id_counter
    task_id_counter = 0

    tasks_list = initiator_agent(objective=OBJECTIVE).tasks_list

    while True:

        if tasks_list and not task_id_counter:
            # Execute all the initial tasks
            task_id_counter = len(tasks_list.list) - 1
            for task in tasks_list.list:
                execution = execution_agent(task=task, objective=OBJECTIVE)
                task.result = execution.result
                task.done = True
                print(task.result)
        else:
            response = task_agent(objective=OBJECTIVE, tasks_list=tasks_list)

            if response.add:
                tasks_list.list.append(response.new_task)

            execution = execution_agent(task=response.new_task, objective=OBJECTIVE)

            tasks_list.list[task_id_counter].result = execution.result
            tasks_list.list[task_id_counter].done = True

            if execution.stop:
                print(f"Final response: {execution.result}\n")
                break
            else:
                print(f"{execution.result}\n")
                task_id_counter += 1
