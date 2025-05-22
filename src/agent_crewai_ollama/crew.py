from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class SoftwareHouseCrew():
    """Software House Development Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def technical_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_lead'],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def full_stack_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['full_stack_engineer'],
            verbose=True
        )

    @agent
    def qa_automation_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_automation_engineer'],
            verbose=True
        )

    @agent
    def devops_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['devops_engineer'],
            verbose=True
        )

    @agent
    def ux_technical_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['ux_technical_specialist'],
            verbose=True
        )

    @task
    def technical_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['technical_design_task'],
            output_file='technical_spec.md'
        )

    @task
    def feature_development_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_development_task'],
            output_file='feature_implementation'
        )

    @task
    def quality_assurance_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_task'],
            output_file='quality_report.md'
        )

    @task
    def deployment_pipeline_task(self) -> Task:
        return Task(
            config=self.tasks_config['deployment_pipeline_task'],
            output_file='pipeline_config'
        )

    @task
    def ux_implementation_task(self) -> Task:
        return Task(
            config=self.tasks_config['ux_implementation_task'],
            output_file='ux_package'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Software House Development Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=2,
            memory=True,
            max_iter=15
        )
