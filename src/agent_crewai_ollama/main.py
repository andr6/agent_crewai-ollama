#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from software_house.crew import SoftwareHouseCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """Run the software development crew"""
    inputs = {
        'domain': 'Healthcare SaaS Platform',
        'tech_stack': 'React/Node.js/PostgreSQL',
        'industry': 'HIPAA',
        'current_year': str(datetime.now().year)
    }
    
    try:
        SoftwareHouseCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"Software development crew execution failed: {e}")

def train():
    """Train the development crew with specific iterations"""
    inputs = {
        'domain': 'E-commerce Platform',
        'tech_stack': 'Next.js/Spring Boot/MongoDB',
        'industry': 'PCI-DSS',
        'current_year': str(datetime.now().year)
    }
    try:
        SoftwareHouseCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Failed to train development crew: {e}")

def replay():
    """Replay specific development task"""
    try:
        SoftwareHouseCrew().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"Task replay failed: {e}")

def test():
    """Test crew performance"""
    inputs = {
        'domain': 'IoT Fleet Management',
        'tech_stack': 'React Native/Python/AWS IoT',
        'industry': 'ISO 27001',
        'current_year': str(datetime.now().year)
    }
    try:
        SoftwareHouseCrew().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Development crew testing failed: {e}")

if __name__ == "__main__":
    run()
