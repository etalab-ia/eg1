from sqlalchemy.orm import Session

class TestApi:
    """Base test class that all test classes should inherit from."""
    pass  # Database setup is now handled by fixtures in conftest.py