.. _contributing:

Contributing
===========

We welcome contributions to the Nenya project! This page provides guidelines for contributing to the codebase.

Getting Started
-------------

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a branch** for your changes
4. **Make your changes** following the coding guidelines
5. **Write tests** for your changes
6. **Submit a pull request**

Development Setup
---------------

To set up a development environment:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/nenya.git
   cd nenya
   
   # Install in development mode
   pip install -e ".[dev]"
   
   # Run tests
   pytest

Coding Guidelines
---------------

1. **Follow PEP 8** for code style
2. **Use type hints** for function parameters and return values
3. **Document your code** with docstrings following NumPy/Google style
4. **Write clear commit messages**
5. **Keep functions focused** on a single task
6. **Use meaningful variable names**

Documentation
-----------

All new features should include:

1. **Docstrings** for all functions, classes, and methods
2. **Examples** showing how to use the feature
3. **Updates to the appropriate .rst files** in the documentation

Example docstring format:

.. code-block:: python

   def my_function(param1, param2=None):
       """Brief description of the function.
       
       More detailed description of what the function does.
       
       Args:
           param1 (type): Description of param1
           param2 (type, optional): Description of param2. Defaults to None.
           
       Returns:
           type: Description of return value
           
       Raises:
           ExceptionType: When and why this exception is raised
           
       Example:
           >>> result = my_function('value', param2='other_value')
           >>> print(result)
           Expected output
       """
       # Function implementation...

Testing
------

We use pytest for testing. Tests should be placed in the `tests/` directory.

1. **Unit tests** should test individual components
2. **Integration tests** should test interactions between components
3. **Test both success cases and error cases**
4. **Use fixtures** where appropriate

Example test:

.. code-block:: python

   def test_my_function():
       # Setup
       input_value = 'test'
       
       # Exercise
       result = my_function(input_value)
       
       # Verify
       assert result == expected_result
       
       # Teardown - if needed

Pull Request Process
------------------

1. **Update documentation** if needed
2. **Ensure all tests pass**
3. **Add yourself** to the list of contributors if you're not already there
4. **Submit the PR** with a clear description of the changes and their purpose

Code Review
----------

Pull requests will be reviewed based on:

1. **Functionality**: Does it work as expected?
2. **Code quality**: Is the code clean, well-structured, and maintainable?
3. **Tests**: Are there adequate tests for the changes?
4. **Documentation**: Is the documentation updated to reflect the changes?

Versioning
---------

We follow semantic versioning:

- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

License
------

By contributing to Nenya, you agree that your contributions will be licensed under the same license as the project.

Questions?
---------

If you have questions about contributing, please open an issue on GitHub or contact the maintainers directly.
