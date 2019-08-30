---
How to contribute
---

Third-party modifications, additions and bug fixes are welcomed for improving SeismicReduction.

---

Framework
=========
* __For overly complex methods/functions:__ consider implementing as a function into the utils.py module to remove overly complex code from
the core.py file

* __For small/medium feature additions:__ include directly into the relevant modules already developed rather than create new ones
as these already serve to cover each key component of the analysis workflow.

* __For larger or more complex features which may cover completely new areas:__ consider the use of new modules but implement the control 
of these into the main workflow controlled by the main modules.

* __For implementation of new machine learning/dimensionality reduction models:__ create a new class inheriting from ModelAgent
    * Ensure the model follows the structure of previously implemented model classes.
    * Run the model within an overloaded reduce() method, which modifies the embedding attribute to the model output.
    * Ensure the output format is the expected shape: (number of samples, number of dimensions).
  


Bug Fixes and Improvements
==========================

The easiest way to help is by submitting issues reporting defects or
requesting additional features.

* Make sure you have a GitHub account [or sign up here](https://github.com/signup/free)

* Submit an issue, assuming one does not already exist.

  * Clearly describe the issue including steps to reproduce when it is a bug.
  
  * Make sure you mention the earliest version that you know has the issue.

Making Changes
==============

* Make small commits in logical units.
* Ensure your code is in the spirit of [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Documentation follows the NumPy standards
* Make sure your commit messages are in the proper format::

 ` Issue #1234 - Make the example in CONTRIBUTING imperative and concrete `

* Make sure you have added the necessary tests for your changes.
* Run **all** the tests to assure nothing else was accidentally broken.


Submitting Changes
==================

* Push your changes to a topic branch in your fork of the repository.
* Submit a pull request to the main repository.


Additional Resources
====================
* [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* [General GitHub documentation](http://help.github.com/)
* [GitHub pull request documentation](http://help.github.com/send-pull-requests/)
