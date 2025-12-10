Development Process
===================

CFDverify's development process is described here. All regular development to CFD is expected to follow this process to ensure the quality of the developed software.

#. Initiate: Code contributions should be tied to an issue opend on the CFDverify GitHub. This allows code maintainers a chance to review the proposed contributions for consistency with CFDverify's development goals before effort is expended. 
#. Develop: Contributions should be developed with the following attributes in mind to facilitate a speedy review process.
   #. Style: New code shall follow the style of CFDverify. For now, CFDverify style means following `PEP 8`_ for code and `numpydoc`_ for docstrings. 
   #. Documentation: All contributions shall be documented. New code should include in line comments where needed, function and class docstrings, and appropriate Sphinx documentation.
   #. Unit tests: All new code shall have unit tests. The goal of CFDverify is to have 100% coverage, but the precise extent of tests required for new code will be determined on a case-by-case basis.
   #. Regression tests: New methods should add regression tests based on primary literature of the method.
#. Review: Pull requests shall be reviewed by a maintainer. Currently, CFDverify is mainly developed and maintained by a single individual and it is expected he may serve both roles. 
#. Merge: Reviewed and accepted code shall be merged into a development branch of CFDverify.
#. Deploy: Development branches shall be merged into the main branch of CFDverify following semantic versioning 2.0.0 (see :ref:`semantic-versioning-label`).

.. _PEP 8: https://peps.python.org/pep-0008/

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html

.. _semantic-versioning-label:

Semantic Versioning
-------------------

CFDverify uses `semantic versioning <https://semver.org/>`_ for identifying releases. Currently, CFDverify is in initial development, so the public application programming interface (API) may change at any time.