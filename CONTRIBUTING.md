Contributing to coreax
======================

Getting started
---------------

If you would like to contribute to the development of coreax, you can do so in a number of ways:
- Highlight any bugs you encounter during usage, or any feature requests that would improve coreax by raising appropriate issues.
- Develop solutions to open issues and create pull requests (PRs) for the development team to review.
- Implement optimisations in the codebase to improve performance.
- Contribute example usages & documentation improvements.
- Increase awareness of coreax with other potential users.

All contributors must sign the [GCHQ Contributor Licence Agreement][cla].

Developers should install additional packages required for development using
`pip install -e .[dev]`. Then, set up pre-commit hooks using `pre-commit install`.

Reporting issues
--------------

- [Search existing issues][github-issues] (both open **and** closed).
- Make sure you are using the latest version of coreax.
- Open a new issue:
  - For bugs use the [bug report issue template][gh-bug-report].
  - For features use the [feature request issue template][gh-feature-request].
  - This will make the issue a candidate for inclusion in future sprints, as-well as open to the community to address.
- If you are able to fix the bug or implement the feature, [create a pull request](#pull-requests) with the relevant changes.

Pull requests
-------------
Currently, we are using a [GitHub Flow][github-flow] development approach.

- To avoid duplicate work, [search existing pull requests][gh-prs].
- All pull requests should relate to an existing issue.
  - If the pull request addresses something not currently covered by an issue, create a new issue first.
- Make changes on a [feature branch][git-feature-branch] instead of the main branch.
- Branch names should take one of the following forms:
  - `feature/<feature-name>`: for adding, removing or refactoring a feature.
  - `bugfix/<bug-name>`: for bug fixes.
- Avoid changes to unrelated files in the same commit.
- Changes must conform to the [code](#code) guidelines.
- Changes must have sufficient [test coverage][run-tests].

### Pull request process
- Create a [Draft pull request][pr-draft] while you are working on the changes to allow others to monitor progress and see the issue is being worked on.
- Pull in changes from upstream often to minimise merge conflicts.
- Make any required changes.
- Resolve any conflicts with the target branch.
- [Change your PR to ready][pr-ready] when the PR is ready for review. You can convert back to Draft at any time.

Do **not** add labels like `[RFC]` or `[WIP]` to the title of your PR to indicate its state.
Non-Draft PRs are assumed to be open for comments; if you want feedback from specific people, `@`-mention them in a comment.

### Pull request commenting process
- Use a comment thread for each required change.
- Reviewer closes the thread once the comment has been resolved.
- Only the reviewer may mark a thread they opened as resolved.

### Commit messages

Follow the [conventional commits guidelines][conventional_commits] to *make reviews easier* and to make the git logs more valuable.
An example commit, including reference to some GitHub issue #123, might take the form:

```
feat: add gpu support for matrix multiplication

If a gpu is available on the system, it is automatically used when performing matrix
multiplication within the code.

BREAKING CHANGE: numpy 1.0.2 no longer supported

Refs: #123
```

Code
------

Code must be documented, adequately tested and compliant with in style prior to merging into the main branch. To
facilitate code review, code should meet these standards prior to creating a pull request.

Some of the following points are checked by pre-commit hooks, although others require
manual implementation by authors and reviewers. Conversely, further style points that
are not documented here are enforced by pre-commit hooks; it is unlikely that authors
need to be aware of them.

### Style

A high level overview of the expected style is:
- Follow [PEP 8][pep-8] style where possible.
- Use clear naming of variables rather than mathematical shorthand (e.g. kernel instead of k).
- [Black][black] will be applied by the pre-commit hook but will not reformat strings,
  comments or docstrings. These must be manually checked and limited to 88 characters
  per line starting from the left margin and including any indentation.
- Avoid using inline comments.
- Type annotations must be used for all function or method parameters.

### Spelling and grammar

This project uses British English. Spelling is checked automatically by [cspell]. When a
word is missing from the dictionary, double check that it is a real word spelled
correctly. Contractions in object or reference names should be avoided unless the
meaning is obvious; consider inserting an underscore to effectively split into two
words. If you need to add a word to the dictionary, use the appropriate dictionary
inside the `.cspell` folder:

- `library-terms.txt` for object names in third-party libraries,
- `people.txt` for the names of people,
- `custom-misc.txt` for anything that does not fit into the above categories.

If the word fragment only makes sense as part of a longer phase, add the longer phrase
to avoid inadvertently permitting spelling errors elsewhere, e.g. add `Blu-Tack`
instead of `Blu`.

### External dependencies
Use standard library and existing well maintained external libraries where possible. New external libraries should be licensed permissive (e.g [MIT][mit]) or weak copyleft (e.g. [LGPL][lgpl]).

### Testing
[Unittest][unittest] is used to write tests for coreax, while [Pytest][pytest] is used to run tests. As much effort should
be put into developing tests as the code. Tests should be provided to test functionality and also ensuring exceptions and
warnings are raised or managed appropriately. This includes:
- Unit testing of new functions added to the codebase
- Verifying all existing tests pass with the integrated changes

Keep in mind the impact on runtime when writing your tests. Favour more tests that are smaller rather than a few large
tests with many assert statements unless it would significantly affect run time, e.g. due to excess set up or duplicated
function calls.

Use the form: (actual, expected) in asserts, e.g.
```python
assertEqual(actualValue, expectedValue)
```

### Abstract functions
Abstract methods, functions and properties should only contain a docstring. They should not contain a `pass` statement.

### Exceptions and error messages
Custom exceptions should be derived from the most specific relevant Exception class. Custom messages should be succinct and, where easy to implement, offer suggestions to the user on how to rectify the exception.

Avoid stating how the program will handle the error, e.g. avoid Aborting, since it will be evident that the program has terminated. This enables the exception to be caught and the program to continue in the future.

### Docstrings

Docstrings must:
- Be written for private functions, methods and classes where their purpose or usage is not immediately obvious.
- Be written in [reStructured Text][sphinx-rst] ready to be compiled into documentation via [Sphinx][sphinx].
- Follow the [PEP 257][pep-257] style guide.
- Not have a blank line inserted after a function or method docstring unless the following statement is a function, method or class definition.
- Start with a capital letter unless referring to the name of an object, in which case match that case sensitively.
- Have a full stop at the end of the one-line descriptive sentence.
- Use full stops in extended paragraphs of text.
- Not have full stops at the end of parameter definitions.
- If a `:param:` or similar line requires more than the max line length, use multiple lines. Each additional line should
  be indented by a further 4 spaces.
- Class `__init__` methods should not have docstrings. All constructor parameters should be listed at the end of the class
  docstring. `__init__` docstrings will not be rendered by Sphinx. Any developer comments should be contained in a regular
  comment.

Each docstring for a public object should take the following structure:
```
"""
Write a one-line descriptive sentence as an active command.

As many paragraphs as is required to document the object.

:param a: Description of parameter a
:param b: Description of parameter b
:raises SyntaxError: Description of why a SyntaxError might be raised
:return: Description of return from function
"""
```
If the function does not return anything, the return line above can be omitted.

### Comments

Comments must:
- Start with a capital letter unless referring to the name of an object, in which case match that case sensitively.
- Not end in a full stop for single-line comments in code.
- End with a full stop for multi-line comments.

### Maths overflow

Prioritise overfull lines for mathematical expressions over artificially splitting them into multiple equations in both comments and docstrings.

### Thousands separators

For hardcoded integers >= 1000, an underscore should be written to separate the thousands, e.g. 10_000 instead of 10000.

### Documentation and references
The coreax documentation should reference papers and mathematical descriptions as appropriate. New references should be placed in the [`references.bib`](references.bib) file. An entry with key word `RefYY` can then be referenced within a docstring anywhere with `[RefYY]_`.

### Generating docs with Sphinx

You can generate Sphinx documentation with:
```sh
documentation/make html
```

[github-issues]: https://github.com/gchq/coreax/issues?q=
[gh-bug-report]: https://github.com/gchq/coreax/issues/new?assignees=&labels=bug%2Cnew&projects=&template=bug_report.yml&title=%5BBug%5D%3A+
[gh-feature-request]: https://github.com/gchq/coreax/issues/new?assignees=&labels=enhancement%2Cnew&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+
[gh-prs]: https://github.com/gchq/coreax/pulls?q=
[run-tests]: https://github.com/gchq/coreax/actions/workflows/unittests.yml
[conventional_commits]: https://www.conventionalcommits.org
[git-feature-branch]: https://www.atlassian.com/git/tutorials/comparing-workflows
[pr-draft]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[pr-ready]: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request
[pep-8]: https://peps.python.org/pep-0008/
[black]: https://black.readthedocs.io/en/stable/
[sphinx-rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[pep-257]: https://peps.python.org/pep-0257/
[cla]: https://cla-assistant.io/gchq/coreax
[github-flow]: https://docs.github.com/en/get-started/quickstart/github-flow
[mit]: https://opensource.org/license/mit/
[lgpl]: https://opensource.org/license/lgpl-license-html/
[unittest]: https://docs.python.org/3/library/unittest.html
[pytest]: https://docs.pytest.org/
[cspell]: https://cspell.org/
