# Contributing

Contributions are welcome and very much appreciated!

## Code contributions

We accept code contributions through pull requests.
In short, this is how that works.

1. Fork [the repository](https://github.com/BlockResearchGroup/compas_singular) and clone the fork.
2. Create the development environment with conda:

   ```bash
   conda env create -f environment.yml
   conda activate singular-dev
   ```

   or install the development dependencies into an environment of your choice:

   ```bash
   pip install -e ".[dev]"
   ```

3. Check the environment by running `invoke`, which lists the available tasks.

4. Make sure all tests pass:

   ```bash
   invoke test
   ```

5. Start making your changes to the **master** branch (or branch off of it).
6. Make sure all tests still pass:

   ```bash
   invoke test
   ```

7. Add yourself to the *Contributors* section of `AUTHORS.md`.
8. Commit your changes and push your branch to GitHub.
9. Create a [pull request](https://help.github.com/articles/about-pull-requests/) through the GitHub website.

During development, use [pyinvoke](http://docs.pyinvoke.org/) tasks on the
command line to ease recurring operations:

* `invoke clean`: Clean all generated artifacts.
* `invoke lint`: Check the code style with ruff (`invoke format` reformats).
* `invoke docs`: Build the documentation (MkDocs) into `dist/docs`; `mkdocs serve` gives a live preview.
* `invoke test`: Run all tests.
* `invoke release patch|minor|major`: Bump the version, tag it, build, and push (asks first).
* `invoke`: Show available tasks.

## Bug reports

When [reporting a bug](https://github.com/BlockResearchGroup/compas_singular/issues) please include:

* Operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Feature requests

When [proposing a new feature](https://github.com/BlockResearchGroup/compas_singular/issues) please include:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
