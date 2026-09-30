# Releasing uqcalibrate

How a version gets from `main` to PyPI. The rules are in TECH-STACK.md section 8; this page is
the click-by-click. Nothing is ever uploaded without Carol's approval: GitHub holds every upload
job until she approves it.

## Part 1: one-time setup (about 10 minutes)

Do these in order. The GitHub environments must exist, with the reviewer set, **before** the
publish workflow first runs; otherwise GitHub creates them on the fly without the approval step.

**1. TestPyPI pending publisher.** Log in at https://test.pypi.org, open *Your account*, then
*Publishing*, and under *Add a new pending publisher* choose the *GitHub* tab:

| Field | Value |
|---|---|
| PyPI project name | `uqcalibrate` |
| Owner | `cgudumotu` |
| Repository name | `uq` |
| Workflow name | `publish.yml` (the file name only) |
| Environment name | `testpypi` |

**2. PyPI pending publisher.** The same at https://pypi.org, with environment name `pypi`.
A pending publisher does not reserve the name; the first upload does.

**3. GitHub environments.** In the repository: *Settings*, *Environments*, *New environment*.

- `testpypi`: under *Deployment protection rules* tick *Required reviewers*, add `cgudumotu`,
  leave *Prevent self-review* unticked, and click *Save protection rules*.
- `pypi`: the same. Recommended extra: under *Deployment branches and tags* choose *Selected
  branches and tags* and add the tag rule `v*`, so only a release tag can ever reach PyPI.

## Part 2: dry run on TestPyPI

TestPyPI is a separate practice index. Uploads there are real uploads (a version number can
never be reused there either), but nobody installs from it by accident.

1. Push `main`. In the *Actions* tab, open *publish* and click *Run workflow* on `main`.
2. The *build* job builds the wheel and source distribution and runs `twine check`.
3. The *testpypi* job waits with *Review deployments*. Click it, tick `testpypi`, *Approve and
   deploy*.
4. Check the page: https://test.pypi.org/project/uqcalibrate/ (README rendering, links, author,
   license, Python versions).
5. Install it in a fresh environment. TestPyPI does not carry NumPy reliably, so NumPy comes
   from PyPI:

   ```
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ uqcalibrate
   python -c "import uqcalibrate as uqc; print(uqc.__version__, len(uqc.__all__))"
   ```

   Expected: `0.1.0 9`.

## Part 3: the real release

Checklist first (TECH-STACK section 8):

- [ ] Carol has reviewed the code line by line.
- [ ] CI is green on the commit to release.
- [ ] `__version__` in `src/uqcalibrate/__init__.py` is the version to release.
- [ ] CHANGELOG.md: the version's heading carries today's date instead of "Unreleased".
- [ ] The TestPyPI page for this version looked right.

Then:

1. Commit and push the dated CHANGELOG.
2. On GitHub: *Releases*, *Draft a new release*. *Choose a tag*: type `v0.1.0` and create it on
   `main`. Title `0.1.0`. Paste the CHANGELOG section as the notes. *Publish release*.
3. The *publish* workflow starts. The build job stops if the tag does not match `__version__`.
4. Approve `testpypi` (it skips files already uploaded by the dry run), then approve `pypi`.
5. Check https://pypi.org/project/uqcalibrate/ and `pip install uqcalibrate` in a fresh
   environment.

## If something goes wrong

- **A version number can never be uploaded twice**, on PyPI or TestPyPI, even after deleting
  it. Fix the problem, bump the version (0.1.0 to 0.1.1), add a CHANGELOG entry and release
  again.
- **A bad release is yanked, not deleted.** *Yank* (in the project's PyPI settings) hides the
  version from `pip install uqcalibrate` while keeping it installable for anyone who pinned it
  exactly.
- **"Invalid publisher" or "not trusted"** from the upload step means the pending publisher's
  fields differ from reality: owner, repository, workflow file name and environment name must
  match exactly.
